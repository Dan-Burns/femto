"""Utilities for interacting with MPI and CUDA MPS."""

import contextlib
import functools
import logging
import os
import pathlib
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import typing

import GPUtil
import numpy

_LOGGER = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from mpi4py import MPI


_K = typing.TypeVar("_K", bound=str | int)
_T = typing.TypeVar("_T")


_REDUCE_DICT_OP = None
_INSIDE_MPI_COMM = False


def is_rank_zero() -> bool:
    """Returns true if the current MPI rank is zero, or if the application is not
    running using MPI."""

    mpi_env_vars = {"PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE"}

    if all(env_var not in os.environ for env_var in mpi_env_vars):
        return True

    from mpi4py import MPI

    return MPI.COMM_WORLD.rank == 0


@contextlib.contextmanager
def get_mpi_comm() -> typing.ContextManager["MPI.Intracomm"]:
    """A context manager that returns the main MPI communicator and installs signal
    handlers to abort MPI on exceptions.

    The signal handlers are restored to their defaults when the context manager exits.

    Returns:
        The global MPI communicator.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    global _INSIDE_MPI_COMM

    if _INSIDE_MPI_COMM:
        yield comm
        return

    _INSIDE_MPI_COMM = True

    original_signal_handlers = {
        signal_code: signal.getsignal(signal_code)
        for signal_code in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]
    }

    def abort_comm():
        if comm.size > 1:
            _LOGGER.warning("Aborting MPI")
            comm.Abort(1)

    def abort_comm_handler(signal_code, _):
        abort_comm()

        signal.signal(signal_code, original_signal_handlers[signal_code])
        signal.raise_signal(signal_code)

    try:
        for signal_code in original_signal_handlers:
            signal.signal(signal_code, abort_comm_handler)

        yield comm
    except BaseException as e:
        _LOGGER.exception(e)
        abort_comm()
        raise e
    finally:
        for signal_code in original_signal_handlers:
            signal.signal(signal_code, original_signal_handlers[signal_code])

        _INSIDE_MPI_COMM = False


def _reduce_dict_fn(dict_1: dict[str, float], dict_2: dict[str, float], _):
    """Sum the values of two dictionaries with the same keys."""
    for k, v in dict_2.items():
        if k not in dict_1:
            dict_1[k] = v
        else:
            dict_1[k] += v
    return dict_1


def reduce_dict(
    value: dict[_K, _T], mpi_comm: "MPI.Intracomm", root: int | None = None
) -> dict[_K, _T]:
    """Reduce a dictionary of values across MPI ranks.

    Args:
        value: The dictionary of values to reduce.
        mpi_comm: The MPI communicator to use for the reduction.
        root: The rank to which the reduced dictionary should be sent. If None, the
            reduced dictionary will be broadcast to all ranks.

    Returns:
        The reduced dictionary of values.
    """
    import mpi4py.MPI

    global _REDUCE_DICT_OP

    if _REDUCE_DICT_OP is None:
        _REDUCE_DICT_OP = mpi4py.MPI.Op.Create(_reduce_dict_fn, commute=True)

    if root is not None:
        return mpi_comm.reduce({**value}, op=_REDUCE_DICT_OP, root=root)
    else:
        return mpi_comm.allreduce({**value}, op=_REDUCE_DICT_OP)


def divide_tasks(mpi_comm: "MPI.Intracomm", n_tasks: int) -> tuple[int, int]:
    """Determine how many tasks the current MPI process should run given the total
    number that need to be distributed across all ranks.

    Args:
        mpi_comm: The main MPI communicator.
        n_tasks: The total number of tasks to run.

    Returns:
        The number of tasks to run on the current MPI process, and the index of the
        first task to be run by this worker.
    """
    n_workers = mpi_comm.size
    worker_idx = mpi_comm.rank

    n_each, n_extra = divmod(n_tasks, n_workers)

    replica_idx_offsets = numpy.array(
        [0] + n_extra * [n_each + 1] + (n_workers - n_extra) * [n_each]
    )
    replica_idx_offset = replica_idx_offsets.cumsum()[worker_idx]

    n_replicas = n_each + 1 if worker_idx < n_extra else n_each

    hostname = socket.gethostname()
    _LOGGER.debug(
        f"hostname={hostname} rank={mpi_comm.rank} will run {n_replicas} replicas"
    )

    return n_replicas, replica_idx_offset


def _visible_devices() -> list[str]:
    """Return the ids of the CUDA devices visible to this process."""

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [d for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d]

    try:
        return [str(gpu.id) for gpu in GPUtil.getGPUs()]
    except Exception:
        return []


def is_inside_mpi() -> bool:
    """Check if the current process was launched under an MPI runtime."""

    mpi_env_vars = {"PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE"}
    return any(v in os.environ for v in mpi_env_vars)


def _mps_pid_file() -> pathlib.Path:
    """The pid file the MPS control daemon writes into its pipe directory."""
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    return pathlib.Path(pipe_dir, "nvidia-cuda-mps-control.pid")


def is_mps_running() -> bool:
    """Check if a CUDA MPS control daemon is serving ``CUDA_MPS_PIPE_DIRECTORY``.

    Reads the daemon's pid file rather than querying the interactive control
    pipe, which can hang on newer NVIDIA drivers (580+).
    """

    try:
        os.kill(int(_mps_pid_file().read_text()), 0)
    except PermissionError:
        return True  # alive, but owned by another user
    except (OSError, ValueError):
        return False

    return True


def start_mps() -> None:
    """Start a CUDA MPS control daemon on ``CUDA_MPS_PIPE_DIRECTORY``.

    Raises:
        RuntimeError: If ``nvidia-cuda-mps-control`` is not found or fails to start.
    """

    if not shutil.which("nvidia-cuda-mps-control"):
        raise RuntimeError(
            "nvidia-cuda-mps-control not found. Ensure the NVIDIA CUDA toolkit "
            "is installed and on your PATH."
        )

    if is_mps_running():
        _LOGGER.info("CUDA MPS daemon is already running")
        return

    _LOGGER.info("Starting CUDA MPS daemon")
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)

    # wait for the daemon to become ready before any clients attach
    for _ in range(100):
        if is_mps_running():
            return
        time.sleep(0.1)

    raise RuntimeError("CUDA MPS control daemon did not start")


def stop_mps() -> None:
    """Stop the CUDA MPS daemon."""

    _LOGGER.info("Stopping CUDA MPS daemon")

    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        check=False,
        timeout=10,
    )


@contextlib.contextmanager
def mps_context():
    """Context manager that starts the MPS daemon on entry and stops it on exit.

    If MPS is already running when the context is entered, it will not be stopped
    on exit.
    """

    already_running = is_mps_running()

    if not already_running:
        start_mps()

    try:
        yield
    finally:
        if not already_running:
            stop_mps()


@contextlib.contextmanager
def node_mps(mpi_comm: "MPI.Intracomm"):
    """Run one CUDA MPS control daemon per node while more ranks than GPUs share it.

    The first rank on each host starts a daemon on a fresh pipe directory, and stops
    it again on exit, so only daemons started here are ever shut down. If
    ``CUDA_MPS_PIPE_DIRECTORY`` is already set (e.g. a site managed daemon), that
    daemon is used as-is. Must be entered before any CUDA context is created.
    """
    from mpi4py import MPI

    node_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    n_devices = len(_visible_devices())

    if n_devices == 0 or node_comm.size <= n_devices:
        yield
        return

    ranks_per_gpu = -(-node_comm.size // n_devices)
    os.environ.setdefault(
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", str(max(1, 200 // ranks_per_gpu))
    )

    owned = "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
    root_dir, error = None, None

    if owned and node_comm.rank == 0:
        # the pipe socket path must stay short, so avoid a (long) $TMPDIR
        root_dir = tempfile.mkdtemp(prefix="femto-mps-", dir="/tmp")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = os.path.join(root_dir, "pipe")
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = os.path.join(root_dir, "log")
        try:
            start_mps()
        except Exception as e:
            error = f"{socket.gethostname()}: {e}"

    if owned:
        root_dir, error = node_comm.bcast((root_dir, error), root=0)
        if error is not None:
            raise RuntimeError(f"failed to start CUDA MPS on {error}")

        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = os.path.join(root_dir, "pipe")
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = os.path.join(root_dir, "log")

    try:
        yield
    finally:
        if owned:
            node_comm.Barrier()
            if node_comm.rank == 0:
                # 'quit' blocks until every client (including this process) has
                # disconnected, so let it finish in the background after we exit
                subprocess.Popen(
                    "echo quit | nvidia-cuda-mps-control; "
                    f"rm -rf {shlex.quote(root_dir)}",
                    shell=True,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


def _strip_oversubscribe_from_argv(argv: list[str]) -> list[str]:
    """Remove ``--oversubscribe`` and its value from an argv list."""

    result = []
    skip_next = False

    for arg in argv:
        if skip_next:
            skip_next = False
            continue

        if arg in ("--oversubscribe", "-o"):
            skip_next = True
            continue

        if arg.startswith("--oversubscribe=") or arg.startswith("-o="):
            continue

        result.append(arg)

    return result


def launch_with_mps(
    oversubscribe: int,
    mpi_command: str | None = None,
) -> int:
    """Re-launch the current command under MPI with CUDA MPS enabled.

    This function detects the available GPUs, starts the MPS daemon, sets
    ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``, and re-execs the current command
    under ``mpirun`` with ``n_gpus * oversubscribe`` ranks.

    Args:
        oversubscribe: The number of MPI ranks to place on each GPU.
        mpi_command: The MPI launcher command (e.g. ``"mpirun"`` or
            ``"srun --mpi=pmix"``). If ``None``, ``mpirun`` or ``mpiexec``
            will be auto-detected.

    Returns:
        The return code of the MPI process.

    Raises:
        RuntimeError: If no CUDA devices or MPI launcher can be found.
    """

    n_gpus = len(_visible_devices())

    if n_gpus == 0:
        raise RuntimeError(
            "No CUDA devices found. Cannot launch with MPS oversubscription."
        )

    n_ranks = n_gpus * oversubscribe
    thread_pct = max(1, 200 // oversubscribe)

    child_argv = _strip_oversubscribe_from_argv(sys.argv)
    
    if child_argv and child_argv[0].endswith(".py"):
        child_argv = [sys.executable, *child_argv]

    if mpi_command is not None:
        mpi_cmd = shlex.split(mpi_command) + ["-n", str(n_ranks)]
    else:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")

        if mpirun is None:
            raise RuntimeError(
                "Cannot find mpirun or mpiexec on PATH. Please install an MPI "
                "implementation (e.g. Open MPI) or specify --mpi-command."
            )

        mpi_cmd = [mpirun, "-n", str(n_ranks), "--oversubscribe"]

    env = {**os.environ, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(thread_pct)}

    _LOGGER.info(
        f"Launching with CUDA MPS: {n_gpus} GPUs × {oversubscribe} ranks/GPU = "
        f"{n_ranks} total ranks, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={thread_pct}%"
    )

    with mps_context():
        result = subprocess.run([*mpi_cmd, *child_argv], env=env)

    return result.returncode


def divide_gpus():
    """Attempts to divide the GPUs visible on each node across the MPI ranks running
    on that node. If there are more ranks than GPUs, then each GPU will be assigned
    to multiple ranks.
    """
    import mpi4py.MPI

    hostname = socket.gethostname()

    with get_mpi_comm() as mpi_comm:
        node_comm = mpi_comm.Split_type(mpi4py.MPI.COMM_TYPE_SHARED)
        devices = _visible_devices()

        if len(devices) > 0:
            local_idx = node_comm.rank % len(devices)
            ranks_per_gpu = -(-node_comm.size // len(devices))
            use_mps = ranks_per_gpu > 1 and is_mps_running()

            # MPS clients see devices renumbered relative to the daemon's visible list
            device_id = str(local_idx) if use_mps else devices[local_idx]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id

            _LOGGER.debug(
                f"hostname={hostname} "
                f"rank={mpi_comm.rank} will use GPU={device_id} "
                f"({ranks_per_gpu} ranks/GPU)"
            )

            if ranks_per_gpu > 1 and not use_mps:
                _LOGGER.warning(
                    f"Multiple MPI ranks ({ranks_per_gpu}) are sharing each GPU "
                    f"but CUDA MPS does not appear to be running. Performance "
                    f"will be degraded due to context switching. Start MPS with: "
                    f"nvidia-cuda-mps-control -d"
                )

        else:
            _LOGGER.debug(f"hostname={hostname} has no GPUs")


def run_on_rank_zero(func):
    """A convenient decorator that ensures the function is only run on rank zero and
    that the outputs are broadcast to the other ranks.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        outputs = None
        rank = 0

        with get_mpi_comm() as mpi_comm:
            if mpi_comm.rank == rank:
                outputs = func(*args, **kwargs)

            outputs = mpi_comm.bcast(outputs, root=rank)

        return outputs

    return wrapper
