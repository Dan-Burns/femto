import os
import shutil
import signal

import pytest

import femto.md.utils.mpi


def test_is_rank_zero_not_mpi(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    assert femto.md.utils.mpi.is_rank_zero()


@pytest.mark.parametrize("rank, expected", [(0, True), (1, False)])
def test_is_rank_zero(rank, expected, mocker):
    import mpi4py.MPI

    mocker.patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "1"}, clear=True)
    mocker.patch.object(mpi4py.MPI, "COMM_WORLD", mocker.MagicMock(rank=rank))

    assert femto.md.utils.mpi.is_rank_zero() == expected


def test_get_mpi_comm_nested(mocker):
    """Only the top level ctx manager should set the signals"""
    spied_signal = mocker.spy(signal, "getsignal")

    with femto.md.utils.mpi.get_mpi_comm():
        assert spied_signal.call_count == 3  # int term abrt
        assert femto.md.utils.mpi._INSIDE_MPI_COMM is True

        with femto.md.utils.mpi.get_mpi_comm():
            assert spied_signal.call_count == 3

        assert femto.md.utils.mpi._INSIDE_MPI_COMM is True

    assert femto.md.utils.mpi._INSIDE_MPI_COMM is False


def test_get_mpi_comm_abort_on_error(mocker):
    mock_comm = mocker.patch("mpi4py.MPI.COMM_WORLD")
    mock_comm.size = 2

    sigint_handler = signal.getsignal(signal.SIGINT)

    with pytest.raises(RuntimeError, match="dummy-error"):
        with femto.md.utils.mpi.get_mpi_comm():
            assert femto.md.utils.mpi._INSIDE_MPI_COMM is True
            assert signal.getsignal(signal.SIGINT) != sigint_handler
            raise RuntimeError("dummy-error")

    assert signal.getsignal(signal.SIGINT) == sigint_handler
    assert femto.md.utils.mpi._INSIDE_MPI_COMM is False

    mock_comm.Abort.assert_called_once()


def test_get_mpi_comm_abort_on_signal(mocker):
    mock_comm = mocker.patch("mpi4py.MPI.COMM_WORLD")
    mock_comm.size = 2

    original_sigint_handler = signal.getsignal(signal.SIGINT)

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        with femto.md.utils.mpi.get_mpi_comm():
            assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
            signal.raise_signal(signal.SIGINT)

        mock_comm.Abort.assert_called_once()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


@pytest.mark.parametrize("rank", [None, 0])
def test_reduce_dict(rank):
    value = {"a": 1.0, "b": 2.0, "c": 3.0}

    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        return_value = femto.md.utils.mpi.reduce_dict(value, mpi_comm, rank)

    assert value == return_value


def test_divide_tasks(mocker):
    world_size = 5
    n_total_tasks = 9  # two tasks per worker except one worker with one

    return_values = []

    for rank in range(world_size):
        mock_comm = mocker.MagicMock()
        mock_comm.size = world_size
        mock_comm.rank = rank

        return_values.append(femto.md.utils.mpi.divide_tasks(mock_comm, n_total_tasks))

    # workers should receive two tasks each except the last worker,
    # i.e. (0, 1), (2, 3), (4, 5), (6, 7), (8,)
    assert return_values == [(2, 0), (2, 2), (2, 4), (2, 6), (1, 8)]


@pytest.mark.parametrize(
    "rank, expected_gpu_idx", [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)]
)
def test_divide_gpus(rank, expected_gpu_idx, mocker):
    world_size = 5

    mock_comm_ctx = mocker.MagicMock()
    mock_comm = mock_comm_ctx.__enter__.return_value
    mock_comm.size = world_size
    mock_comm.rank = rank

    mocker.patch(
        "femto.md.utils.mpi.get_mpi_comm",
        autospec=True,
        return_value=mock_comm_ctx,
    )
    mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"})

    femto.md.utils.mpi.divide_gpus()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(expected_gpu_idx)


def test_run_on_rank_zero():
    @femto.md.utils.mpi.run_on_rank_zero
    def dummy_func(arg_a):
        return arg_a * 2

    return_value = dummy_func(2)
    assert return_value == 4


class TestIsInsideMpi:
    def test_not_inside_mpi(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        assert femto.md.utils.mpi.is_inside_mpi() is False

    @pytest.mark.parametrize("env_var", ["PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE"])
    def test_inside_mpi(self, env_var, mocker):
        mocker.patch.dict(os.environ, {env_var: "0"}, clear=True)
        assert femto.md.utils.mpi.is_inside_mpi() is True


class TestIsMpsRunning:
    def test_no_mps_control_binary(self, mocker):
        mocker.patch("shutil.which", return_value=None)
        assert femto.md.utils.mpi.is_mps_running() is False

    def test_mps_running(self, mocker):
        mocker.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control")
        mocker.patch(
            "subprocess.run",
            return_value=mocker.MagicMock(returncode=0),
        )
        assert femto.md.utils.mpi.is_mps_running() is True

    def test_mps_not_running(self, mocker):
        mocker.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control")
        mocker.patch(
            "subprocess.run",
            return_value=mocker.MagicMock(returncode=1),
        )
        assert femto.md.utils.mpi.is_mps_running() is False


class TestStartStopMps:
    def test_start_mps_no_binary(self, mocker):
        mocker.patch("shutil.which", return_value=None)
        with pytest.raises(RuntimeError, match="nvidia-cuda-mps-control not found"):
            femto.md.utils.mpi.start_mps()

    def test_start_mps_already_running(self, mocker):
        mocker.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control")
        mocker.patch("femto.md.utils.mpi.is_mps_running", return_value=True)
        mock_run = mocker.patch("subprocess.run")

        femto.md.utils.mpi.start_mps()

        # Should not have called nvidia-cuda-mps-control -d
        mock_run.assert_not_called()

    def test_start_mps(self, mocker):
        mocker.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control")
        mocker.patch("femto.md.utils.mpi.is_mps_running", return_value=False)
        mock_run = mocker.patch("subprocess.run")

        femto.md.utils.mpi.start_mps()

        mock_run.assert_called_once_with(
            ["nvidia-cuda-mps-control", "-d"], check=True
        )

    def test_stop_mps(self, mocker):
        mock_run = mocker.patch("subprocess.run")

        femto.md.utils.mpi.stop_mps()

        mock_run.assert_called_once_with(
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            text=True,
            check=False,
            timeout=10,
        )


class TestMpsContext:
    def test_starts_and_stops_mps(self, mocker):
        mocker.patch("femto.md.utils.mpi.is_mps_running", return_value=False)
        mock_start = mocker.patch("femto.md.utils.mpi.start_mps")
        mock_stop = mocker.patch("femto.md.utils.mpi.stop_mps")

        with femto.md.utils.mpi.mps_context():
            mock_start.assert_called_once()

        mock_stop.assert_called_once()

    def test_skips_if_already_running(self, mocker):
        mocker.patch("femto.md.utils.mpi.is_mps_running", return_value=True)
        mock_start = mocker.patch("femto.md.utils.mpi.start_mps")
        mock_stop = mocker.patch("femto.md.utils.mpi.stop_mps")

        with femto.md.utils.mpi.mps_context():
            mock_start.assert_not_called()

        mock_stop.assert_not_called()

    def test_stops_on_exception(self, mocker):
        mocker.patch("femto.md.utils.mpi.is_mps_running", return_value=False)
        mocker.patch("femto.md.utils.mpi.start_mps")
        mock_stop = mocker.patch("femto.md.utils.mpi.stop_mps")

        with pytest.raises(RuntimeError):
            with femto.md.utils.mpi.mps_context():
                raise RuntimeError("test")

        mock_stop.assert_called_once()


class TestStripOversubscribeFromArgv:
    def test_strip_separate_args(self):
        argv = ["femto", "septop", "run-complex", "--oversubscribe", "4", "--output-dir", "/tmp"]
        result = femto.md.utils.mpi._strip_oversubscribe_from_argv(argv)
        assert result == ["femto", "septop", "run-complex", "--output-dir", "/tmp"]

    def test_strip_equals_form(self):
        argv = ["femto", "septop", "run-complex", "--oversubscribe=4", "--output-dir", "/tmp"]
        result = femto.md.utils.mpi._strip_oversubscribe_from_argv(argv)
        assert result == ["femto", "septop", "run-complex", "--output-dir", "/tmp"]

    def test_no_oversubscribe(self):
        argv = ["femto", "septop", "run-complex", "--output-dir", "/tmp"]
        result = femto.md.utils.mpi._strip_oversubscribe_from_argv(argv)
        assert result == argv


class TestLaunchWithMps:
    def test_no_gpus(self, mocker):
        mocker.patch("femto.md.utils.mpi._count_cuda_devices", return_value=0)

        with pytest.raises(RuntimeError, match="No CUDA devices found"):
            femto.md.utils.mpi.launch_with_mps(2)

    def test_no_mpirun(self, mocker):
        mocker.patch("femto.md.utils.mpi._count_cuda_devices", return_value=2)
        mocker.patch("shutil.which", return_value=None)
        mocker.patch("sys.argv", ["femto", "septop", "run-complex"])

        with pytest.raises(RuntimeError, match="Cannot find mpirun"):
            femto.md.utils.mpi.launch_with_mps(2)

    def test_launch_auto_detect_mpi(self, mocker):
        mocker.patch("femto.md.utils.mpi._count_cuda_devices", return_value=2)
        mocker.patch(
            "shutil.which",
            side_effect=lambda cmd: "/usr/bin/mpirun" if cmd == "mpirun" else None,
        )
        mocker.patch(
            "sys.argv",
            ["femto", "septop", "run-complex", "--oversubscribe", "4", "--output-dir", "/tmp"],
        )
        mock_mps_ctx = mocker.patch("femto.md.utils.mpi.mps_context")
        mock_mps_ctx.return_value.__enter__ = mocker.MagicMock()
        mock_mps_ctx.return_value.__exit__ = mocker.MagicMock(return_value=False)

        mock_run = mocker.patch(
            "subprocess.run",
            return_value=mocker.MagicMock(returncode=0),
        )

        rc = femto.md.utils.mpi.launch_with_mps(4)

        assert rc == 0

        # Should launch with 2 GPUs * 4 = 8 ranks
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[:4] == ["/usr/bin/mpirun", "-n", "8", "--oversubscribe"]
        # --oversubscribe should be stripped from the child argv
        assert "--oversubscribe" not in cmd[4:]
        assert "4" not in cmd[4:] or cmd[4:].index("4") != cmd[4:].index("--oversubscribe") + 1 if "--oversubscribe" in cmd[4:] else True

        # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE should be 200 // 4 = 50
        env = call_args[1]["env"]
        assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "50"

    def test_launch_custom_mpi_command(self, mocker):
        mocker.patch("femto.md.utils.mpi._count_cuda_devices", return_value=1)
        mocker.patch("sys.argv", ["femto", "septop", "run-complex", "--output-dir", "/tmp"])
        mock_mps_ctx = mocker.patch("femto.md.utils.mpi.mps_context")
        mock_mps_ctx.return_value.__enter__ = mocker.MagicMock()
        mock_mps_ctx.return_value.__exit__ = mocker.MagicMock(return_value=False)

        mock_run = mocker.patch(
            "subprocess.run",
            return_value=mocker.MagicMock(returncode=0),
        )

        rc = femto.md.utils.mpi.launch_with_mps(2, mpi_command="srun --mpi=pmix")

        assert rc == 0

        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["srun", "--mpi=pmix", "-n", "2"]


class TestDivideGpusMps:
    def test_warns_no_mps(self, mocker, caplog):
        """Should warn when multiple ranks share a GPU but MPS is not running."""
        mock_comm_ctx = mocker.MagicMock()
        mock_comm = mock_comm_ctx.__enter__.return_value
        mock_comm.size = 4
        mock_comm.rank = 0

        mocker.patch(
            "femto.md.utils.mpi.get_mpi_comm",
            autospec=True,
            return_value=mock_comm_ctx,
        )
        mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
        mocker.patch("femto.md.utils.mpi.is_mps_running", return_value=False)

        import logging

        with caplog.at_level(logging.WARNING, logger="femto.md.utils.mpi"):
            femto.md.utils.mpi.divide_gpus()

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        assert any("CUDA MPS does not appear to be running" in msg for msg in caplog.messages)

