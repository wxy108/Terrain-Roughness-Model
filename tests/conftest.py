from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="Run complete real-terrain scans.")
    parser.addoption("--run-compiled", action="store_true", help="Run configured CUDA compiler check.")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="Enable --run-slow for full real-terrain scans."))
        if "compiled" in item.keywords and not config.getoption("--run-compiled"):
            item.add_marker(pytest.mark.skip(reason="Enable --run-compiled with a working CUDA compiler."))


@pytest.fixture(scope="session", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def root():
    return ROOT
