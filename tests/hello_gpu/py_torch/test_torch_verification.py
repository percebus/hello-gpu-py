import pytest
from hamcrest import assert_that, not_, raises

from src.hello_gpu.py_torch import verification


@pytest.mark.gpu
def test__verify_all__does_NOT_raises_Exception() -> None:
    assert_that(verification.verify_all(), not_(raises(Exception)))
