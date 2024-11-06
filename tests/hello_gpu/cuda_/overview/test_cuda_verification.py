import pytest
from hamcrest import assert_that, not_, raises

from src.hello_gpu.cuda_.overview import verification


@pytest.mark.gpu
def test_verify_does_NOT_raises_Exception() -> None:
    assert_that(verification.verify(), not_(raises(Exception)))
