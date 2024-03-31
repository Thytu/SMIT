import pytest


from src.fn_factory import (
    _access_value,
)


class TestAccesValue:

    test_dict = {
        "foo": 1,
        "bar": ["1"],

        "inner": {
            "foo": "a",
            "bar": 2,
        },
    }

    @pytest.mark.parametrize(
        "key",
        ["foo", "bar"],
    )
    def test_basic_key(self, key: str) -> None:
        assert _access_value(self.test_dict, key) == self.test_dict[key]

    @pytest.mark.parametrize(
        "key",
        ["inner.foo", "inner.bar"],
    )
    def test_inner_key(self, key: str) -> None:
        value = self.test_dict

        for _key in key.split("."):
            value = value[_key]

        assert _access_value(self.test_dict, key) == value
