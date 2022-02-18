import pytest
from pytest_cases import parametrize_with_cases

from mibi_bin_tools import panel_utils


class TestPanels:

    def case_single_row_named(self):
        return 98, 'Mo98', 0.5, 0.5

    def case_multirow_named(self):
        return [92, 98], ['Mo92', 'Mo98'], 0.5, 0.5

    def case_single_row_unnamed(self):
        return 98, None, 0.5, 0.5

    def case_multirow_unnamed(self):
        return [92, 98], None, 0.5, 0.5

    def case_multirange_named(self):
        return [92, 98], ['Mo92', 'Mo98'], [0.3, 0.5], [0.3, 0.5]

    def case_multirange_unnamed(self):
        return [92, 98], None, [0.3, 0.5], [0.3, 0.5]

    def case_splitrange(self):
        return [92, 98], None, [0.3, 0.5], 0.3

    @pytest.mark.xfail(raises=ValueError)
    def case_single_row_multiname(self):
        return 98, ['Mo92', 'Mo98'], 0.5, 0.5

    @pytest.mark.xfail(raises=ValueError)
    def case_multirow_singlename(self):
        return [92, 98], 'Mo92', 0.5, 0.5

    @pytest.mark.xfail(raises=ValueError)
    def case_single_row_multirange(self):
        return 98, None, [0.3, 0.5], [0.3, 0.5]


@parametrize_with_cases('mass, target_name, low_range, high_range', cases=TestPanels)
def test_make_panel(mass, target_name, low_range, high_range):
    _ = panel_utils.make_panel(mass,  target_name, low_range, high_range)
