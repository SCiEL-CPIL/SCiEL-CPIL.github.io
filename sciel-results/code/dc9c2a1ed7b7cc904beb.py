    # Modify edge_list management for O(1) removal
    # We need to find the index of (u, v) in edge_list.
    # This is still O(N) unless we maintain a dict mapping edge -> index.
    # But indices change as we swap.
    # Maintaining a dict edge -> index is complex with swaps.
    # Given N=401, O(N) removal is likely fast enough in C-implemented list.remove?
    # list.remove is C code. 401 comparisons is negligible.
