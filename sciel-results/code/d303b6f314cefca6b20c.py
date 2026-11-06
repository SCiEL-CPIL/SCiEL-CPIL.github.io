def _repair_add(base_mask, v, rng):
    """
    Try to add v to base_mask.
    If it fails, try to remove minimal elements to make it feasible.
    This is a heuristic: try removing elements that are part of the conflict.
    """
    # First check if direct add works
    cand = _add_value(base_mask, v)
    if _is_200_free(cand):
        return cand
    
    # If not, we need to remove some elements.
    # We don't easily know which ones without solving subset sum.
    # Heuristic: try removing each element u in base_mask one by one and see if base_mask \ {u} U {v} is feasible.
    # This is O(N * Cost_Check).
    # Since N=100, this is 100 checks.
    # If any single removal works, do it.
    present = _set_bits(base_mask)
    rng.shuffle(present)
    for u in present:
        cand_swap = _add_value(_remove_value(base_mask, u), v)
        if _is_200_free(cand_swap):
            return cand_swap
            
    # If no single removal works, try removing pairs?
    # This is O(N^2). 10000 checks. Might be too slow if done often.
    # But we only do this in initialization or specific repair steps.
    # Let's stick to single removal for now in the local search.
    return base_mask
