def is_violation_with_substitution(cid, e_idx, new_color):
    cycle = cycles[cid]
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    # Substitute e_idx
    if cycle[0] == e_idx: c0 = new_color
    elif cycle[1] == e_idx: c1 = new_color
    elif cycle[2] == e_idx: c2 = new_color
    elif cycle[3] == e_idx: c3 = new_color
    
    if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
        return True
    return False
