<<<<<<< SEARCH
def _greedy_build(order, max_elem=MAX_ELEM):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= max_elem:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected
=======
def _can_add(x, selected):
    """
    Check if x can be added to selected without violating constraints.
    """
    m = len(selected)
    
    # 1. Check: does x divide sum of any 3 from selected?
    if m >= 3:
        for i in range(m):
            a = selected[i]
            for j in range(i + 1, m):
                b = selected[j]
                for k in range(j + 1, m):
                    c = selected[k]
                    if (a + b + c) % x == 0:
                        return False
    
    # 2. Check: for each a in selected, does a divide sum of any 3 from (selected U {x}) \ {a}?
    for i, a in enumerate(selected):
        for j in range(m):
            if j == i:
                continue
            b = selected[j]
            for k in range(j + 1, m):
                if k == i:
                    continue
                c = selected[k]
                if (x + b + c) % a == 0:
                    return False
                    
    return True

def _greedy_build(order, max_elem=MAX_ELEM):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= max_elem:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected
>>>>>>> REPLACE

