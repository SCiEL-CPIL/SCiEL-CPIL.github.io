"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # First try the original greedy to get a baseline
    baseline = _greedy_construct(n_rows, k)
    if baseline is not None:
        scope = max(max(r) for r in baseline)
        best_rows = baseline
        best_scope = scope
    
    # Try multiple random restarts with different row orderings
    rng = random.Random(42)
    max_restarts = 50000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        result = _greedy_construct_with_order(n_rows, k, order)
        if result is not None:
            scope = max(max(r) for r in result)
            if scope < best_scope:
                best_scope = scope
                best_rows = result
                # If we hit the counting bound, we can't do better
                if best_scope <= 105:
                    break
    
    if best_rows is None:
        # Fallback: return a valid (though possibly suboptimal) result
        return _greedy_construct(n_rows, k)
    return best_rows


def _greedy_construct(n_rows, k):
    """Original greedy: construct rows in order 0,1,2,..."""
    order = list(range(n_rows))
    return _greedy_construct_with_order(n_rows, k, order)


def _greedy_construct_with_order(n_rows, k, order):
    """Greedy construction with specified row order."""
    used = set()
    rows = [None] * n_rows
    
    for row_idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows[row_idx] = row
    
    return rows


# EVOLVE-BLOCK-END
