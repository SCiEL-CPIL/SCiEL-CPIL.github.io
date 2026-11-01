import time
import numpy as np
import itertools
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode (Radius 2 for n>4, but valid format)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (invalid for score but valid format)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    # balls[w] is a list of words in the ball of radius 1 around w
    # Shape: (total, 1 + 2*n) = (6561, 17)
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Strategy:
    # 1. Solve LP relaxation of Set Cover
    # 2. Weighted Greedy Rounding based on LP values
    # 3. Prune
    # 4. Local Search (if time permits)

    # 1. LP Relaxation
    # Minimize sum(x_i)
    # Subject to: sum_{j in ball(i)} x_j >= 1 for all i
    # 0 <= x_j <= 1
    
    # A_cover[i, j] = 1 if j in ball(i)
    # Constraint: A_cover @ x >= 1
    # For linprog: -A_cover @ x <= -1
    
    rows = np.repeat(np.arange(total), 1 + 2*n)
    cols = balls.flatten()
    data = np.ones(total * (1 + 2*n), dtype=np.float64)
    
    A_cover_sparse = csr_matrix((data, (rows, cols)), shape=(total, total))
    
    c = np.ones(total, dtype=np.float64)
    A_ub = -A_cover_sparse
    b_ub = -np.ones(total, dtype=np.float64)
    bounds = [(0, 1) for _ in range(total)]
    
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={"presolve": True})
        if res.success:
            x_opt = res.x
        else:
            print("LP failed, falling back to greedy")
    except Exception as e:
        print(f"LP exception: {e}")

    if x_opt is None:
        # Greedy Construction Fallback
        selected = _greedy_construct(balls, deadline)
    else:
        # 2. Weighted Greedy Rounding
        # Sort all indices by x_opt descending
        order = np.argsort(x_opt)[::-1]
        selected_indices = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        for j in order:
            if not np.any(uncovered_mask):
                break
            # Check if j covers any uncovered point
            if np.any(uncovered_mask[balls[j]]):
                selected_indices.append(j)
                cover_count[balls[j]] += 1
                uncovered_mask = (cover_count == 0)
                
        # If holes remain, add arbitrary codewords to cover them
        if np.any(uncovered_mask):
            for j in range(total):
                if np.any(uncovered_mask[balls[j]]):
                    # Check if already added
                    # Since we iterated all j in order, if x_opt[j] was low, we might have skipped it.
                    # We need to add it now.
                    # We can't easily check "in selected_indices" for large lists efficiently without a set.
                    # But N=6561 is small.
                    if j not in selected_indices:
                        selected_indices.append(j)
                        cover_count[balls[j]] += 1
                        uncovered_mask = (cover_count == 0)
                        if not np.any(uncovered_mask):
                            break

        selected = selected_indices

    # 3. Pruning
    # Recalculate cover counts for the selected set
    cover_count = np.zeros(total, dtype=np.int32)
    for j in selected:
        cover_count[balls[j]] += 1
        
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        new_selected = []
        for c in selected:
            # Check if removing c leaves all its neighbors covered by others
            # i.e., cover_count[balls[c]] > 1 for all w in balls[c]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    # 4. Local Search (Swap)
    # Try to swap a codeword for another that allows removal of others?
    # Given time, we skip complex swaps and rely on the LP start + pruning.
    
    # Convert selected to list of digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

def _greedy_construct(balls, deadline):
    total = balls.shape[0]
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    while np.any(current_uncovered) and time.time() < deadline - 5.0:
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        # Deterministic pick: first one
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    return selected
