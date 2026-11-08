"""
Strategy: Constraint satisfaction search over binary vectors of length L using
backtracking with incremental bitmask maintenance and strong pruning.

The key insight is that for small sets (n <= 6), we can model the problem as
choosing which elements from {0, 1, ..., L-1} to include. We use bitmasks to
incrementally maintain |U+U| and |U-U|, and prune branches where even the
maximum possible future ratio cannot beat the best known score.

We combine this with:
1. Structured initial candidates (geometric, arithmetic, known good sets)
2. A fast numpy-based scoring for candidate evaluation
3. Local search polish (add/remove/swap) to refine the best found set
4. Deadline guard to stay within 150s wall-clock
"""
import time
import math
import numpy as np

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2*max(U) + 1)
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    sum_set = set()
    diff_set = set()
    
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def _score_numpy(U):
    """Fast score using numpy for larger sets."""
    if not U or len(U) < 2:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 0.0
    
    sums = U_arr[:, None] + U_arr[None, :]
    p = len(np.unique(sums))
    
    diffs = U_arr[:, None] - U_arr[None, :]
    d = len(np.unique(diffs))
    
    if p == 0:
        return 0.0
    
    ratio = d / p
    if ratio <= 0:
        return 0.0
    
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    
    return 1.0 + math.log(ratio) / math.log(denom)

def _backtrack_search(L, target_n, best_score_holder, best_U_holder, deadline):
    """
    Backtracking search over subsets of {0, 1, ..., L-1} of size target_n
    that must include 0. Uses incremental bitmask for sum/diff sets.
    Prunes branches where upper bound on score cannot exceed best.
    
    best_score_holder: list [score] to allow mutation
    best_U_holder: list [U] to allow mutation
    """
    if time.time() > deadline:
        return
    
    # We'll search for sets of exactly target_n elements including 0.
    # So we need to choose (target_n - 1) elements from {1, ..., L-1}.
    
    # Upper bound on |U+U| for a set with max element M: at most 2*M+1 distinct sums in [0, 2M]
    # Upper bound on |U-U| for a set with max element M: at most 2*M+1 distinct diffs in [-M, M]
    # So ratio <= 1 always? No, |U-U| can be up to 2M+1 and |U+U| can be as low as...
    # Actually for a set U, |U+U| >= 2|U|-1 and |U-U| <= 2|U|-1... no that's not right.
    # |U-U| is at most 2*max(U)+1 and at least 2*|U|-1 (for distinct elements).
    # |U+U| is at most 2*max(U)+1 and at least 2*|U|-1.
    
    # The ratio |U-U|/|U+U| can be > 1 when differences are more spread out than sums.
    # This happens when the set has "gaps" that create many distinct differences.
    
    # For pruning: given current partial set S with max element m, and we need to add (target_n - len(S)) more elements
    # all <= L-1. The final max will be at most L-1.
    # Current |S-S| and |S+S| give a lower bound on final sizes.
    # The ratio can only increase if we add elements that create more differences than sums.
    # Hard to bound tightly, so we use a looser bound:
    # Final |U-U| <= 2*(L-1)+1 = 2L-1
    # Final |U+U| >= 2*target_n - 1 (minimum for distinct nonneg ints)
    # So max possible ratio <= (2L-1)/(2*target_n-1)
    # Score upper bound = 1 + log((2L-1)/(2*target_n-1)) / log(2*(L-1)+1)
    
    max_ratio_ub = (2 * L - 1) / (2 * target_n - 1)
    if max_ratio_ub <= 1:
        return
    score_ub = 1.0 + math.log(max_ratio_ub) / math.log(2 * (L - 1) + 1)
    if score_ub <= best_score_holder[0]:
        return
    
    # Recursive backtracking
    def backtrack(start, chosen, current_sum_mask, current_diff_mask, sum_count, diff_count):
        """
        start: next index to consider in {1, ..., L-1}
        chosen: list of chosen elements (excluding 0)
        current_sum_mask: bitmask of sums (bits 0..2*(L-1))
        current_diff_mask: bitmask of diffs (bits 0..2*(L-1)-1, offset by L-1)
        sum_count: number of bits set in current_sum_mask
        diff_count: number of bits set in current_diff_mask
        """
        if time.time() > deadline:
            return
        
        needed = target_n - 1 - len(chosen)
        
        if needed == 0:
            # Complete set found
            U = [0] + sorted(chosen)
            maxU = U[-1]
            ratio = diff_count / sum_count
            if ratio > 0:
                s = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
                if s > best_score_holder[0]:
                    best_score_holder[0] = s
                    best_U_holder[0] = U[:]
            return
        
        # Pruning: not enough elements left
        if L - 1 - start + 1 < needed:
            return
        
        # Upper bound on achievable score from this branch
        # Even if we add elements to maximize ratio, what's the best we can do?
        # Current ratio: diff_count / sum_count
        # Adding elements can increase both. The max final ratio is bounded by (2L-1)/(2*target_n-1)
        # But we can also bound based on current state:
        # If current ratio is already low and we can't increase it much, prune.
        # Loose bound: final diff_count <= 2L-1, final sum_count >= current sum_count
        # So max ratio from here <= (2L-1) / sum_count
        # Score ub = 1 + log((2L-1)/sum_count) / log(2*(L-1)+1)
        if sum_count > 0:
            ratio_ub = (2 * L - 1) / sum_count
            if ratio_ub > 0:
                s_ub = 1.0 + math.log(ratio_ub) / math.log(2 * (L - 1) + 1)
                if s_ub <= best_score_holder[0]:
                    return
        
        # Try skipping 'start'
        if start <= L - 1:
            backtrack(start + 1, chosen, current_sum_mask, current_diff_mask, sum_count, diff_count)
        
        # Try including 'start'
        if start <= L - 1:
            x = start
            new_sum_mask = current_sum_mask
            new_diff_mask = current_diff_mask
            new_sum_count = sum_count
            new_diff_count = diff_count
            
            # Add x to the set (0 is already included)
            # New sums: 0+x=x, x+0=x, x+x=2x, and for each existing element y in chosen: x+y
            # Since 0 is in the set, sums include x (from 0+x) and 2x (from x+x)
            # For each y in chosen: sum = x+y
            # Diffs: x-0=x, 0-x=-x, x-x=0, and for each y: x-y, y-x
            
            # Check if x is new in sum mask
            if not (current_sum_mask & (1 << x)):
                new_sum_mask |= (1 << x)
                new_sum_count += 1
            if not (current_sum_mask & (1 << (2 * x))):
                new_sum_mask |= (1 << (2 * x))
                new_sum_count += 1
            
            # Diffs: x (offset by L-1), -x (offset by L-1 => L-1-x), 0 (offset by L-1)
            # Diff mask index: diff + (L-1)
            diff_x_idx = x + (L - 1)
            diff_negx_idx = -x + (L - 1)
            diff_0_idx = 0 + (L - 1)
            
            if not (current_diff_mask & (1 << diff_x_idx)):
                new_diff_mask |= (1 << diff_x_idx)
                new_diff_count += 1
            if not (current_diff_mask & (1 << diff_negx_idx)):
                new_diff_mask |= (1 << diff_negx_idx)
                new_diff_count += 1
            if not (current_diff_mask & (1 << diff_0_idx)):
                new_diff_mask |= (1 << diff_0_idx)
                new_diff_count += 1
            
            # For each existing chosen element y:
            for y in chosen:
                s_val = x + y
                if not (current_sum_mask & (1 << s_val)):
                    new_sum_mask |= (1 << s_val)
                    new_sum_count += 1
                
                d_val = x - y
                d_idx = d_val + (L - 1)
                if not (current_diff_mask & (1 << d_idx)):
                    new_diff_mask |= (1 << d_idx)
                    new_diff_count += 1
                
                d_val2 = y - x
                d_idx2 = d_val2 + (L - 1)
                if not (current_diff_mask & (1 << d_idx2)):
                    new_diff_mask |= (1 << d_idx2)
                    new_diff_count += 1
            
            new_chosen = chosen + [x]
            backtrack(start + 1, new_chosen, new_sum_mask, new_diff_mask, new_sum_count, new_diff_count)
    
    # Initialize: set contains {0}
    # Sum mask: 0+0=0
    # Diff mask: 0-0=0
    initial_sum_mask = 1 << 0  # sum 0
    initial_diff_mask = 1 << (L - 1)  # diff 0 at index L-1
    
    backtrack(1, [], initial_sum_mask, initial_diff_mask, 1, 1)

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s kill
    
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline
    
    # Phase 1: Structured candidates
    # Geometric sequences
    for base in [2, 3, 4, 5, 6]:
        for n in range(3, 8):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            try_set(U)
    
    # {0, 1, a, a*b} type
    for a in range(2, 30):
        if time.time() > deadline:
            break
        for b in range(2, 20):
            if time.time() > deadline:
                break
            c = a * b
            if c > 500:
                break
            try_set([0, 1, a, c])
    
    # {0, 1, a, b, c} geometric-like
    for a in range(2, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            if time.time() > deadline:
                break
            for c in range(b + 1, 80):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Exhaustive 4-element sets {0, a, b, c} with 0 < a < b < c <= 60
    for a in range(1, 25):
        if time.time() > deadline:
            break
        for b in range(a + 1, 40):
            if time.time() > deadline:
                break
            for c in range(b + 1, 61):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 35
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 15):
            if time.time() > deadline:
                break
            for c in range(b + 1, 25):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 36):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Phase 2: Backtracking search with bitmask for small L
    # Search for sets of size 4, 5, 6 within {0, ..., L-1}
    for n_target in [4, 5, 6]:
        for L in range(n_target + 1, 25):
            if time.time() > deadline:
                break
            best_score_holder = [best_score]
            best_U_holder = [best_U[:]]
            _backtrack_search(L, n_target, best_score_holder, best_U_holder, deadline)
            if best_score_holder[0] > best_score:
                best_score = best_score_holder[0]
                best_U = best_U_holder[0][:]
    
    # Phase 3: Local search polish
    current_U = list(best_U)
    current_score = best_score
    
    while time.time() < deadline - 5:
        improved = False
        current_max = max(current_U) if current_U else 0
        
        # Try adding an element
        for x in range(1, current_max + 15):
            if x not in current_U:
                new_U = sorted(list(set(current_U + [x])))
                s = compute_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
        
        # Try removing an element
        if not improved:
            for x in current_U:
                if x == 0:
                    continue
                new_U = [y for y in current_U if y != x]
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
        
        # Try swapping
        if not improved:
            for i, x in enumerate(current_U):
                if x == 0:
                    continue
                for y in range(1, current_max + 10):
                    if y == x or y in current_U:
                        continue
                    new_U = list(current_U)
                    new_U[i] = y
                    new_U = sorted(list(set(new_U)))
                    if len(new_U) != len(current_U):
                        continue
                    s = compute_score(new_U)
                    if s > current_score:
                        current_U = new_U
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    if current_score > best_score:
        best_U = current_U
        best_score = current_score
    
    return best_U
