import time
import math
import numpy as np

def _score_set_fast(U):
    """Fast score computation using numpy for larger sets."""
    if not U or 0 not in U:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    max_u = U_arr[-1]
    
    # Compute sumset U+U
    sums = U_arr[:, None] + U_arr[None, :]
    p = len(np.unique(sums))
    
    # Compute difference set U-U
    diffs = U_arr[:, None] - U_arr[None, :]
    d = len(np.unique(diffs))
    
    if p == 0 or max_u < 0:
        return 0.0
    
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    
    ratio = d / p
    if ratio <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / math.log(denom)
    return score

def _score_set(U):
    """Compute the score for a finite set U of nonnegative integers containing 0."""
    if len(U) > 30:
        return _score_set_fast(U)
    if not U or 0 not in U:
        return 0.0
    U_sorted = sorted(set(U))
    n = len(U_sorted)
    max_u = U_sorted[-1]
    
    # Compute U+U (sumset)
    sum_set = set()
    for a in U_sorted:
        for b in U_sorted:
            sum_set.add(a + b)
    p = len(sum_set)
    
    # Compute U-U (difference set)
    diff_set = set()
    for a in U_sorted:
        for b in U_sorted:
            diff_set.add(a - b)
    d = len(diff_set)
    
    if p == 0 or max_u < 0:
        return 0.0
    
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    
    ratio = d / p
    if ratio <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / math.log(denom)
    return score

def _try_add_element(U, candidate):
    """Try adding a candidate element to U and return the new score."""
    new_U = list(U) + [candidate]
    return _score_set(new_U)

def _try_remove_element(U, idx):
    """Try removing element at index idx from U and return the new score."""
    new_U = [U[i] for i in range(len(U)) if i != idx]
    if not new_U or 0 not in new_U:
        return 0.0
    return _score_set(new_U)

def _gap_based_score(U, candidate):
    """
    Compute a 'frequency gap' score for adding candidate to U.
    Measures how many new differences are created vs new sums.
    Higher is better: we want many new differences, few new sums.
    """
    if candidate in U:
        return -1.0
    new_U = sorted(U + [candidate])
    
    # Count new differences
    new_diffs = 0
    for a in new_U:
        for b in new_U:
            if a != candidate and b != candidate:
                continue
            diff = a - b
            # Check if this diff is new
            is_new = True
            for x in U:
                for y in U:
                    if x - y == diff:
                        is_new = False
                        break
                if not is_new:
                    break
            if is_new:
                new_diffs += 1
    
    # Count new sums
    new_sums = 0
    for a in new_U:
        for b in new_U:
            if a != candidate and b != candidate:
                continue
            s = a + b
            is_new = True
            for x in U:
                for y in U:
                    if x + y == s:
                        is_new = False
                        break
                if not is_new:
                    break
            if is_new:
                new_sums += 1
    
    # The gap score: maximize new_diffs / (new_sums + 1)
    # This encourages elements that create many new differences but few new sums
    if new_sums == 0:
        return new_diffs * 10.0
    return new_diffs / (new_sums + 1.0)

def _local_search(U, deadline, max_iter=500):
    """Perform local search by adding/removing elements with gap-based candidate selection."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    
    current_score = _score_set(U)
    max_u = U[-1] if U else 0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_move = None
        best_new_U = None
        
        # Generate candidates using gap-based scoring
        candidates = set()
        
        # Near existing elements
        for u in U:
            for delta in range(-5, 6):
                c = u + delta
                if 0 <= c <= max_u + 20 and c not in U:
                    candidates.add(c)
        
        # Strategic positions: multiples, gaps
        for u in U:
            for mult in [2, 3, 4, 5]:
                c = u * mult
                if 0 <= c <= max_u * 3 + 10 and c not in U:
                    candidates.add(c)
        
        # Large jumps to expand range
        for offset in [5, 10, 15, 20, 30, 50]:
            c = max_u + offset
            if c not in U:
                candidates.add(c)
        
        # Sort candidates by gap-based score (descending)
        scored_candidates = []
        for cand in candidates:
            gap_score = _gap_based_score(U, cand)
            scored_candidates.append((gap_score, cand))
        
        scored_candidates.sort(key=lambda x: -x[0])
        
        # Evaluate top candidates by actual score
        for gap_score, cand in scored_candidates[:30]:  # Limit to top 30
            if time.time() > deadline:
                break
            new_score = _score_set(U + [cand])
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_move = ('add', cand)
                best_new_U = U + [cand]
        
        # Try removing elements (except 0)
        for i in range(len(U)):
            if i == 0 and U[0] == 0:
                continue  # Don't remove 0
            if time.time() > deadline:
                break
            new_score = _score_set([U[j] for j in range(len(U)) if j != i])
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_move = ('remove', i)
                best_new_U = [U[j] for j in range(len(U)) if j != i]
        
        if best_move is not None and best_gain > 1e-9:
            U = sorted(best_new_U)
            current_score = _score_set(U)
            max_u = U[-1] if U else 0
        else:
            break
    
    return U, current_score

def _construct_greedy(deadline):
    """Build a set greedily using gap-based candidate selection."""
    U = [0]
    current_score = _score_set(U)
    
    max_u = 0
    for step in range(200):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_cand = None
        
        # Generate candidates and score by gap
        candidates = set()
        
        # Near existing elements
        for u in U:
            for delta in range(-3, 4):
                c = u + delta
                if 0 <= c <= max_u + 30 and c not in U:
                    candidates.add(c)
        
        # Strategic positions
        for u in U:
            for mult in [2, 3, 4]:
                c = u * mult
                if 0 <= c <= max_u * 3 + 5 and c not in U:
                    candidates.add(c)
        
        # Range expansion
        for offset in [3, 5, 8, 12, 18, 25, 35, 50]:
            c = max_u + offset
            if c not in U:
                candidates.add(c)
        
        # Score candidates by gap
        scored = []
        for cand in candidates:
            gs = _gap_based_score(U, cand)
            scored.append((gs, cand))
        scored.sort(key=lambda x: -x[0])
        
        # Evaluate top 20 by actual score
        for gs, cand in scored[:20]:
            if time.time() > deadline:
                break
            new_score = _score_set(U + [cand])
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        
        if best_cand is not None and best_gain > 1e-9:
            U.append(best_cand)
            U = sorted(U)
            current_score = _score_set(U)
            max_u = U[-1]
        else:
            break
    
    return U, current_score

def _random_restart(deadline, base_set, seed=42):
    """Do a random restart with perturbation."""
    import random
    rng = random.Random(seed)
    
    U = list(base_set)
    current_score = _score_set(U)
    
    # Perturb: add a few random elements
    max_u = max(U) if U else 0
    for _ in range(3):
        if time.time() > deadline:
            break
        cand = rng.randint(1, max_u + 30)
        if cand not in U:
            U.append(cand)
            U = sorted(U)
    
    U, current_score = _local_search(U, deadline, max_iter=100)
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Start with a reasonable initial set
    initial_sets = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 7],
    ]
    
    best_U = None
    best_score = 0.0
    
    # Evaluate initial sets
    for init in initial_sets:
        if time.time() > deadline:
            break
        s = _score_set(init)
        if s > best_score:
            best_score = s
            best_U = list(init)
    
    if best_U is None:
        return [0, 1, 3]
    
    # Phase 1: Greedy construction with gap-based selection
    greedy_deadline = deadline - 40.0
    greedy_U, greedy_score = _construct_greedy(greedy_deadline)
    if greedy_score > best_score:
        best_score = greedy_score
        best_U = greedy_U
    
    # Phase 2: Local search from best
    if time.time() < deadline - 30.0:
        ls_deadline = deadline - 30.0
        ls_U, ls_score = _local_search(best_U, ls_deadline, max_iter=300)
        if ls_score > best_score:
            best_score = ls_score
            best_U = ls_U
    
    # Phase 3: Random restarts with different seeds
    restart_deadline = deadline - 15.0
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > restart_deadline:
            break
        rr_deadline = min(restart_deadline, time.time() + 15.0)
        rr_U, rr_score = _random_restart(rr_deadline, best_U, seed=seed)
        if rr_score > best_score:
            best_score = rr_score
            best_U = rr_U
    
    # Final polish: one more local search with more iterations
    if time.time() < deadline - 5.0:
        final_deadline = deadline - 3.0
        final_U, final_score = _local_search(best_U, final_deadline, max_iter=200)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
    
    # Ensure 0 is in the set and return sorted
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.append(0)
        best_U = sorted(best_U)
    
    return best_U
