import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_aps():
    """
    Precompute all 3-term arithmetic progressions in 1..512.
    Returns a list of lists, where ap_by_elem[x] contains tuples (d, other1, other2)
    for all APs involving x.
    """
    ap_by_elem = [[] for _ in range(513)]
    for d in range(1, 256):
        max_a = 512 - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            ap_by_elem[a].append((d, b, c))
            ap_by_elem[b].append((d, a, c))
            ap_by_elem[c].append((d, a, b))
    return ap_by_elem

AP_BY_ELEM = _precompute_aps()

def _init_state(S):
    """
    Initialize state. Returns (S_set, mask, d_count, score).
    mask is a numpy array for the SA incremental updates.
    """
    S_set = set(S)
    mask = np.zeros(513, dtype=np.uint8)
    for x in S:
        mask[x] = 1
    d_count = np.zeros(256, dtype=np.int32)
    for d in range(1, 256):
        end = 513 - 2 * d
        if end <= 0:
            break
        d_count[d] = int(np.sum(mask[0:end] & mask[d:d + end] & mask[2 * d:2 * d + end]))
    score = int(np.sum(d_count > 0))
    return S_set, mask, d_count, score

def _apply_add(S_set, mask, d_count, score, x):
    for d, e1, e2 in AP_BY_ELEM[x]:
        if mask[e1] and mask[e2]:
            if d_count[d] == 0:
                score += 1
            d_count[d] += 1
    mask[x] = 1
    S_set.add(x)
    return score

def _apply_remove(S_set, mask, d_count, score, x):
    for d, e1, e2 in AP_BY_ELEM[x]:
        if mask[e1] and mask[e2]:
            d_count[d] -= 1
            if d_count[d] == 0:
                score -= 1
    mask[x] = 0
    S_set.discard(x)
    return score

def _simulated_annealing(S_init, time_budget, rng):
    S = list(S_init)
    S_set, mask, d_count, current_score = _init_state(S)
    best_S = list(S)
    best_score = current_score
    start_time = time.time()
    T = 5.0
    cooling = 0.9995
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.005:
            break
        i = rng.randint(0, 31)
        old_elem = S[i]
        new_elem = rng.randint(1, 512)
        if new_elem == old_elem or new_elem in S_set:
            continue
        _apply_remove(S_set, mask, d_count, current_score, old_elem)
        new_score = _apply_add(S_set, mask, d_count, current_score, new_elem)
        delta = new_score - current_score
        if delta > 0 or (delta == 0 and rng.random() < 0.5) or rng.random() < np.exp(delta / T):
            current_score = new_score
            S[i] = new_elem
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            _apply_remove(S_set, mask, d_count, current_score, new_elem)
            _apply_add(S_set, mask, d_count, current_score, old_elem)
    return best_S, best_score

def _score_mask_bitset(mask_int):
    """
    Score a set represented as a Python integer bitset.
    Bit i-1 corresponds to integer i.
    """
    count = 0
    # d can be at most 255 (1, 1+d, 1+2d <= 512 => 2d <= 511 => d <= 255)
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are in S
        # This is equivalent to checking if S & (S >> d) & (S >> 2*d) is non-zero
        if (mask_int & (mask_int >> d) & (mask_int >> (2 * d))) != 0:
            count += 1
    return count

def _fast_greedy(S_init, time_limit=10.0):
    """
    Fast greedy local search using Python integer bitsets for scoring.
    This is significantly faster than the NumPy-based incremental updates for 
    full re-evaluation, allowing many more passes.
    """
    S = list(S_init)
    s_set = set(S)
    
    # Convert to bitset
    mask_int = 0
    for x in S:
        mask_int |= (1 << (x - 1))
    
    current_score = _score_mask_bitset(mask_int)
    start_time = time.time()
    
    # Repeat passes until no improvement or time limit
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            
            # Remove old_val from mask and set
            mask_without = mask_int & ~(1 << (old_val - 1))
            # We don't need to remove from s_set yet, just check membership carefully
            
            best_cand = None
            best_new_score = current_score
            
            # Try all candidates
            for cand in range(1, 513):
                if cand == old_val:
                    continue
                if cand in s_set:
                    continue
                
                new_mask = mask_without | (1 << (cand - 1))
                new_score = _score_mask_bitset(new_mask)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_cand = cand
            
            if best_cand is not None:
                # Apply the improvement
                s_set.remove(old_val)
                s_set.add(best_cand)
                S[i] = best_cand
                mask_int = mask_without | (1 << (best_cand - 1))
                current_score = best_new_score
                improved = True
        
        if not improved:
            break
            
    return S, current_score

def _structured_init(rng):
    r = rng.random()
    if r < 0.3:
        start = rng.randint(1, 480)
        S = sorted(set(range(start, start + 32)))
        while len(S) < 32:
            S.append(rng.randint(1, 512))
        return sorted(set(S))[:32]
    elif r < 0.6:
        step = 512 // 32
        S = [rng.randint(1, step * (i + 1)) for i in range(32)]
        S = sorted(set(S))
        while len(S) < 32:
            S.append(rng.randint(1, 512))
        return sorted(set(S))[:32]
    else:
        return sorted(rng.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim for 145s to be safe.
    deadline = start_time + 145.0
    rng = random.Random(123)
    
    # Initialize with a simple set
    best_S = list(range(1, 33))
    best_score = _init_state(best_S)[3]
    
    # Phase 1: Simulated Annealing with multiple restarts
    # We allocate about 60% of the time to SA to explore different basins.
    sa_deadline = start_time + 80.0
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > sa_deadline:
            break
            
        if restart == 0:
            S_init = list(range(1, 33))
        elif restart == 1:
            S_init = _structured_init(rng)
        elif restart == 2:
            # Spread out
            S_init = sorted(set([max(1, min(512, int(512 * (i + 1) / 33))) for i in range(32)]))
            while len(S_init) < 32:
                S_init.append(rng.randint(1, 512))
            S_init = sorted(set(S_init))[:32]
        else:
            S_init = sorted(rng.sample(range(1, 513), 32))
            
        # Calculate time for this restart
        remaining_sa = sa_deadline - time.time()
        if remaining_sa < 5.0:
            break
            
        # Allocate time per restart, ensuring we don't overshoot
        time_per_restart = min(7.0, remaining_sa - 2.0)
        if time_per_restart < 2.0:
            break
            
        S, score = _simulated_annealing(S_init, time_per_restart, rng)
        
        if score > best_score:
            best_score = score
            best_S = list(S)
            
    # Phase 2: Fast Greedy Polish
    # Use the remaining time for the fast bitset-based greedy search.
    # This is much more efficient at finding local optima.
    if time.time() < deadline - 5.0:
        greedy_time = deadline - 5.0 - time.time()
        # Run fast greedy on the best solution found so far
        best_S, best_score = _fast_greedy(best_S, greedy_time)
        
    return sorted(best_S)

# EVOLVE-BLOCK-END
