import time
import random
import sys

def _score_from_bitmask(mask):
    """Count distinct d > 0 such that a, a+d, a+2d all in set represented by bitmask."""
    diffs = 0
    for d in range(1, 257):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            diffs += 1
    return diffs

def _set_to_bitmask(S):
    """Convert set to bitmask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def _bitmask_to_set(mask):
    """Convert bitmask to sorted list of values."""
    result = []
    for i in range(512):
        if mask & (1 << i):
            result.append(i + 1)
    return result

def _score_from_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d in S for some a."""
    mask = _set_to_bitmask(S)
    return _score_from_bitmask(mask)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    N = 32
    MAX_VAL = 512
    
    def random_set():
        return set(random.sample(range(1, MAX_VAL + 1), N))
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Dense sets at various positions
    for offset in [1, 10, 50, 100, 200, 300, 400, 480]:
        S = set(range(offset, offset + N))
        if max(S) <= MAX_VAL:
            initializations.append(S)
    
    # Arithmetic progressions
    for step in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        if 1 + (N - 1) * step <= MAX_VAL:
            initializations.append(set(1 + i * step for i in range(N)))
            if (MAX_VAL - N * step) > 0:
                initializations.append(set(MAX_VAL - (N - 1 - i) * step for i in range(N)))
    
    # Union of two APs with different steps
    for d1 in [2, 3, 4, 5, 6, 8]:
        for d2 in [3, 4, 5, 6, 8, 10, 12]:
            if d1 == d2:
                continue
            for n1 in range(10, N - 9):
                n2 = N - n1
                # Try several offset combinations
                for a in [1, 5, 10, 20, 50]:
                    for b in [1, 5, 10, 20, 50, 100, 200]:
                        A = set(a + i * d1 for i in range(n1))
                        B = set(b + j * d2 for j in range(n2))
                        if not (A & B) and max(A) <= MAX_VAL and max(B) <= MAX_VAL:
                            if min(A) >= 1 and min(B) >= 1:
                                S = A | B
                                if len(S) == N:
                                    initializations.append(S)
    
    # Random sets
    for _ in range(20):
        initializations.append(random_set())
    
    # Evaluate all initializations
    best_set = None
    best_score = -1
    for S in initializations:
        if len(S) != N:
            continue
        score = _score_from_set(S)
        if score > best_score:
            best_score = score
            best_set = set(S)
    
    current_set = best_set
    current_score = best_score
    
    # Phase 2: Simulated Annealing with bitmask representation
    def simulated_annealing(S, time_budget=30.0):
        S = set(S)
        mask = _set_to_bitmask(S)
        score = _score_from_bitmask(mask)
        best_S = set(S)
        best_score = score
        
        T = 2.0  # Initial temperature
        T_min = 0.01
        cooling = 0.995
        sa_start = time.time()
        
        while T > T_min and (time.time() - sa_start) < time_budget:
            if time.time() > deadline - 5.0:
                break
            
            # Choose a random element to remove
            x = random.choice(list(S))
            # Choose a random value to add
            y = random.randint(1, MAX_VAL)
            
            if y == x or y in S:
                T *= cooling
                continue
            
            # Compute new score using bitmask
            new_mask = mask & ~(1 << (x - 1)) | (1 << (y - 1))
            new_score = _score_from_bitmask(new_mask)
            
            delta = new_score - score
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta > 0:
                S = S - {x} | {y}
                mask = new_mask
                score = new_score
                if score > best_score:
                    best_score = score
                    best_S = set(S)
            else:
                # Compute acceptance probability
                # delta is negative or zero
                try:
                    prob = min(1.0, 2.718281828 ** (delta / T))
                except:
                    prob = 0.0
                
                if random.random() < prob:
                    S = S - {x} | {y}
                    mask = new_mask
                    score = new_score
            
            T *= cooling
        
        return best_S, best_score
    
    # Phase 3: Targeted local search using difference coverage
    def targeted_local_search(S, time_budget=20.0):
        S = set(S)
        score = _score_from_set(S)
        best_S = set(S)
        best_score = score
        ts_start = time.time()
        
        # Build difference coverage map: for each d, which elements are involved in 3-APs with difference d
        def get_diff_coverage(S):
            mask = _set_to_bitmask(S)
            coverage = {}
            for d in range(1, 257):
                if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                    # Find which elements participate
                    participants = set()
                    for i in range(512):
                        if mask & (1 << i):
                            a = i + 1
                            if mask & (1 << (i + d)) and mask & (1 << (i + 2 * d)):
                                participants.add(a)
                                participants.add(a + d)
                                participants.add(a + 2 * d)
                    coverage[d] = participants
            return coverage
        
        improved = True
        while improved and (time.time() - ts_start) < time_budget:
            improved = False
            if time.time() > deadline - 5.0:
                break
            
            coverage = get_diff_coverage(S)
            used_diffs = set(coverage.keys())
            
            # Find elements that are critical for many differences
            element_criticality = {}
            for x in S:
                count = 0
                for d in used_diffs:
                    if x in coverage[d]:
                        count += 1
                element_criticality[x] = count
            
            # Try to improve by replacing low-criticality elements
            for x in sorted(S, key=lambda v: element_criticality.get(v, 0)):
                if time.time() > deadline - 5.0:
                    break
                
                # Find candidate replacements that add new differences
                best_y = None
                best_new_score = score
                
                # Try values that would create new 3-APs
                for y in range(1, MAX_VAL + 1):
                    if y in S or y == x:
                        continue
                    
                    new_S = S - {x} | {y}
                    new_score = _score_from_set(new_S)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_y = y
                
                if best_y is not None and best_new_score > score:
                    S = S - {x} | {best_y}
                    score = best_new_score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_S = set(S)
                    break
        
        return best_S, best_score
    
    # Run optimization phases
    # First, try simulated annealing on the best initial solution
    if time.time() < deadline - 60.0:
        sa_set, sa_score = simulated_annealing(current_set, time_budget=25.0)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
            current_set = sa_set
    
    # Then, try targeted local search
    if time.time() < deadline - 40.0:
        ts_set, ts_score = targeted_local_search(current_set, time_budget=20.0)
        if ts_score > best_score:
            best_score = ts_score
            best_set = ts_set
            current_set = ts_set
    
    # Restart with different initializations and re-optimize
    restart_count = 0
    while time.time() < deadline - 20.0 and restart_count < 5:
        restart_count += 1
        
        # Pick a random or structured initialization
        if random.random() < 0.5:
            current_set = random_set()
        else:
            # Structured: mix of dense and sparse
            if random.random() < 0.3:
                offset = random.randint(1, MAX_VAL - N)
                current_set = set(range(offset, offset + N))
            else:
                step = random.choice([2, 3, 4, 5, 6, 8, 10])
                if 1 + (N - 1) * step <= MAX_VAL:
                    start = random.randint(1, MAX_VAL - (N - 1) * step)
                    current_set = set(start + i * step for i in range(N))
                else:
                    current_set = random_set()
        
        # Optimize
        if time.time() < deadline - 15.0:
            sa_set, sa_score = simulated_annealing(current_set, time_budget=15.0)
            if sa_score > best_score:
                best_score = sa_score
                best_set = sa_set
        
        if time.time() < deadline - 10.0:
            ts_set, ts_score = targeted_local_search(best_set, time_budget=8.0)
            if ts_score > best_score:
                best_score = ts_score
                best_set = ts_set
    
    # Final polish
    if time.time() < deadline - 5.0:
        final_set, final_score = targeted_local_search(best_set, time_budget=5.0)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    return sorted(best_set)
