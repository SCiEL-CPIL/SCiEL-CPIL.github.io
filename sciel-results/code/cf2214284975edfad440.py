"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
DEADLINE = 85.0  # 5s safety margin before 90s kill

def count_3aps(S):
    """Count distinct 3-term APs (positive diff, counted once) in binary array S (1-indexed internally)."""
    # S is 0-indexed, representing elements 1..N
    count = 0
    # Precompute all possible 3-term APs
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            if S[a] and S[a + d] and S[a + 2 * d]:
                count += 1
    return count

def count_3aps_fast(S):
    """Faster count using numpy for all 3-term APs."""
    # Vectorized: for each diff d, count positions a where S[a] & S[a+d] & S[a+2d]
    total = 0
    for d in range(1, N // 2 + 1):
        if N - 2 * d <= 0:
            break
        seg1 = S[:N - 2 * d]
        seg2 = S[d:N - d]
        seg3 = S[2 * d:]
        total += np.sum(seg1 & seg2 & seg3)
    return int(total)

def has_4ap(S):
    """Check if S contains any nonconstant 4-term AP."""
    for d in range(1, N // 3 + 1):
        if N - 3 * d <= 0:
            break
        seg1 = S[:N - 3 * d]
        seg2 = S[d:N - 2 * d]
        seg3 = S[2 * d:N - d]
        seg4 = S[3 * d:]
        if np.any(seg1 & seg2 & seg3 & seg4):
            return True
    return False

def greedy_construct():
    """Greedy: add elements one by one, preferring those that maximize 3-APs while avoiding 4-APs."""
    S = np.zeros(N, dtype=np.int8)
    # Try adding elements in order of "value" — start with a random-ish order
    order = list(range(N))
    random.shuffle(order)
    
    for idx in order:
        if time.time() > DEADLINE:
            break
        S[idx] = 1
        if has_4ap(S):
            S[idx] = 0
    return S

def local_search(S, max_time):
    """Local search: try swaps to improve 3-AP count while maintaining no-4-AP constraint."""
    best_S = S.copy()
    best_score = count_3aps_fast(best_S)
    start = time.time()
    
    # Precompute all 4-APs for quick checking
    # For efficiency, we'll check 4-AP constraint incrementally
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        current_score = best_score
        
        # Try each element for removal
        for i in np.where(best_S == 1)[0]:
            if time.time() > DEADLINE:
                break
            
            S_temp = best_S.copy()
            S_temp[i] = 0
            base_score = count_3aps_fast(S_temp)
            
            # Try adding each absent element
            for j in np.where(S_temp == 0)[0]:
                if time.time() > DEADLINE:
                    break
                
                S_temp[j] = 1
                if has_4ap(S_temp):
                    S_temp[j] = 0
                    continue
                
                new_score = count_3aps_fast(S_temp)
                if new_score > current_score:
                    best_S = S_temp.copy()
                    best_score = new_score
                    current_score = new_score
                    improved = True
                else:
                    S_temp[j] = 0
    
    return best_S, best_score

def multi_start_search(total_time):
    """Multi-start local search with different initial seeds."""
    start_time = time.time()
    best_S = None
    best_score = -1
    
    # Start with greedy construction
    S0 = greedy_construct()
    S0, score0 = local_search(S0, total_time * 0.3)
    best_S = S0
    best_score = score0
    
    elapsed = time.time() - start_time
    if elapsed >= total_time * 0.7:
        return best_S, best_score
    
    # Additional random starts
    rng = random.Random(42)
    num_starts = 5
    remaining = total_time * 0.7 - elapsed
    per_start = remaining / num_starts
    
    for k in range(num_starts):
        if time.time() > DEADLINE:
            break
        
        # Random initial set with bias toward sparse
        S_rand = np.zeros(N, dtype=np.int8)
        # Start with ~30% density
        candidates = list(range(N))
        rng.shuffle(candidates)
        n_initial = int(N * 0.3)
        
        for idx in candidates[:n_initial]:
            if time.time() > DEADLINE:
                break
            S_rand[idx] = 1
            if has_4ap(S_rand):
                S_rand[idx] = 0
        
        S_rand, score_rand = local_search(S_rand, per_start)
        if score_rand > best_score:
            best_S = S_rand
            best_score = score_rand
    
    return best_S, best_score

def construct():
    start_time = time.time()
    
    # Use multi-start local search
    S_best, score_best = multi_start_search(DEADLINE - start_time)
    
    # Final check: ensure no 4-AP
    if has_4ap(S_best):
        # Fallback: greedy construction
        S_best = greedy_construct()
        if has_4ap(S_best):
            # Remove elements causing 4-APs
            for d in range(1, N // 3 + 1):
                for a in range(0, N - 3 * d):
                    if all(S_best[a + k * d] for k in range(4)):
                        # Remove the last element of this AP
                        S_best[a + 3 * d] = 0
    
    # Ensure nonempty
    if not np.any(S_best):
        S_best = np.zeros(N, dtype=np.int8)
        S_best[0] = 1  # Element 1
    
    # Convert to 1-indexed list
    result = sorted(np.where(S_best == 1)[0].tolist())
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
