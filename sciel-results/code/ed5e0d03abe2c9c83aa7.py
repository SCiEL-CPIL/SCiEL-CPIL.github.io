"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys
import numpy as np

def _triple_sums_involving_x(x, S):
    """Compute all triple sums involving x and elements of S (including x itself)."""
    sums = set()
    sums.add(3 * x)
    for y in S:
        sums.add(2 * x + y)
        sums.add(x + 2 * y)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            sums.add(x + S[i] + S[j])
    return sums

def _check_candidate_fast(x, S, existing_sums):
    """Fast check if x can be added to S without triple-sum collisions."""
    # Quick rejection: check if 3x, 2x+y, x+2y collide with existing
    if 3 * x in existing_sums:
        return False, None
    
    new_sums = {3 * x}
    for y in S:
        s1 = 2 * x + y
        s2 = x + 2 * y
        if s1 in existing_sums or s2 in existing_sums:
            return False, None
        if s1 == s2:  # This means 2x+y == x+2y => x==y, but x not in S so ok
            pass
        new_sums.add(s1)
        new_sums.add(s2)
    
    # Check internal collisions among new sums
    if len(new_sums) < 1 + 2 * len(S):
        return False, None
    
    # Check x+y+z for y<=z in S
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            s = x + S[i] + S[j]
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
    
    return True, new_sums

def _greedy_increasing(deadline):
    """Simple greedy in increasing order."""
    S = []
    existing_sums = set()
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        valid, new_sums = _check_candidate_fast(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    return S

def _greedy_vectorized(deadline, seed=0):
    """
    Greedy using vectorized NumPy operations for collision checking.
    Maintains a boolean array of forbidden triple sums.
    """
    rng = random.Random(seed)
    S = []
    
    # Max possible triple sum is 3*4096 = 12288
    max_sum = 3 * 4096
    forbidden = np.zeros(max_sum + 1, dtype=bool)
    
    used = [False] * 4097
    
    candidates = list(range(1, 4097))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if used[x]:
            continue
        
        # Vectorized check for 3x
        if forbidden[3 * x]:
            continue
        
        # Vectorized check for 2x+y and x+2y for all y in S
        if S:
            S_arr = np.array(S)
            sums_2x_y = 2 * x + S_arr
            sums_x_2y = x + 2 * S_arr
            
            # Check if any of these are forbidden
            if np.any(forbidden[sums_2x_y]) or np.any(forbidden[sums_x_2y]):
                continue
            
            # Check internal collisions: 2x+y1 == 2x+y2 implies y1==y2 (ok)
            # 2x+y1 == x+2y2 implies x+y1 == 2y2
            # x+2y1 == x+2y2 implies y1==y2 (ok)
            # Need to check if 2x+y_i collides with x+2y_j for any i,j
            # 2x + y_i == x + 2y_j => x + y_i == 2y_j
            # This is hard to vectorize simply, so do a quick check
            combined = np.concatenate([sums_2x_y, sums_x_2y])
            if len(set(combined.tolist())) < len(combined):
                continue
        
        # Check x+y+z for y<=z in S
        if len(S) >= 2:
            S_arr = np.array(S)
            n = len(S_arr)
            # Generate all pairs y<=z
            i_idx, j_idx = np.triu_indices(n, k=0)
            pair_sums = S_arr[i_idx] + S_arr[j_idx]
            triple_sums = x + pair_sums
            
            if np.any(forbidden[triple_sums]):
                continue
            
            # Check internal collisions
            if len(set(triple_sums.tolist())) < len(triple_sums):
                continue
            
            # Also check collision with 2x+y and x+2y
            if S:
                S_arr2 = np.array(S)
                sums_2x_y = 2 * x + S_arr2
                sums_x_2y = x + 2 * S_arr2
                all_new = np.concatenate([np.array([3*x]), sums_2x_y, sums_x_2y, triple_sums])
                if len(set(all_new.tolist())) < len(all_new):
                    continue
        
        # Valid! Add x
        S.append(x)
        used[x] = True
        
        # Update forbidden array
        forbidden[3 * x] = True
        for y in S[:-1]:  # All previous elements
            forbidden[2 * x + y] = True
            forbidden[x + 2 * y] = True
        for i in range(len(S) - 1):
            for j in range(i + 1, len(S) - 1):
                forbidden[x + S[i] + S[j]] = True
    
    return S

def _local_search(S, deadline, seed=0):
    """Local search: try to swap elements to improve the set."""
    rng = random.Random(seed)
    S = sorted(S)
    best = S[:]
    best_size = len(S)
    
    # Precompute existing sums
    def compute_sums(arr):
        sums = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                for k in range(j, len(arr)):
                    sums.add(arr[i] + arr[j] + arr[k])
        return sums
    
    existing_sums = compute_sums(S)
    
    while time.time() < deadline:
        improved = False
        
        # Try removing one element and adding a smaller one
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            
            removed = S[idx]
            S_new = S[:idx] + S[idx+1:]
            sums_without = compute_sums(S_new)
            
            # Try to add a smaller element
            for x in range(1, removed):
                if x in S_new:
                    continue
                
                valid, new_sums = _check_candidate_fast(x, S_new, sums_without)
                if valid:
                    S_final = sorted(S_new + [x])
                    if len(S_final) > best_size:
                        best = S_final
                        best_size = len(S_final)
                        S = S_final
                        existing_sums = compute_sums(S)
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Strategy 1: Simple greedy in increasing order
    S1 = _greedy_increasing(deadline)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Vectorized greedy
    if time.time() < deadline:
        S2 = _greedy_vectorized(deadline, seed=42)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Local search from best
    if time.time() < deadline and best:
        S3 = _local_search(best, deadline, seed=123)
        if len(S3) > len(best):
            best = S3[:]
    
    return best
