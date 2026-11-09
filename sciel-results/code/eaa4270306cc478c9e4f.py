import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    N = 1 << n
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    # 1. Precompute Comparability Matrix
    # M[i, j] = 1 if i and j are comparable (i subset j or j subset i) and i != j
    all_masks = np.arange(N, dtype=np.int32)
    
    M = np.zeros((N, N), dtype=np.int32)
    
    # Vectorized construction of M
    for i in range(N):
        # i subset j: (i & j) == i
        sub1 = (i & all_masks) == i
        # j subset i: (j & i) == j
        sub2 = (all_masks & i) == all_masks
        row = sub1 | sub2
        row[i] = 0 # Strict subset, so diagonal is 0
        M[i, :] = row.astype(np.int32)
        
    best_score = -1
    best_family = None
    
    def _greedy_init():
        """Build a family greedily by adding the set with highest marginal gain."""
        fam = []
        in_fam = np.zeros(N, dtype=bool)
        count_vec = np.zeros(N, dtype=np.int32)
        
        # Start with a random set to break symmetry
        first_set = random.randint(0, N - 1)
        fam.append(first_set)
        in_fam[first_set] = True
        count_vec += M[first_set, :]
        
        for _ in range(m - 1):
            # Gain of adding x is count_vec[x]
            gains = count_vec.copy()
            gains[in_fam] = -1
            best_x = int(np.argmax(gains))
            
            fam.append(best_x)
            in_fam[best_x] = True
            count_vec += M[best_x, :]
            
        return fam, count_vec, in_fam

    def _random_init():
        fam = random.sample(range(N), m)
        fam = [int(x) for x in fam]
        in_fam = np.zeros(N, dtype=bool)
        for x in fam:
            in_fam[x] = True
        count_vec = np.zeros(N, dtype=np.int32)
        for x in fam:
            count_vec += M[x, :]
        return fam, count_vec, in_fam

    def _calc_score(fam, count_vec):
        # Score is sum of M[i,j] for i<j in fam
        # Sum of count_vec[i] for i in fam is 2 * Score
        return int(np.sum(count_vec[np.array(fam)]) // 2)

    def _local_search(fam, count_vec, in_fam, score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                A = fam[i]
                old_score_contribution = count_vec[A]
                
                # Calculate potential gain for swapping A with any B
                # Delta(B) = (count_vec[B] - M[A, B]) - count_vec[A]
                # We want to maximize Delta(B) for B not in fam
                
                gains = count_vec - M[A, :] - old_score_contribution
                
                # Mask out sets already in family
                gains[in_fam] = -1000000 # Large negative
                
                best_B = int(np.argmax(gains))
                best_gain = gains[best_B]
                
                if best_gain > 0:
                    # Perform swap
                    # Remove A
                    in_fam[A] = False
                    count_vec -= M[A, :]
                    
                    # Add B
                    in_fam[best_B] = True
                    count_vec += M[best_B, :]
                    
                    fam[i] = best_B
                    score += best_gain
                    improved = True
                    
                    if score > best_score:
                        best_score = score
                        best_family = fam[:]
        
        return fam, count_vec, in_fam, score

    # Main Loop
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        # Alternate between random and greedy init
        if r % 2 == 0:
            fam, count_vec, in_fam = _greedy_init()
        else:
            fam, count_vec, in_fam = _random_init()
            
        score = _calc_score(fam, count_vec)
        
        if score > best_score:
            best_score = score
            best_family = fam[:]
            
        # Run local search
        fam, count_vec, in_fam, score = _local_search(fam, count_vec, in_fam, score)
        
        if score > best_score:
            best_score = score
            best_family = fam[:]
            
    # Final Polish: Run local search on the best family found so far
    if best_family is not None:
        # Reconstruct state for best family
        in_fam = np.zeros(N, dtype=bool)
        for x in best_family:
            in_fam[x] = True
        count_vec = np.zeros(N, dtype=np.int32)
        for x in best_family:
            count_vec += M[x, :]
        
        best_score = _calc_score(best_family, count_vec)
        
        # Run local search again, potentially for longer
        fam, count_vec, in_fam, score = _local_search(best_family, count_vec, in_fam, best_score)
        
        if score > best_score:
            best_score = score
            best_family = fam[:]

    # Convert best_family to lists
    if best_family is None:
        best_family = list(range(m))
        
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
