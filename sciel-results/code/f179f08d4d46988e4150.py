import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    N = 1 << n
    deadline = time.time() + 50.0  # Safety margin
    
    # 1. Precompute Comparability Matrix
    # M[i, j] = 1 if i and j are comparable (i subset j or j subset i) and i != j
    all_masks = np.arange(N, dtype=np.int32)
    
    # We can compute this in chunks or just loop 4096 times.
    # 4096 * 4096 operations is ~16M, which is fast in numpy.
    M = np.zeros((N, N), dtype=np.uint8)
    
    for i in range(N):
        # i is subset of j: (i & j) == i
        # j is subset of i: (j & i) == j
        # Since i != j, these are disjoint conditions for strict subset.
        # If i == j, both are true, but we want 0.
        
        # Vectorized calculation for row i
        # sub1: i is subset of j
        sub1 = (i & all_masks) == i
        # sub2: j is subset of i
        sub2 = (all_masks & i) == all_masks
        
        # Combine
        row = sub1 | sub2
        
        # Set diagonal to 0
        row[i] = 0
        
        M[i, :] = row.astype(np.uint8)
        
    # Ensure symmetry (should be, but just in case)
    # M = (M + M.T) // 2 ? No, if M[i,j]=1, M[j,i]=1.
    # The calculation is symmetric.
    
    best_score = -1
    best_family = None
    
    # 2. Multiple Restarts
    num_restarts = 50  # Adjust based on time
    
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize family
        # Random initialization
        family = random.sample(range(N), m)
        family = [int(x) for x in family]
        
        in_family = np.zeros(N, dtype=bool)
        for x in family:
            in_family[x] = True
            
        count_vec = np.zeros(N, dtype=np.int32)
        score = 0
        
        # Calculate initial score and count_vec
        for i in range(m):
            A = family[i]
            # Add contributions from A to others
            # We need count_vec[j] = number of sets in family comparable with j
            # So for each A in family, add M[A, :] to count_vec
            count_vec += M[A, :]
            
        # Score is sum of M[A, B] for all pairs.
        # Sum of count_vec is 2 * score (since each pair is counted twice, once for each node)
        # Wait, M[A, A] = 0.
        # Sum_{A in F} count_vec[A] = Sum_{A in F} Sum_{B in F} M[A, B] = 2 * Score
        score = int(np.sum(count_vec) // 2)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
            
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                    
                A = family[i]
                old_loss = count_vec[A]
                
                # Remove A
                count_vec -= M[A, :]
                in_family[A] = False
                
                # Find best B
                masked = count_vec.copy()
                masked[in_family] = -1
                B = int(np.argmax(masked))
                new_gain = count_vec[B]
                
                if new_gain > old_loss:
                    # Accept swap
                    family[i] = B
                    in_family[B] = True
                    count_vec += M[B, :]
                    score += (new_gain - old_loss)
                    improved = True
                    
                    if score > best_score:
                        best_score = score
                        best_family = family[:]
                else:
                    # Revert
                    count_vec += M[A, :]
                    in_family[A] = True
                    
        # Check if we can improve with a different init strategy?
        # The local search is quite strong.
        
    # Convert best_family to lists
    if best_family is None:
        # Fallback
        best_family = list(range(m))
        
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
