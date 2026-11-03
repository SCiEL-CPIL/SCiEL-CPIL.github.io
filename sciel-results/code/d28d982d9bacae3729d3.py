import numpy as np
import time

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def construct():
    # Find the largest Sidon set in [0, N-1]
    # A Sidon set has all pairwise sums distinct.
    # We use a greedy approach with backtracking to find a large set.
    
    def find_sidon_set(n, target_size=None):
        if target_size is None:
            # Estimate max size ~ sqrt(n)
            target_size = int(np.sqrt(n)) + 2
            
        # Try to find a set of size target_size, then decrease if fails
        for k in range(target_size, 2, -1):
            # Greedy with backtracking
            indices = []
            used_sums = set()
            
            # Heuristic: Start with small numbers? Or spread out?
            # Standard greedy: pick smallest available that doesn't conflict
            
            def backtrack(start, current_indices, current_sums, k_remaining):
                if len(current_indices) == k:
                    return current_indices[:]
                
                # If we can't fill the remaining slots, prune
                # Max possible additions is (n - start)
                if (n - start) < k_remaining:
                    return None

                for i in range(start, n):
                    # Check if i can be added
                    valid = True
                    for s in current_indices:
                        if (s + i) in current_sums:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    if (i + i) in current_sums:
                        continue
                    
                    # Add i
                    new_sums = set()
                    for s in current_indices:
                        new_sums.add(s + i)
                    new_sums.add(i + i)
                    
                    current_indices.append(i)
                    current_sums.update(new_sums)
                    
                    res = backtrack(i + 1, current_indices, current_sums, k_remaining - 1)
                    if res:
                        return res
                    
                    # Undo
                    current_sums.difference_update(new_sums)
                    current_indices.pop()
                
                return None

            result = backtrack(0, [], set(), k)
            if result:
                return result
        
        return None

    sidon_set = find_sidon_set(N)
    
    if sidon_set is None:
        # Fallback: Uniform
        h = np.ones(N) / N
        return list(h)
    
    # Construct the height vector: Uniform on the Sidon set
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0 / len(sidon_set)
        
    return list(h)
