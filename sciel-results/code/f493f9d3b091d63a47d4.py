import random
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def find_max_sidon_set(n):
        """
        Find a large Sidon set in {0, ..., n-1}.
        A Sidon set has distinct pairwise sums.
        """
        # Greedy approach with backtracking or randomized restarts
        best_set = [0]
        best_size = 1
        
        # Try multiple random starts
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            S = []
            used_sums = set()
            # Randomize the order of addition to find different Sidon sets
            candidates = list(range(n))
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if adding x creates any sum collision
                new_sums = [x + s for s in S]
                new_sums.append(2 * x)
                
                # Check for collisions with existing sums or within new_sums
                # Note: 2*x is a sum x+x.
                # We need all sums s_i + s_j (i<=j) to be distinct.
                
                conflict = False
                for s in new_sums:
                    if s in used_sums:
                        conflict = True
                        break
                    if new_sums.count(s) > 1: # This check is inefficient, better to use a set
                        # Actually, if x is in S, x+x is a new sum.
                        # If x not in S, x+s for s in S are distinct from each other?
                        # x+s1 = x+s2 => s1=s2. Yes.
                        # So we just need to check against used_sums.
                        pass
                
                if not conflict:
                    S.append(x)
                    for s in new_sums:
                        used_sums.add(s)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
                        
        return best_set, best_size

    # Find the best Sidon set
    S, size = find_max_sidon_set(N)
    
    # Create the height vector
    h = np.zeros(N)
    if size > 0:
        # Uniform weights on the Sidon set
        # Normalize to L1 = 1
        val = 1.0 / size
        for idx in S:
            h[idx] = val
            
    # Fallback if something goes wrong
    if np.sum(h) == 0:
        h[0] = 1.0
        
    return list(h)
# EVOLVE-BLOCK-END
