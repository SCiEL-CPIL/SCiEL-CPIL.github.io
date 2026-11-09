import numpy as np
import time
import itertools
import sys

def generate_candidates(R2, d=11):
    """
    Generates all integer vectors in Z^d with squared norm <= R2.
    """
    candidates = []
    # Possible squared values for a single coordinate
    # x^2 <= R2 => |x| <= sqrt(R2)
    max_coord = int(np.sqrt(R2))
    # We can iterate over all combinations of squared values?
    # Or just iterate over all possible vectors?
    # For R2=16, max_coord=4.
    # 9^11 is too big.
    
    # Use the partition method
    # Find all multisets of squares {s1, ..., sd} such that sum(si) <= R2
    # si must be a perfect square: 0, 1, 4, 9, 16...
    
    possible_squares = [i*i for i in range(max_coord + 1) if i*i <= R2]
    
    # Generate all combinations of length d with replacement from possible_squares
    # such that sum <= R2
    # This is still large if d=11.
    # Number of partitions of 16 into 11 parts is small.
    
    # Better: iterate over all possible "patterns" of non-zero coordinates.
    # Actually, just iterating over the product of possible coordinate values is 9^11.
    # We can prune.
    
    # Alternative:
    # Generate all valid vectors by recursion.
    
    def gen(idx, remaining, current):
        if idx == d:
            if remaining >= 0:
                candidates.append(current[:])
            return
        
        # Determine max value for this coordinate
        # x^2 <= remaining
        max_x = int(np.sqrt(remaining))
        for x in range(-max_x, max_x + 1):
            current[idx] = x
            gen(idx + 1, remaining - x*x, current)

    # This recursion is 9^11 in worst case, but pruned heavily.
    # For R2=16, depth 11.
    # Level 0: x in -4..4 (9)
    # Level 1: x in -4..4 (9) ...
    # It might still be slow in Python.
    
    # Let's try a different generation method:
    # Iterate over all possible squared norms n from 0 to R2.
    # For each n, find all vectors with norm squared n.
    # This is the same problem.
    
    # Let's use the partition of n into d squares.
    
    all_vectors = []
    for n in range(R2 + 1):
        # Find all compositions of n into d squares
        # This is equivalent to finding all x in Z^d with ||x||^2 = n
        
        # We can use a helper to generate these
        def gen_exact(idx, remaining, current):
            if idx == d:
                if remaining == 0:
                    all_vectors.append(current[:])
                return
            
            max_x = int(np.sqrt(remaining))
            for x in range(-max_x, max_x + 1):
                current[idx] = x
                gen_exact(idx + 1, remaining - x*x, current)

        gen_exact(0, n, [0]*d)
        
    return np.array(all_vectors, dtype=np.int64)

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    best_set = None
    best_score = 0
    
    # Try different R2 values
    # Start with high potential R2, but ensure we have a fallback
    R2_values = [16, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    for R2 in R2_values:
        if time.time() > deadline - 60: # Leave 60s for final check/return
            break
            
        try:
            points = generate_candidates(R2)
            N = len(points)
            if N < 2:
                continue
                
            # If N is small, use explicit graph MIS
            # If N is large, use vectorized greedy
            
            # Estimate N
            if N < 50000:
                # Build graph
                # dist_sq[i,j] = ||points[i] - points[j]||^2
                # We need dist_sq < R2 to be an edge (conflict)
                
                # Compute distances
                # To save memory, we can do it in blocks or just use the vectorized greedy
                # Vectorized greedy is O(K*N) vector ops.
                # For N=50000, K~25000, ops ~ 1.25 * 10^9.
                # This might be slow in Python loop.
                
                # Let's stick to vectorized greedy for all, but optimize it.
                
                # Shuffle points
                # We need to run multiple restarts.
                
                best_local_set = None
                best_local_size = 0
                
                num_restarts = 50
                for r in range(num_restarts):
                    if time.time() > deadline - 30:
                        break
                        
                    order = np.random.permutation(N)
                    available = np.ones(N, dtype=bool)
                    selected = []
                    
                    for idx in order:
                        if not available[idx]:
                            continue
                            
                        selected.append(idx)
                        
                        # Compute distances from points[idx] to all points
                        diff = points - points[idx]
                        dist_sq = np.sum(diff**2, axis=1)
                        
                        # Mark conflicts
                        # dist_sq < R2 means conflict
                        # Note: dist_sq[idx] is 0, so it will be marked
                        available[dist_sq < R2] = False
                        
                    if len(selected) > best_local_size:
                        best_local_size = len(selected)
                        best_local_set = [points[i] for i in selected]
                        
                if best_local_set and best_local_size > best_score:
                    best_score = best_local_size
                    best_set = best_local_set
                    
        except Exception as e:
            print(f"Error for R2={R2}: {e}", file=sys.stderr)
            continue
            
    if best_set is None:
        # Fallback
        return np.array([[1]*11, [-1]*11], dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(res.shape)
