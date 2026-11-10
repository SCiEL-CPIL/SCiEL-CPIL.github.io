import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: ensure we return well before 120s/150s limit
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Parameters for the specific task
    # q=3, n=7, radius=1
    # Total space size: 3^7 = 2187
    # We represent words as integers 0..2186
    # A codeword covers itself and all words at Hamming distance 1.
    # Number of words covered by one codeword: 1 + n*(q-1) = 1 + 7*2 = 15.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if needed, though task is fixed
        # For now, just return a simple greedy for small cases or seed logic
        # Since the task is fixed to q=3, n=7, radius=1, we focus there.
        # But to be safe:
        if q == 3 and n >= 4:
            import itertools
            tetracode = [[a, b, (a + b) % 3, (2 * a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Actually, the seed used a specific tetracode. Let's stick to a simple fallback.
            # The primary path below handles the specific case.
            pass
        
    # Precompute the coverage for every possible word in the space
    # Space size: 3^7 = 2187
    # For each word w, compute the set of words it covers.
    # To speed up, we can precompute the neighbors.
    
    # Convert int to base-3 tuple
    def int_to_tuple(x, n):
        return [x // (3**i) % 3 for i in range(n)]
    
    def tuple_to_int(t):
        return sum(t[i] * (3**i) for i in range(n))
    
    # Precompute coverage sets as bitsets or lists?
    # With 2187 words, a boolean array of size 2187 is efficient.
    # We can represent the "uncovered" state as a boolean array.
    # However, checking if a codeword covers any new words requires iterating its neighbors.
    
    # Let's precompute the list of neighbors for each word.
    # neighbors[i] = list of indices covered by word i (including i)
    neighbors = []
    for i in range(3**n):
        t = int_to_tuple(i, n)
        covered = [i]
        for pos in range(n):
            orig_val = t[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t[pos] = new_val
                covered.append(tuple_to_int(t))
        neighbors.append(covered)
        
    # Greedy Set Cover
    # State: covered_mask (boolean array or set of covered indices)
    # Using a set of covered indices is slower than a boolean array + count, 
    # but checking "is any neighbor uncovered" is fast with boolean array.
    
    # We want to pick the codeword that maximizes the number of NEWLY covered words.
    # This is a standard greedy heuristic.
    
    covered = np.zeros(3**n, dtype=bool)
    codewords = []
    
    # Optimization: 
    # Instead of scanning all 2187 words every time to find the best one,
    # we can just scan all. 2187 * 186 iterations * 15 neighbors check ~ 6 million ops.
    # This is very fast in Python if optimized, or NumPy.
    # Let's use a simple loop first, it should be under 1 second.
    
    # To speed up the "count new covers" step:
    # For a candidate codeword c, new_covers = count(not covered[neighbor] for neighbor in neighbors[c])
    
    # We can maintain a list of "active" candidates? 
    # Actually, just scanning all 2187 is fine.
    
    while time.time() < deadline:
        best_idx = -1
        best_new_count = 0
        
        # Check all possible codewords
        # We can skip codewords that are already in the set? 
        # No, we don't need to add a codeword if it covers 0 new words.
        # But we must ensure we don't pick the same codeword twice.
        # Since we only add if best_new_count > 0, and once added, its neighbors are covered,
        # it will yield 0 new covers next time. So no duplicates.
        
        for i in range(3**n):
            # If time is running out, break
            if time.time() > deadline - 5.0:
                break
                
            # Count how many neighbors are NOT yet covered
            # Using NumPy for this specific check might be faster than Python loop
            # covered is a numpy array. neighbors[i] is a list of ints.
            # new_covered = np.sum(~covered[neighbors[i]])
            
            # Let's try a Python loop first, it might be fast enough for 2187*186
            # 2187 candidates * 186 steps * 15 checks = ~6M ops. Python does ~10M-50M simple ops/sec.
            # This should take < 1 second.
            
            count = 0
            for nb in neighbors[i]:
                if not covered[nb]:
                    count += 1
                    if count > best_new_count:
                        # Early exit if we already beat the current best
                        # But we need the exact count to compare? 
                        # If count > best_new_count, we can break early IF we only care about the max.
                        # Yes, if count > best_new_count, this candidate is better.
                        # But we need to know if it's the BEST. 
                        # We can break early if count > best_new_count, but we still need to finish counting 
                        # to know if it's even better than another candidate? 
                        # No, we just need the max. 
                        # However, breaking early prevents us from knowing the exact count, 
                        # but we don't need the exact count, we just need to know it's better than current best.
                        # Wait, if we break, we don't know the final count. 
                        # But if count > best_new_count, we update best_new_count to count (which is an underestimate? No, it's the current partial count).
                        # Actually, if we break, the true count is >= current count.
                        # So if current count > best_new_count, the true count is definitely > best_new_count.
                        # So we can update best_idx = i, best_new_count = count (underestimate is fine for comparison? 
                        # No, if we set best_new_count to an underestimate, a later candidate might have a higher partial count but lower true count?
                        # No, if candidate A has partial count 10 (true 15) and candidate B has partial count 12 (true 12), 
                        # we pick B. But A was better. 
                        # So we CANNOT break early if we want the exact max.
                        # We must count all neighbors to get the exact new coverage count.
                        pass
            
            # Let's just count all. It's small.
            count = 0
            for nb in neighbors[i]:
                if not covered[nb]:
                    count += 1
            
            if count > best_new_count:
                best_new_count = count
                best_idx = i
        
        if best_idx == -1:
            break # All covered
        
        # Add the best codeword
        codewords.append(best_idx)
        for nb in neighbors[best_idx]:
            covered[nb] = True
        
        # Check if all covered
        if np.all(covered):
            break

    # Convert indices back to lists of integers
    result = [int_to_tuple(i, n) for i in codewords]
    
    # Final validity check (optional, but good for debugging)
    # The greedy algorithm guarantees coverage by construction (it stops when all covered).
    
    return result
