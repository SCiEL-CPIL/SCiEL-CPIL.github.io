import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        # Use the seed logic for general fallback if possible
        # The seed logic works for q=3, n>=4.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial valid code: all words
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as bitmasks
    # sphere[i] is an integer with bits set for all words within distance 1 of all_words[i]
    spheres = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        spheres[i] = mask
        
    full_mask = (1 << N) - 1
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_code = None
    best_size = N # worst case
    
    # Use vectorized NumPy operations for speed
    # spheres_arr is shape (N, N) boolean matrix where spheres_arr[i, j] is True if j is in sphere of i
    spheres_arr = np.zeros((N, N), dtype=bool)
    for i in range(N):
        s = spheres[i]
        # Efficiently set bits in numpy array from integer mask
        # Since N=729, we can do this by iterating set bits or using unpackbits if we pad to power of 2,
        # but simple loop is fine for setup.
        # Actually, let's just build it directly to avoid conversion overhead in setup if possible, 
        # but we already have integer masks. 
        # Let's convert integer masks to numpy boolean rows.
        # A faster way: create a lookup or use bit operations.
        # For N=729, converting 729 integers of 729 bits is fast enough.
        # We can use np.unpackbits but it requires padding to multiple of 8.
        # Let's just do a simple loop for the setup phase, it's one-time cost.
        # Or better, we can compute the numpy matrix directly during precomputation.
        pass

    # Let's re-do the precomputation to directly populate numpy array for speed
    spheres_arr = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        spheres_arr[i, i] = True
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                spheres_arr[i, j] = True

    # Run multiple greedy restarts
    # We use a fixed number of restarts or until time runs out
    # Each restart is very fast with vectorization, so we can do many.
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r)
        candidates = list(range(N))
        random.shuffle(candidates)
        
        # covered is a boolean numpy array of size N
        covered = np.zeros(N, dtype=bool)
        chosen = []
        
        # We need to check if all are covered. 
        # np.all(covered) is expensive to check every step? 
        # Better to track count or just check if any False remain.
        # Let's track uncovered count or just use np.any(~covered).
        
        while np.any(~covered):
            # Calculate gain for all candidates at once
            # gain[i] = number of uncovered words in sphere of i
            # uncovered = ~covered
            # gains = spheres_arr[candidates, :].dot(uncovered)
            # But we only care about candidates.
            
            # Vectorized gain calculation
            # spheres_arr is (N, N). uncovered is (N,).
            # gains_all = spheres_arr.dot(uncovered) -> (N,)
            # Then select for candidates.
            # However, computing full dot product every step is N*N operations.
            # N=729, N^2 = 531k. This is very fast.
            
            uncovered = ~covered
            # Compute gains for all possible codewords
            gains = spheres_arr.dot(uncovered)
            
            # Select max gain among candidates
            # We can just take max over candidates
            # candidate_gains = gains[candidates]
            # max_idx_in_candidates = np.argmax(candidate_gains)
            # best_candidate = candidates[max_idx_in_candidates]
            
            # Optimization: If we maintain a list of active candidates, we can reduce size.
            # But for now, let's just use the full array and pick from candidates.
            
            candidate_gains = gains[candidates]
            max_idx = np.argmax(candidate_gains)
            best_gain = candidate_gains[max_idx]
            
            if best_gain == 0:
                break
                
            best_candidate = candidates[max_idx]
            
            chosen.append(best_candidate)
            # Update covered
            covered |= spheres_arr[best_candidate]
            
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_code = [all_words[i] for i in chosen]
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in best_code]
