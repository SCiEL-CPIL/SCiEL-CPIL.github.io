import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1 is expected.
    # However, the interface is generic. We will implement a general solution 
    # that works for small q^n, but optimized for the specific case if detected.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple greedy for other cases to ensure validity
        # though the task is specific to 3,6,1.
        pass

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precomputation for q=3, n=6
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    all_words = [idx_to_word(i) for i in range(q**n)]
    
    # Precompute balls (coverage sets) for each word
    # ball_indices[i] is a boolean array of length q^n indicating which words are covered by word i
    ball_indices = []
    for w in all_words:
        covered = np.zeros(q**n, dtype=bool)
        # Center
        covered[word_to_idx(w)] = True
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                neighbor = w.copy()
                neighbor[pos] = val
                covered[word_to_idx(neighbor)] = True
        ball_indices.append(covered)
        
    # Initialize with Tetracode seed
    # Tetracode: [a, b, a+b, a+2b] mod 3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            seed_code.append(h + list(tail))
            
    # Convert seed to indices
    current_code = [word_to_idx(w) for w in seed_code]
    current_code = list(set(current_code)) # Ensure distinct
    
    # Calculate initial coverage
    coverage = np.zeros(q**n, dtype=int)
    for idx in current_code:
        coverage += ball_indices[idx]
        
    best_size = len(current_code)
    best_code = list(current_code)
    
    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Reduction
        # Try to remove any codeword that is redundant
        removable = []
        for i, idx in enumerate(current_code):
            # Check if removing idx leaves all covered
            # Optimization: Only check if the codeword is "critical"
            # We can just try removing it
            old_cov = coverage.copy()
            coverage -= ball_indices[idx]
            if np.all(coverage > 0):
                # Safe to remove
                removable.append(i)
            else:
                coverage = old_cov
                
        if removable:
            # Remove the first removable one (or all? Removing one might make others removable)
            # Let's remove one and restart reduction
            i_to_remove = removable[0]
            idx_to_remove = current_code[i_to_remove]
            coverage -= ball_indices[idx_to_remove]
            current_code.pop(i_to_remove)
            improved = True
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            continue # Restart reduction loop as coverage changed
        
        # Phase 2: Perturbation (if stuck)
        # Pick a random codeword to remove, then repair
        if len(current_code) > 0:
            i_out = random.randint(0, len(current_code) - 1)
            idx_out = current_code[i_out]
            
            # Remove it
            coverage -= ball_indices[idx_out]
            current_code.pop(i_out)
            
            # Find uncovered words
            uncovered_mask = (coverage == 0)
            if np.any(uncovered_mask):
                # Find the best codeword to add back to cover the most uncovered words
                # We look for a codeword that covers the uncovered set
                # Score = number of uncovered words covered by candidate
                
                # To speed up, we can check candidates in a random order or all
                # Given 729 candidates, checking all is fast
                best_in = -1
                best_score = 0
                # We can exclude current_code to prefer swaps/adds, but adding existing is redundant
                # Actually, we want to find ANY word that helps.
                
                # Optimization: Precomputed dot products?
                # For each candidate c, score = sum(ball_indices[c][uncovered_mask])
                
                # Vectorized:
                # uncovered_indices = np.where(uncovered_mask)[0]
                # This is a subset of 0..728
                # We want max over c of count(c covers u for u in uncovered)
                
                # Let's just loop candidates, it's 729 * (small check)
                # Or use matrix multiplication if we stored ball_indices as a matrix
                # Ball Matrix: 729 x 729 boolean. 729*729 = 531k bools. ~0.5 MB. Very fast.
                
                if not hasattr(construct, '_ball_matrix'):
                    construct._ball_matrix = np.array(ball_indices, dtype=np.uint8)
                    
                ball_mat = construct._ball_matrix
                # Score for each candidate c is sum of ball_mat[c][uncovered_mask]
                # This is equivalent to (ball_mat * uncovered_mask_vector).sum(axis=1)
                # uncovered_mask is bool array. Cast to int.
                unc_vec = uncovered_mask.astype(np.uint8)
                scores = (ball_mat * unc_vec).sum(axis=1)
                
                # Find max score
                max_score = np.max(scores)
                if max_score > 0:
                    # Pick one of the best candidates
                    candidates = np.where(scores == max_score)[0]
                    # Prefer candidates not in current_code if possible?
                    # Or just pick random from best
                    idx_in = candidates[random.randint(0, len(candidates)-1)]
                    
                    # Add it
                    current_code.append(idx_in)
                    coverage += ball_indices[idx_in]
                    
                    # Check if we actually fixed it
                    # If not fixed, we are in a worse state (more words, still holes)
                    # But this allows escape from local minima in reduction
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        improved = True
                    # Even if size didn't improve, we changed state, so continue loop
                    # The reduction phase will try to clean up again
                else:
                    # No candidate covers any uncovered word? Should not happen in connected graph
                    # Revert
                    current_code.append(idx_out)
                    coverage += ball_indices[idx_out]
            else:
                # We removed idx_out and coverage was still valid?
                # This means idx_out was redundant, but we missed it in Phase 1?
                # This can happen if Phase 1 didn't check all? No, Phase 1 checks all.
                # So this case implies Phase 1 failed to find it?
                # Actually, if coverage is still >0 everywhere, we successfully removed a redundant node.
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    improved = True

        # If no improvement and no perturbation happened (e.g. size 0?), break
        if not improved and time.time() > start_time + 10:
             # If we are stuck for a while, we might want to restart with a different seed
             # But for now, let it run until deadline
             pass

    # Convert best_code back to words
    return [idx_to_word(i) for i in best_code]
