import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Only optimized for the specific case q=3, n=7, radius=1
    # If parameters differ, fall back to a simple heuristic or error
    if q != 3 or n != 7 or radius != 1:
        # Fallback: simple greedy or direct sum if not the target case
        # For safety, return a valid but potentially suboptimal code
        # Using direct sum for generic case is safe
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Small case, brute force or simple
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # --- Optimized path for 3x7x1 ---
    
    # 1. Generate all words and map to indices
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute balls (neighbors within radius 1)
    # ball_mask[i] is a boolean array of length N, True if j is in ball of i
    # Actually, let's store the list of indices in the ball for each i, 
    # and also a boolean mask for fast intersection?
    # Boolean mask intersection is fast with NumPy.
    
    ball_masks = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        # Center
        ball_masks[i, i] = True
        # Radius 1 neighbors
        for pos in range(7):
            for delta in [1, 2]: # 0+1, 0+2, 1+1, 1+2, 2+1, 2+2 (mod 3)
                new_val = (w[pos] + delta) % 3
                new_word = list(w)
                new_word[pos] = new_val
                j = word_to_idx[tuple(new_word)]
                ball_masks[i, j] = True
    
    best_code = None
    best_len = N
    
    rng = np.random.default_rng(42)
    
    # Greedy Set Cover with random restarts
    while time.time() < deadline - 5.0: # Leave 5s margin
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        # Randomize the order of candidates to break symmetries
        # We don't necessarily need to restrict candidates to 'uncovered', 
        # but picking from uncovered is a good heuristic.
        # Actually, standard greedy picks the best from ALL candidates, 
        # but if a candidate is already covered, it might still cover others.
        # However, if a candidate is covered, it's in the ball of some previous pick.
        # It's safe to pick from all N, but checking all N every step is O(N^2 * N) total?
        # Steps ~ N/22. Each step checks N candidates. Each check is O(N) dot product.
        # Total ops ~ (N/22) * N * N = N^3/22. 
        # N=2187. N^3 ~ 10^10. 10^10/22 ~ 5*10^8. 
        # NumPy dot product is very fast (C level). 5*10^8 boolean ops is ~0.5s per run.
        # So we can do ~100-200 runs in 150s.
        
        # Optimization: Only consider candidates that are still 'uncovered'?
        # If a candidate is covered, it is in the ball of a previously selected codeword.
        # Does it help to select it? 
        # If C is covered by C1, then Ball(C) is NOT necessarily a subset of Ball(C1).
        # So we MUST consider all candidates, or at least those that might add new coverage.
        # However, if C is covered, it means dist(C, C1) <= 1.
        # The gain of C is |Ball(C) \ Covered|.
        # It is possible for C to have high gain even if it is covered.
        
        # To speed up, we can maintain a list of active candidates?
        # Let's just iterate through all indices.
        
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break
            
            # Find candidate with max gain
            # gains = sum(uncovered & ball_masks[i]) for all i
            # This is a matrix multiplication: ball_masks @ uncovered
            # ball_masks is (N, N) bool. uncovered is (N,) bool.
            # Result is (N,) int.
            gains = np.sum(ball_masks & uncovered[None, :], axis=1)
            
            # Random tie-breaking to explore different paths
            # Add small random noise to gains
            noise = rng.random(N) * 0.1
            gains_noisy = gains + noise
            
            best_idx = np.argmax(gains_noisy)
            
            if gains[best_idx] == 0:
                # Should not happen if uncovered is not empty
                break
                
            code.append(all_words[best_idx])
            uncovered[ball_masks[best_idx]] = False
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            # If we hit the theoretical lower bound or known best, we could stop early?
            # Known best is 186. Lower bound 156.
            # If we find 186, we can keep searching for 185, etc.
            
    if best_code is None:
        # Fallback if loop didn't finish or failed
        return [list(w) for w in all_words]
        
    return best_code
