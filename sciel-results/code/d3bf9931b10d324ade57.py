"""Low discrepancy on all arithmetic progressions - Greedy Worst-AP Repair."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_max_ap_discrepancy_vectorized(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns the maximum absolute discrepancy and the indices of one such worst AP 
    (start, step, length) to allow targeted repair.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    
    max_abs = 0
    best_start = 0
    best_step = 1
    best_len = 1
    
    for d in range(1, n):
        max_start = n - d
        if max_start <= 0:
            continue
            
        for i in range(max_start):
            seq = s[i:n:d]
            # seq length is at least 2
            cumsum = np.cumsum(seq)
            abs_cumsum = np.abs(cumsum)
            current_max = np.max(abs_cumsum)
            
            if current_max > max_abs:
                max_abs = current_max
                # Find the length k where this max occurred
                k = np.argmax(abs_cumsum) + 1
                best_start = i
                best_step = d
                best_len = k
                # Early exit optimization: if we hit a very high value, 
                # we might not need to check all, but for repair we need the specific one.
                # Actually, we just need ONE worst AP to repair.
                # We can break out of loops if we are confident, but let's finish to be safe 
                # or just return the first one found that is high?
                # For greedy repair, finding the current global max is best.
                # We can't easily break early without knowing if a higher one exists.
                # But n=256 is small.
                
    return int(max_abs), best_start, best_step, best_len

def _eval_single(signs):
    """Fast evaluation returning just the max discrepancy."""
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    max_abs = 0
    for d in range(1, n):
        max_start = n - d
        if max_start <= 0:
            continue
        for i in range(max_start):
            seq = s[i:n:d]
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                # Optional early exit if max_abs reaches theoretical max? 
                # Max possible is n. If we hit n, it's as bad as it gets.
                if max_abs >= n:
                    return int(max_abs)
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int32)
    
    # Evaluate initial
    current_disc, cur_start, cur_step, cur_len = _get_max_ap_discrepancy_vectorized(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Greedy Repair Loop
    # We will try to reduce the discrepancy by flipping elements in the worst AP.
    # If flipping any element in the worst AP doesn't reduce the global max,
    # we might try random flips or other strategies, but pure greedy on worst AP
    # is a strong baseline for this specific metric.
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Identify current worst AP
        current_disc, cur_start, cur_step, cur_len = _get_max_ap_discrepancy_vectorized(signs)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs.copy()
            no_improve_count = 0
            if best_disc == 1:
                break
        else:
            no_improve_count += 1
            # If stuck for a while, perturb randomly to escape local minima
            if no_improve_count > 100:
                # Random restart or large jump
                # Let's try flipping a random few signs
                for _ in range(5):
                    idx = rng.integers(0, n)
                    signs[idx] = -signs[idx]
                no_improve_count = 0
                continue

        # Get the elements of the worst AP
        ap_indices = list(range(cur_start, cur_start + cur_len * cur_step, cur_step))
        
        best_move_idx = -1
        best_move_disc = current_disc
        
        # Try flipping each element in the worst AP
        for idx in ap_indices:
            # Flip
            signs[idx] = -signs[idx]
            
            # Evaluate new discrepancy
            # Optimization: We only care if it's better than current_disc.
            # _eval_single returns the max.
            new_disc = _eval_single(signs)
            
            if new_disc < best_move_disc:
                best_move_disc = new_disc
                best_move_idx = idx
            
            # Revert
            signs[idx] = -signs[idx]
        
        if best_move_idx != -1:
            # Apply the best move found
            signs[best_move_idx] = -signs[best_move_idx]
            # Update current state
            # We don't need to re-eval fully here if we trust the local eval, 
            # but the loop will re-eval at the start.
        else:
            # No single flip in the worst AP improves the global max.
            # This is a local minimum for single-flip moves within the worst AP.
            # Strategy: Try flipping a random element elsewhere? 
            # Or accept a move that keeps the max the same but might help later?
            # Greedy usually gets stuck here. 
            # Let's try a "hill climbing" step: pick a random flip that doesn't increase the max.
            # Or just random walk.
            
            # Let's try random flips that keep the discrepancy <= current_disc
            improved = False
            for _ in range(10):
                idx = rng.integers(0, n)
                signs[idx] = -signs[idx]
                new_disc = _eval_single(signs)
                if new_disc <= current_disc:
                    improved = True
                    # Keep it
                    break
                else:
                    # Revert
                    signs[idx] = -signs[idx]
            
            if not improved:
                # If even random flips that don't increase max are rare, 
                # we are likely in a deep local minimum.
                # Force a random flip to escape?
                # This might increase discrepancy temporarily, but SA-style escape is needed.
                # Since we are not doing full SA, let's do a "kick": flip 3 random signs.
                for _ in range(3):
                    idx = rng.integers(0, n)
                    signs[idx] = -signs[idx]
                no_improve_count += 50 # Force a quicker check for stagnation

    return best_signs.tolist()

# EVOLVE-BLOCK-END
