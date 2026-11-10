import time
import random
import numpy as np

def _max_abs_ap_sum(signs, threshold=None):
    """
    Compute the maximum absolute sum over any arithmetic progression
    contained in [1, n].
    signs: list or numpy array of length n, values -1 or 1.
    threshold: if provided, early-exit and return threshold+1 if exceeded.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_sum = 0
    
    for d in range(1, n + 1):
        # Process all chains for this step size
        for s in range(d):
            idx = np.arange(s, n, d)
            if len(idx) == 0:
                continue
            chain = signs_arr[idx]
            prefix = np.concatenate(([0], np.cumsum(chain)))
            current_max = int(prefix.max() - prefix.min())
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    
    return max_sum

def _get_tight_ap_indices(signs, top_k=50):
    """
    Find indices that appear in APs with sums close to the maximum.
    Returns a list of candidate indices to flip, weighted by tightness.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    # Compute all AP sums and track which are near max
    ap_sums = []  # (abs_sum, start, step)
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            idx = np.arange(s, n, d)
            if len(idx) == 0:
                continue
            chain = signs_arr[idx]
            prefix = np.concatenate(([0], np.cumsum(chain)))
            current_max = int(prefix.max() - prefix.min())
            if current_max > 0:
                ap_sums.append((current_max, s, d))
            if current_max > max_sum:
                max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    # Keep APs within 1 of max (tight APs)
    tight_threshold = max_sum - 1
    tight_aps = [(s, d) for (val, s, d) in ap_sums if val >= tight_threshold]
    
    # Count how many tight APs each index appears in
    idx_count = np.zeros(n, dtype=np.int32)
    for s, d in tight_aps:
        idx = np.arange(s, n, d)
        idx_count[idx] += 1
    
    # Return indices sorted by count (most constrained first)
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 256
    rng = random.Random(42)
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # 1. Alternating
    initializations.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # 2. Thue-Morse inspired
    tm = []
    for i in range(n):
        # Thue-Morse: parity of number of 1s in binary representation
        tm.append(1 if bin(i).count('1') % 2 == 0 else -1)
    initializations.append(tm)
    
    # 3. Random
    initializations.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # 4. Another random
    initializations.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # 5. Block pattern: +1+1-1-1 repeating
    initializations.append([1 if (i // 2) % 2 == 0 else -1 for i in range(n)])
    
    best_signs = None
    best_score = float('inf')
    
    for init in initializations:
        score = _max_abs_ap_sum(init)
        if score < best_score:
            best_score = score
            best_signs = init[:]
    
    # Local Search with guided flips
    current_signs = best_signs[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 80  # Restart if stuck
    
    # Track how long we've been running for adaptive restart
    restart_interval = 2000  # Number of iterations before forced restart
    
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Periodic restart with new random start
        if no_improve_count > max_no_improve or iteration % restart_interval == 0:
            # Generate a new random start, possibly perturbed from best
            if rng.random() < 0.3:
                # Perturb the best solution
                current_signs = best_signs[:]
                for _ in range(rng.randint(5, 20)):
                    idx = rng.randint(0, n - 1)
                    current_signs[idx] = -current_signs[idx]
            else:
                current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            continue
        
        # Get candidate indices to flip (guided by tight APs)
        if iteration % 50 == 0:
            # Recompute tight APs periodically
            candidates = _get_tight_ap_indices(current_signs, top_k=30)
            if not candidates:
                candidates = list(range(n))
        else:
            # Use a mix: 70% from tight APs, 30% random
            if rng.random() < 0.7:
                # Pick from previously computed candidates if available
                if 'candidates' in dir() and candidates:
                    idx = candidates[rng.randint(0, len(candidates) - 1)]
                else:
                    idx = rng.randint(0, n - 1)
            else:
                idx = rng.randint(0, n - 1)
        
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        # Use early-exit evaluation with current_score as threshold
        new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_signs = current_signs[:]
        else:
            # Revert
            current_signs[idx] = old_val
            no_improve_count += 1
    
    return best_signs
