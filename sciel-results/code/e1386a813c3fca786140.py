"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _compute_max_ap_abs_sum(signs):
    """Compute the maximum absolute signed sum over all arithmetic progressions in [1, N].
    signs: list or array of 256 values in {-1, 1}.
    Returns the maximum absolute sum (integer).
    """
    s = np.array(signs, dtype=np.int8)
    max_abs = 0
    # For each step d, compute all AP sums efficiently using prefix-like accumulation
    # An AP with start a (1-indexed), step d, length L has indices a-1, a-1+d, a-1+2d, ...
    # We can iterate over d and for each residue class mod d, compute running sums.
    
    for d in range(1, N + 1):
        for start in range(d):
            # indices: start, start+d, start+2d, ...
            idx = list(range(start, N, d))
            if len(idx) < 2:
                continue
            vals = s[idx]
            # Compute all contiguous subarray sums of vals
            # The APs with this step and this residue class are contiguous subarrays of vals
            # We need the max absolute sum of any contiguous subarray
            # Use Kadane-like for both max and min
            cur_max = vals[0]
            cur_min = vals[0]
            best_max = vals[0]
            best_min = vals[0]
            for v in vals[1:]:
                cur_max = max(v, cur_max + v)
                cur_min = min(v, cur_min + v)
                best_max = max(best_max, cur_max)
                best_min = min(best_min, cur_min)
            max_abs = max(max_abs, abs(best_max), abs(best_min))
    
    return int(max_abs)

def _quick_eval(signs):
    """Fast evaluation: only check small steps and limited lengths to speed up local search."""
    s = np.array(signs, dtype=np.int8)
    max_abs = 0
    # Check steps 1 through 64 (most APs are short for large steps)
    for d in range(1, 65):
        for start in range(d):
            idx = list(range(start, N, d))
            if len(idx) < 2:
                continue
            vals = s[idx]
            cur_max = vals[0]
            cur_min = vals[0]
            best_max = vals[0]
            best_min = vals[0]
            for v in vals[1:]:
                cur_max = max(v, cur_max + v)
                cur_min = min(v, cur_min + v)
                best_max = max(best_max, cur_max)
                best_min = min(best_min, cur_min)
            max_abs = max(max_abs, abs(best_max), abs(best_min))
            if max_abs >= 30:  # early exit if already bad
                return max_abs
    # Also check a sample of larger steps
    for d in range(65, N + 1, 7):
        for start in range(d):
            idx = list(range(start, N, d))
            if len(idx) < 2:
                continue
            vals = s[idx]
            cur_max = vals[0]
            cur_min = vals[0]
            best_max = vals[0]
            best_min = vals[0]
            for v in vals[1:]:
                cur_max = max(v, cur_max + v)
                cur_min = min(v, cur_min + v)
                best_max = max(best_max, cur_max)
                best_min = min(best_min, cur_min)
            max_abs = max(max_abs, abs(best_max), abs(best_min))
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    rng = random.Random(42)
    
    # Start with a random initial solution
    best_signs = [rng.choice((-1, 1)) for _ in range(N)]
    best_score = _compute_max_ap_abs_sum(best_signs)
    
    # Local search: flip individual positions
    # Use a tabu-like approach: avoid recently flipped positions
    max_iters = 500000
    iter_count = 0
    no_improve = 0
    no_improve_limit = 2000
    
    # Precompute for quick evaluation
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        
        # Random flip
        pos = rng.randrange(N)
        old_val = best_signs[pos]
        best_signs[pos] = -old_val
        
        # Quick evaluation
        quick_score = _quick_eval(best_signs)
        
        if quick_score < best_score:
            # Full evaluation to confirm
            full_score = _compute_max_ap_abs_sum(best_signs)
            if full_score < best_score:
                best_score = full_score
                no_improve = 0
            else:
                best_signs[pos] = old_val
                no_improve += 1
        else:
            best_signs[pos] = old_val
            no_improve += 1
        
        iter_count += 1
        
        # Occasional restart if stuck
        if no_improve > no_improve_limit:
            no_improve = 0
            # Perturbation: flip a random subset
            for _ in range(rng.randint(3, 10)):
                p = rng.randrange(N)
                best_signs[p] = -best_signs[p]
            # Re-evaluate
            new_score = _compute_max_ap_abs_sum(best_signs)
            if new_score > best_score:
                # Revert to best
                best_signs = [rng.choice((-1, 1)) for _ in range(N)]
                best_score = _compute_max_ap_abs_sum(best_signs)
    
    # Final full evaluation
    final_score = _compute_max_ap_abs_sum(best_signs)
    if final_score < best_score:
        best_score = final_score
    
    return best_signs

# EVOLVE-BLOCK-END
