import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill

    # 1. Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for s in range(n):
            indices = list(range(s, n, d))
            if indices:
                ap_indices.append(indices)
                
    num_aps = len(ap_indices)
    
    # Build impact map: for each index i, which APs contain it?
    impact = [[] for _ in range(n)]
    for ap_idx, indices in enumerate(ap_indices):
        for i in indices:
            impact[i].append(ap_idx)

    # 2. Initialization
    rng = random.Random(42)
    
    def thue_morse(n):
        return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
        
    signs = thue_morse(n)
    
    # Calculate initial sums
    ap_sums = [0] * num_aps
    for ap_idx, indices in enumerate(ap_indices):
        s = 0
        for i in indices:
            s += signs[i]
        ap_sums[ap_idx] = s
            
    # Build histogram of absolute sums
    max_possible = n
    hist = [0] * (max_possible + 1)
    for s in ap_sums:
        hist[abs(s)] += 1
        
    # Find current max discrepancy
    current_max = 0
    for j in range(max_possible, -1, -1):
        if hist[j] > 0:
            current_max = j
            break
            
    best_signs = signs[:]
    best_disc = current_max
    
    def get_max_from_hist(hist):
        for j in range(max_possible, -1, -1):
            if hist[j] > 0:
                return j
        return 0

    # State update helpers
    def do_flip(idx):
        """Applies flip to signs, ap_sums, hist. Returns (old_sign, old_sums_dict, old_hist_copy, old_max, new_max)"""
        old_sign = signs[idx]
        new_sign = -old_sign
        delta_sum = -2 * old_sign
        
        affected_aps = impact[idx]
        old_sums = {}
        old_hist = hist[:]
        old_max = current_max
        
        for k in affected_aps:
            old_s = ap_sums[k]
            old_sums[k] = old_s
            old_abs = abs(old_s)
            hist[old_abs] -= 1
            
            new_s = old_s + delta_sum
            ap_sums[k] = new_s
            new_abs = abs(new_s)
            hist[new_abs] += 1
            
        signs[idx] = new_sign
        new_max = get_max_from_hist(hist)
        
        return old_sign, old_sums, old_hist, old_max, new_max

    def undo_flip(idx, old_sign, old_sums, old_hist, old_max):
        """Reverts flip. Assumes current state is the flipped state."""
        signs[idx] = old_sign
        # Restore ap_sums
        for k, val in old_sums.items():
            ap_sums[k] = val
        # Restore hist
        hist[:] = old_hist
        # current_max is restored by caller or we just return it
        return old_max

    # ILS Loop
    while time.time() - start_time < time_limit:
        # Phase 1: Greedy Local Search
        # Try to improve or move sideways
        for _ in range(50): # Max 50 greedy steps per iteration
            if time.time() - start_time > time_limit:
                break
                
            # Find worst AP
            # Optimization: If current_max is small, we might not need to scan all APs?
            # But scanning 1000 is fast.
            worst_aps = [k for k in range(num_aps) if abs(ap_sums[k]) == current_max]
            
            if not worst_aps:
                break
            
            target_ap = rng.choice(worst_aps)
            target_indices = ap_indices[target_ap]
            target_sum = ap_sums[target_ap]
            
            # Choose candidate indices in this AP that would reduce |sum|
            candidates = []
            if target_sum > 0:
                for i in target_indices:
                    if signs[i] == 1:
                        candidates.append(i)
            elif target_sum < 0:
                for i in target_indices:
                    if signs[i] == -1:
                        candidates.append(i)
            else:
                candidates = target_indices[:]
                
            if not candidates:
                break
                
            idx = rng.choice(candidates)
            
            # Simulate flip
            old_sign, old_sums, old_hist, old_max, new_max = do_flip(idx)
            
            if new_max < current_max:
                # Improvement
                current_max = new_max
                if current_max < best_disc:
                    best_disc = current_max
                    best_signs = signs[:]
            elif new_max == current_max:
                # Sideways move
                if rng.random() < 0.3: # Accept 30% of sideways moves
                    current_max = new_max
                else:
                    # Revert
                    current_max = undo_flip(idx, old_sign, old_sums, old_hist, old_max)
            else:
                # Revert
                current_max = undo_flip(idx, old_sign, old_sums, old_hist, old_max)

        # Phase 2: Perturbation
        # If we are stuck or just to explore, perturb
        # We perturb by flipping K random bits
        k_perturb = rng.randint(2, 5)
        for _ in range(k_perturb):
            idx = rng.randint(0, n - 1)
            old_sign, old_sums, old_hist, old_max, new_max = do_flip(idx)
            # We do NOT revert perturbation. We accept the new state, even if worse.
            current_max = new_max
            
        # If we are much worse than best, we might consider restarting from best?
        # But let the greedy phase try to fix it first.
        # If current_max is very high (e.g., > best_disc + 10), restart from best.
        if current_max > best_disc + 10:
            # Restart from best
            signs = best_signs[:]
            # Recalculate sums and hist
            ap_sums = [0] * num_aps
            for ap_idx, indices in enumerate(ap_indices):
                s = 0
                for i in indices:
                    s += signs[i]
                ap_sums[ap_idx] = s
            hist = [0] * (max_possible + 1)
            for s in ap_sums:
                hist[abs(s)] += 1
            current_max = get_max_from_hist(hist)

    return best_signs
