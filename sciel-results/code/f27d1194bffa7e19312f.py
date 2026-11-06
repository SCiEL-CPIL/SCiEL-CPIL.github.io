import random
import numpy as np
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Precompute all APs and their indices
    ap_indices_list = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices_list.append(indices)
    
    num_aps = len(ap_indices_list)
    
    # Build flat incidence structure: for each position, list of AP indices
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    # Build flat arrays for fast vectorized access
    # ap_starts[pos] = start index in ap_flat for position pos
    # ap_flat = concatenated list of AP indices per position
    ap_starts = np.zeros(N + 1, dtype=np.int64)
    total_entries = 0
    for p in range(N):
        ap_starts[p] = total_entries
        total_entries += len(pos_to_ap[p])
    ap_starts[N] = total_entries
    
    ap_flat = np.zeros(total_entries, dtype=np.int32)
    for p in range(N):
        start = ap_starts[p]
        end = ap_starts[p + 1]
        ap_flat[start:end] = pos_to_ap[p]
    
    # Build AP membership matrix for initial computation (sparse approach)
    # For each AP, store its indices as a flat array
    ap_member_starts = np.zeros(num_aps + 1, dtype=np.int64)
    total_members = 0
    for ap_idx, indices in enumerate(ap_indices_list):
        ap_member_starts[ap_idx] = total_members
        total_members += len(indices)
    ap_member_starts[num_aps] = total_members
    
    ap_members_flat = np.zeros(total_members, dtype=np.int32)
    for ap_idx, indices in enumerate(ap_indices_list):
        start = ap_member_starts[ap_idx]
        end = ap_member_starts[ap_idx + 1]
        ap_members_flat[start:end] = np.array(indices, dtype=np.int32) - 1
    
    # Precompute per-AP length for quick sum computation
    ap_lengths = np.diff(ap_member_starts).astype(np.int32)
    
    def compute_initial_sums(s):
        """Compute all AP sums using the flat membership arrays."""
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx in range(num_aps):
            start = ap_member_starts[ap_idx]
            end = ap_member_starts[ap_idx + 1]
            S[ap_idx] = np.sum(s[ap_members_flat[start:end]])
        return S
    
    def compute_initial_sums_fast(s):
        """Faster initial sum computation using vectorized ops on flat array."""
        # s is int8 array of length N
        # For each AP, sum s[members]
        S = np.zeros(num_aps, dtype=np.int16)
        # Use the fact that we can gather: for each AP, the members are contiguous in ap_members_flat
        # But we need to index s by those members
        # Vectorized approach: create a single gather
        S = np.add.reduceat(s[ap_members_flat], ap_member_starts[:-1]).astype(np.int16)
        return S
    
    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        return hist, int(np.max(abs_S))
    
    def get_affected_aps(j):
        """Get the flat array of AP indices affected by flipping position j."""
        start = ap_starts[j]
        end = ap_starts[j + 1]
        return ap_flat[start:end]
    
    def try_flip(j, s, S, hist, current_max, temp, rng):
        """Try flipping position j. Returns (accepted, new_max)."""
        old_val = s[j]
        delta = -2 * old_val
        
        affected = get_affected_aps(j)
        n_affected = len(affected)
        if n_affected == 0:
            return True, current_max
        
        old_sums = S[affected]
        new_sums = old_sums + delta
        
        old_abs = np.abs(old_sums)
        new_abs = np.abs(new_sums)
        
        # Update histogram
        np.add.at(hist, old_abs, -1)
        np.add.at(hist, new_abs, 1)
        
        max_new_abs_in_affected = int(np.max(new_abs))
        
        old_max = current_max
        if hist[old_max] == 0:
            if max_new_abs_in_affected > old_max:
                new_max = max_new_abs_in_affected
            else:
                scan_result = old_max - 1
                while scan_result > 0 and hist[scan_result] == 0:
                    scan_result -= 1
                new_max = max(max_new_abs_in_affected, scan_result)
        else:
            new_max = max(old_max, max_new_abs_in_affected)
        
        cost_diff = new_max - old_max
        
        if cost_diff <= 0:
            accept = True
        else:
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            S[affected] = new_sums
            return True, new_max
        else:
            # Revert (S wasn't actually updated yet in this version)
            return False, current_max
    
    def local_search(s, S, hist, current_max, end_time, rng):
        """Simulated annealing with vectorized updates."""
        temp = 3.0
        cooling_rate = 0.9995
        steps = 0
        
        while time.time() < end_time:
            j = rng.randint(0, N - 1)
            
            accepted, new_max = try_flip(j, s, S, hist, current_max, temp, rng)
            
            if accepted:
                current_max = new_max
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max
    
    def greedy_improve(s, S, hist, current_max, end_time, rng, max_rounds=50):
        """Greedy hill-climbing with vectorized updates."""
        for _ in range(max_rounds):
            if time.time() > end_time:
                break
            
            improved = False
            for j in range(N):
                old_val = s[j]
                delta = -2 * old_val
                
                affected = get_affected_aps(j)
                n_affected = len(affected)
                if n_affected == 0:
                    continue
                
                old_sums = S[affected]
                new_sums = old_sums + delta
                new_abs = np.abs(new_sums)
                max_new_abs_in_affected = int(np.max(new_abs))
                
                old_max = current_max
                # Compute what the histogram would look like
                old_abs = np.abs(old_sums)
                
                # Check if we'd improve
                # New max would be:
                if hist[old_max] == 0:
                    if max_new_abs_in_affected > old_max:
                        new_max = max_new_abs_in_affected
                    else:
                        # Need to check if removing old_abs values drops the max
                        # Simplified: just compute
                        new_max = max_new_abs_in_affected
                        # But other APs not affected might still have old_max
                        # Check: are there any non-affected APs with abs == old_max?
                        # hist[old_max] counts all APs with abs == old_max
                        # affected APs with old_abs == old_max: count them
                        count_at_old_max_in_affected = int(np.sum(old_abs == old_max))
                        remaining_at_old_max = hist[old_max] - count_at_old_max_in_affected
                        if remaining_at_old_max > 0:
                            new_max = old_max
                        else:
                            # Scan down
                            scan_result = old_max - 1
                            while scan_result > 0:
                                # Check if any AP has abs == scan_result
                                # This is complex; approximate
                                break
                            new_max = max(max_new_abs_in_affected, scan_result)
                else:
                    new_max = max(old_max, max_new_abs_in_affected)
                
                if new_max < current_max:
                    # Apply flip
                    np.add.at(hist, old_abs, -1)
                    np.add.at(hist, new_abs, 1)
                    S[affected] = new_sums
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
            
            if not improved:
                break
        
        return s, S, hist, current_max
    
    def perturb(s, S, hist, current_max, rng, num_flips=8):
        """Randomly flip some positions to escape local minima."""
        for _ in range(num_flips):
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            affected = get_affected_aps(j)
            if len(affected) == 0:
                continue
            
            old_sums = S[affected]
            new_sums = old_sums + delta
            old_abs = np.abs(old_sums)
            new_abs = np.abs(new_sums)
            
            np.add.at(hist, old_abs, -1)
            np.add.at(hist, new_abs, 1)
            S[affected] = new_sums
            s[j] = -old_val
        
        # Recompute max
        abs_S = np.abs(S)
        current_max = int(np.max(abs_S))
        hist = np.zeros(N + 1, dtype=np.int32)
        np.add.at(hist, abs_S, 1)
        
        return s, S, hist, current_max
    
    # Multiple restarts with different strategies
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Initialize with random signs
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        S = compute_initial_sums_fast(s)
        hist, current_max = build_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # First: greedy improvement to get a good starting point
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            greedy_end = time.time() + remaining_time * 0.15
            s, S, hist, current_max = greedy_improve(s, S, hist, current_max, greedy_end, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Then: simulated annealing
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            sa_end = time.time() + remaining_time * 0.55
            s, S, hist, current_max = local_search(s, S, hist, current_max, sa_end, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Perturbation and re-optimization for escape from local minima
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = perturb(s, S, hist, current_max, rng, num_flips=10)
            
            # Greedy + SA again
            if time.time() < deadline - 3.0:
                greedy_end2 = time.time() + (deadline - time.time()) * 0.25
                s, S, hist, current_max = greedy_improve(s, S, hist, current_max, greedy_end2, rng)
            
            if time.time() < deadline - 2.0:
                sa_end2 = time.time() + (deadline - time.time()) * 0.7
                s, S, hist, current_max = local_search(s, S, hist, current_max, sa_end2, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Try a few more restarts if time permits
        if time.time() > deadline - 3.0:
            break
    
    # Final polish on the best sequence found
    if best_seq is not None and time.time() < deadline - 2.0:
        s_final = best_seq.copy()
        S_final = compute_initial_sums_fast(s_final)
        hist_final, current_max_final = build_hist(S_final)
        
        # Greedy polish
        s_final, S_final, hist_final, current_max_final = greedy_improve(
            s_final, S_final, hist_final, current_max_final, deadline - 1.0, rng, max_rounds=20
        )
        
        # Short SA polish
        if time.time() < deadline - 1.0:
            s_final, S_final, hist_final, current_max_final = local_search(
                s_final, S_final, hist_final, current_max_final, deadline - 0.5, rng
            )
        
        if current_max_final < best_score:
            best_score = current_max_final
            best_seq = s_final.copy()
    
    return [int(x) for x in best_seq]
