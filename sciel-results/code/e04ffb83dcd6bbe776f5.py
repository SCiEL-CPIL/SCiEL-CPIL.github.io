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
    ap_indices = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices.append(indices)
    
    num_aps = len(ap_indices)
    
    # Map position to APs containing it
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    # Convert to numpy arrays for faster access
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in pos_to_ap]
    
    def compute_initial_sums(s):
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S[ap_idx] = np.sum(s[s_indices])
        return S
    
    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        return hist, int(np.max(abs_S))
    
    def try_flip(j, s, S, hist, current_max, temp, rng):
        """Try flipping position j. Returns (accepted, new_max, delta_val)."""
        old_val = s[j]
        delta = -2 * old_val
        
        ap_idx_list = pos_to_ap_np[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        for ap_idx in ap_idx_list:
            old_sum = S[ap_idx]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            S[ap_idx] = new_sum
            
            if new_abs > max_new_abs_in_affected:
                max_new_abs_in_affected = new_abs
        
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
            return True, new_max, delta
        else:
            for ap_idx in ap_idx_list:
                new_sum = S[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
            
            return False, current_max, delta
    
    def local_search(s, S, hist, current_max, deadline, rng, max_steps=20000):
        """Simulated annealing with acceptance criteria."""
        temp = 3.0
        cooling_rate = 0.997
        steps = 0
        
        while time.time() < deadline and steps < max_steps:
            j = rng.randint(0, N - 1)
            
            accepted, new_max, _ = try_flip(j, s, S, hist, current_max, temp, rng)
            
            if accepted:
                current_max = new_max
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max
    
    def greedy_improve(s, S, hist, current_max, deadline, rng, max_rounds=5):
        """Greedy hill-climbing: repeatedly try all positions, accept any improvement."""
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for j in range(N):
                old_max = current_max
                old_val = s[j]
                delta = -2 * old_val
                
                ap_idx_list = pos_to_ap_np[j]
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_idx_list:
                    old_sum = S[ap_idx]
                    new_sum = old_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs > max_new_abs_in_affected:
                        max_new_abs_in_affected = new_abs
                
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
                
                if new_max < current_max:
                    # Accept: actually apply the flip
                    for ap_idx in ap_idx_list:
                        old_sum = S[ap_idx]
                        new_sum = old_sum + delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        
                        S[ap_idx] = new_sum
                    
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
            
            ap_idx_list = pos_to_ap_np[j]
            
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
            
            s[j] = -old_val
        
        # Recompute max
        abs_S = np.abs(S)
        current_max = int(np.max(abs_S))
        hist = np.zeros(N + 1, dtype=np.int32)
        np.add.at(hist, abs_S, 1)
        
        return s, S, hist, current_max
    
    def randomized_greedy_construct(rng):
        """
        Multi-start randomized greedy: construct position by position,
        at each step choose the sign that minimizes the max abs sum over
        all APs that include the current position and are fully determined
        (all positions up to current are set).
        """
        s = np.zeros(N, dtype=np.int8)
        
        # Track sums incrementally
        S = np.zeros(num_aps, dtype=np.int16)
        hist = np.zeros(N + 1, dtype=np.int32)
        current_max = 0
        
        for pos in range(N):
            # Try both signs
            best_sign = 1
            best_max = float('inf')
            
            for sign in [1, -1]:
                # Compute what the max would be if we set s[pos] = sign
                ap_idx_list = pos_to_ap_np[pos]
                
                # Only consider APs where all positions are <= pos (fully determined)
                # Actually, we need to check if all indices in the AP are <= pos+1
                # For simplicity, let's just compute the change for all APs containing pos
                # and track the max
                
                max_new_abs = 0
                for ap_idx in ap_idx_list:
                    indices = ap_indices[ap_idx]
                    # Check if all indices are <= pos+1
                    if all(idx <= pos + 1 for idx in indices):
                        old_sum = S[ap_idx]
                        new_sum = old_sum + sign
                        new_abs = abs(new_sum)
                        if new_abs > max_new_abs:
                            max_new_abs = new_abs
                
                # The new max would be max(current_max, max_new_abs)
                candidate_max = max(current_max, max_new_abs)
                
                if candidate_max < best_max:
                    best_max = candidate_max
                    best_sign = sign
            
            s[pos] = best_sign
            
            # Update S and hist
            ap_idx_list = pos_to_ap_np[pos]
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + best_sign
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
            
            current_max = best_max
        
        return s, S, hist, current_max
    
    # Multiple restarts with different strategies
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Alternate between random init and greedy construct
        if num_restarts % 3 == 0:
            # Use randomized greedy construction
            s, S, hist, current_max = randomized_greedy_construct(rng)
        else:
            # Initialize with random signs
            s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
            S = compute_initial_sums(s)
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
        S_final = compute_initial_sums(s_final)
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
