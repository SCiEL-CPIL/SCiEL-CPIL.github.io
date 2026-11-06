import random
import numpy as np
import time
import math

N = 256

def construct():
    deadline = time.time() + 58.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Precompute all APs and their indices - but be smart about it
    # We only need APs that can actually be "worst" - focus on longer APs
    ap_indices = []
    ap_lengths = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices.append(indices)
                ap_lengths.append(l)
    
    num_aps = len(ap_indices)
    ap_lengths = np.array(ap_lengths, dtype=np.int32)
    
    # Build position-to-AP mapping
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    # Convert to numpy arrays
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
    
    def get_worst_positions(S, current_max, k=20):
        """Get positions that are in the worst APs."""
        worst_aps = np.where(np.abs(S) >= current_max - 1)[0]
        if len(worst_aps) == 0:
            return None
        # Collect all positions in these APs
        pos_set = set()
        for ap_idx in worst_aps:
            for idx in ap_indices[ap_idx]:
                pos_set.add(idx - 1)
        return list(pos_set)
    
    def local_search(s, S, hist, current_max, deadline, rng, max_steps=30000):
        """Targeted simulated annealing focusing on worst APs."""
        temp = 2.0
        cooling_rate = 0.998
        steps = 0
        
        while time.time() < deadline and steps < max_steps:
            # Targeted: 70% chance pick from worst positions, 30% random
            if current_max > 1:
                worst_pos = get_worst_positions(S, current_max)
                if worst_pos and rng.random() < 0.7:
                    j = rng.choice(worst_pos)
                else:
                    j = rng.randint(0, N - 1)
            else:
                j = rng.randint(0, N - 1)
            
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
                current_max = new_max
            else:
                for ap_idx in ap_idx_list:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    
                    old_abs = abs(old_sum)
                    new_abs = abs(new_sum)
                    
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    
                    S[ap_idx] = old_sum
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max
    
    def polish(s, S, hist, current_max, deadline, rng, max_steps=10000):
        """Final polish: only accept improvements, no random acceptance."""
        steps = 0
        improved = True
        
        while time.time() < deadline and steps < max_steps and improved:
            improved = False
            for j in range(N):
                old_val = s[j]
                delta = -2 * old_val
                ap_idx_list = pos_to_ap_np[j]
                
                old_max = current_max
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
                
                if new_max < old_max:
                    # Accept and update
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
            
            steps += 1
        
        return s, S, hist, current_max
    
    # Multiple restarts
    num_restarts = 0
    
    while time.time() < deadline - 5.0:
        num_restarts += 1
        
        # Initialize with random signs
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        S = compute_initial_sums(s)
        hist, current_max = build_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            search_deadline = time.time() + remaining_time * 0.7
            s, S, hist, current_max = local_search(s, S, hist, current_max, search_deadline, rng)
            
            if current_max < best_score:
                best_score = current_max
                best_seq = s.copy()
            
            # Polish if time permits
            if time.time() < deadline - 2.0:
                polish_deadline = time.time() + (deadline - time.time()) * 0.8
                s, S, hist, current_max = polish(s, S, hist, current_max, polish_deadline, rng)
            
            if current_max < best_score:
                best_score = current_max
                best_seq = s.copy()
        
        if time.time() > deadline - 5.0:
            break
    
    # Final polish on best solution
    if best_seq is not None and time.time() < deadline - 1.0:
        s = best_seq.copy()
        S = compute_initial_sums(s)
        hist, current_max = build_hist(S)
        polish_deadline = deadline - 0.5
        s, S, hist, current_max = polish(s, S, hist, current_max, polish_deadline, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
    
    return [int(x) for x in best_seq]
