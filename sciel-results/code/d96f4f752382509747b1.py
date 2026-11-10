import numpy as np
import time
import random
from math import exp

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 24
    S_search = 8000  # Higher resolution for better peak detection during search
    
    # Precompute z powers for fast delta evaluation
    angles_search = 2 * np.pi * np.arange(S_search) / S_search
    z_powers_search = np.zeros((n, S_search), dtype=complex)
    for k in range(n):
        z_powers_search[k] = np.exp(1j * k * angles_search)
    
    # Initial sequence: Golay-Rudin-Shapiro
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_seq = list(gr_sequence(n))
    
    # Compute initial P values
    P_vals = np.sum(np.array(best_seq)[:, None] * z_powers_search, axis=0)
    current_peak = np.max(np.abs(P_vals))
    best_peak = current_peak
    best_seq_copy = list(best_seq)
    
    # Helper to evaluate peak
    def get_peak(P_vals):
        return np.max(np.abs(P_vals))
    
    # Local search: try single flips, then pairs (from inspiration 1)
    def local_search(c, P_vals, max_iters=5000):
        c = list(c)
        P_vals = P_vals.copy()
        cur_val = get_peak(P_vals)
        best_local_val = cur_val
        best_local_seq = list(c)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try single flips
            for k in range(n):
                old_c = c[k]
                c[k] = -c[k]
                delta_val = -2 * old_c * z_powers_search[k]
                new_P_vals = P_vals + delta_val
                new_val = get_peak(new_P_vals)
                if new_val < cur_val - 1e-12:
                    cur_val = new_val
                    P_vals = new_P_vals
                    if cur_val < best_local_val:
                        best_local_val = cur_val
                        best_local_seq = list(c)
                    improved = True
                    break
                else:
                    c[k] = old_c
            
            if improved:
                continue
            
            # Try pair flips
            pair_improved = False
            for k1 in range(n):
                if pair_improved:
                    break
                for k2 in range(k1+1, n):
                    c1_old = c[k1]
                    c2_old = c[k2]
                    c[k1] = -c[k1]
                    c[k2] = -c[k2]
                    delta_val = -2 * c1_old * z_powers_search[k1] - 2 * c2_old * z_powers_search[k2]
                    new_P_vals = P_vals + delta_val
                    new_val = get_peak(new_P_vals)
                    if new_val < cur_val - 1e-12:
                        cur_val = new_val
                        P_vals = new_P_vals
                        if cur_val < best_local_val:
                            best_local_val = cur_val
                            best_local_seq = list(c)
                        pair_improved = True
                        break
                    else:
                        c[k1] = c1_old
                        c[k2] = c2_old
            
            if not pair_improved:
                break
        
        return best_local_seq, best_local_val
    
    # First, run local search from GR sequence
    best_seq_copy, best_peak = local_search(best_seq_copy, P_vals, max_iters=20000)
    
    # Simulated Annealing with multiple restarts
    T_max = 1.5
    T_min = 1e-6
    alpha = 0.999
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Reset to best found so far with perturbations
        if restart == 0:
            seq = list(best_seq_copy)
        else:
            seq = list(best_seq_copy)
            num_flips = random.randint(3, 8) + (restart // 3)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        
        # Recompute P_vals for this sequence
        P_vals = np.sum(np.array(seq)[:, None] * z_powers_search, axis=0)
        current_peak = get_peak(P_vals)
        
        T = T_max
        local_best_peak = current_peak
        local_best_seq = list(seq)
        
        while time.time() < deadline - 10 and T > T_min:
            # Pick a random index to flip
            idx = random.randint(0, n - 1)
            
            # Calculate delta P_vals
            c = seq[idx]
            delta_val = -2 * c * z_powers_search[idx]
            
            new_P_vals = P_vals + delta_val
            new_peak = get_peak(new_P_vals)
            
            if new_peak < current_peak:
                # Accept
                P_vals = new_P_vals
                current_peak = new_peak
                seq[idx] = -seq[idx]
                
                if current_peak < local_best_peak:
                    local_best_peak = current_peak
                    local_best_seq = list(seq)
            else:
                # Metropolis criterion
                delta_peak = new_peak - current_peak
                if random.random() < exp(-delta_peak / T):
                    P_vals = new_P_vals
                    current_peak = new_peak
                    seq[idx] = -seq[idx]
            
            T *= alpha
        
        # Update global best
        if local_best_peak < best_peak:
            best_peak = local_best_peak
            best_seq_copy = list(local_best_seq)
        
        # Run local search from the SA result
        if time.time() < deadline - 10:
            refined_seq, refined_peak = local_search(local_best_seq, 
                                                     np.sum(np.array(local_best_seq)[:, None] * z_powers_search, axis=0),
                                                     max_iters=10000)
            if refined_peak < best_peak:
                best_peak = refined_peak
                best_seq_copy = list(refined_seq)
    
    # Final high-precision polish pass with even more sampling points
    S_polish = 15000
    angles_polish = 2 * np.pi * np.arange(S_polish) / S_polish
    z_powers_polish = np.zeros((n, S_polish), dtype=complex)
    for k in range(n):
        z_powers_polish[k] = np.exp(1j * k * angles_polish)
    
    seq = list(best_seq_copy)
    P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    current_peak_polish = get_peak(P_vals_polish)
    best_peak_polish = current_peak_polish
    best_seq_polish = list(seq)
    
    # Tight local search with smaller temperature using higher resolution
    T = 0.05
    T_min_polish = 1e-7
    alpha_polish = 0.9995
    
    while time.time() < deadline - 2 and T > T_min_polish:
        idx = random.randint(0, n - 1)
        c = seq[idx]
        delta_val = -2 * c * z_powers_polish[idx]
        
        new_P_vals = P_vals_polish + delta_val
        new_peak = get_peak(new_P_vals)
        
        if new_peak < current_peak_polish:
            P_vals_polish = new_P_vals
            current_peak_polish = new_peak
            seq[idx] = -seq[idx]
            
            if current_peak_polish < best_peak_polish:
                best_peak_polish = current_peak_polish
                best_seq_polish = list(seq)
        else:
            delta_peak = new_peak - current_peak_polish
            if random.random() < exp(-delta_peak / T):
                P_vals_polish = new_P_vals
                current_peak_polish = new_peak
                seq[idx] = -seq[idx]
        
        T *= alpha_polish
    
    # Final local search at high resolution
    if time.time() < deadline - 2:
        def local_search_polish(c, P_vals, max_iters=5000):
            c = list(c)
            P_vals = P_vals.copy()
            cur_val = get_peak(P_vals)
            best_local_val = cur_val
            best_local_seq = list(c)
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # Try single flips
                for k in range(n):
                    old_c = c[k]
                    c[k] = -c[k]
                    delta_val = -2 * old_c * z_powers_polish[k]
                    new_P_vals = P_vals + delta_val
                    new_val = get_peak(new_P_vals)
                    if new_val < cur_val - 1e-12:
                        cur_val = new_val
                        P_vals = new_P_vals
                        if cur_val < best_local_val:
                            best_local_val = cur_val
                            best_local_seq = list(c)
                        improved = True
                        break
                    else:
                        c[k] = old_c
                
                if improved:
                    continue
                
                # Try pair flips
                pair_improved = False
                for k1 in range(n):
                    if pair_improved:
                        break
                    for k2 in range(k1+1, n):
                        c1_old = c[k1]
                        c2_old = c[k2]
                        c[k1] = -c[k1]
                        c[k2] = -c[k2]
                        delta_val = -2 * c1_old * z_powers_polish[k1] - 2 * c2_old * z_powers_polish[k2]
                        new_P_vals = P_vals + delta_val
                        new_val = get_peak(new_P_vals)
                        if new_val < cur_val - 1e-12:
                            cur_val = new_val
                            P_vals = new_P_vals
                            if cur_val < best_local_val:
                                best_local_val = cur_val
                                best_local_seq = list(c)
                            pair_improved = True
                            break
                        else:
                            c[k1] = c1_old
                            c[k2] = c2_old
                
                if not pair_improved:
                    break
            
            return best_local_seq, best_local_val
        
        final_seq, final_peak = local_search_polish(best_seq_polish, 
                                                    np.sum(np.array(best_seq_polish)[:, None] * z_powers_polish, axis=0),
                                                    max_iters=10000)
        if final_peak < best_peak_polish:
            best_seq_polish = list(final_seq)
    
    return best_seq_polish
