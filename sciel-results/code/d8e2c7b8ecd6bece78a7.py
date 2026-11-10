import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n = 24
    S = 5000  # Dense sampling for better peak estimation
    
    # Precompute z powers for fast delta evaluation
    angles = 2 * np.pi * np.arange(S) / S
    z_powers = np.zeros((n, S), dtype=complex)
    for k in range(n):
        z_powers[k] = np.exp(1j * k * angles)
    
    # Initial sequence: Golay-Rudin-Shapiro
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_seq = list(gr_sequence(n))
    
    # Compute initial P values
    P_vals = np.sum(np.array(best_seq)[:, None] * z_powers, axis=0)
    current_peak = np.max(np.abs(P_vals))
    best_peak = current_peak
    best_seq_copy = list(best_seq)
    
    # Helper to evaluate peak
    def get_peak(P_vals):
        return np.max(np.abs(P_vals))
    
    # Simulated Annealing with multiple restarts
    T_max = 2.0
    T_min = 1e-5
    alpha = 0.9995
    
    # Run multiple SA restarts to explore the space better
    num_restarts = 5
    restart_budget = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        # Reset to best found so far or a new random start
        if restart == 0:
            seq = list(best_seq_copy)
        else:
            # Start from best found so far with some random perturbations
            seq = list(best_seq_copy)
            for _ in range(random.randint(3, 8)):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        
        # Recompute P_vals for this sequence
        P_vals = np.sum(np.array(seq)[:, None] * z_powers, axis=0)
        current_peak = get_peak(P_vals)
        
        T = T_max
        local_best_peak = current_peak
        local_best_seq = list(seq)
        
        while time.time() < deadline - 5 and T > T_min:
            # Pick a random index to flip
            idx = random.randint(0, n - 1)
            
            # Calculate delta P_vals
            c = seq[idx]
            delta_val = -2 * c * z_powers[idx]
            
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
                if random.random() < np.exp(-delta_peak / T):
                    P_vals = new_P_vals
                    current_peak = new_peak
                    seq[idx] = -seq[idx]
            
            T *= alpha
        
        # Update global best
        if local_best_peak < best_peak:
            best_peak = local_best_peak
            best_seq_copy = list(local_best_seq)
    
    return best_seq_copy
