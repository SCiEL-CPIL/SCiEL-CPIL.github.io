import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 24
    S = 20000  # High density
    
    angles = 2 * np.pi * np.arange(S) / S
    z_powers = np.zeros((n, S), dtype=complex)
    for k in range(n):
        z_powers[k] = np.exp(1j * k * angles)
    
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_seq = list(gr_sequence(n))
    P_vals = np.sum(np.array(best_seq)[:, None] * z_powers, axis=0)
    current_peak = np.max(np.abs(P_vals))
    best_peak = current_peak
    best_seq_copy = list(best_seq)
    
    while time.time() < deadline - 5:
        # Decide start point
        if random.random() < 0.2 and len(best_seq_copy) > 0:
            # Start from perturbed best
            seq = list(best_seq_copy)
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        else:
            # Start from best
            seq = list(best_seq_copy)
            
        P_vals = np.sum(np.array(seq)[:, None] * z_powers, axis=0)
        current_peak = np.max(np.abs(P_vals))
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_flip_idx = -1
            best_flip_peak = current_peak
            
            for i in range(n):
                c = seq[i]
                delta = -2 * c * z_powers[i]
                new_P = P_vals + delta
                new_peak = np.max(np.abs(new_P))
                
                if new_peak < best_flip_peak - 1e-9:
                    best_flip_peak = new_peak
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                i = best_flip_idx
                c = seq[i]
                delta = -2 * c * z_powers[i]
                P_vals = P_vals + delta
                seq[i] = -seq[i]
                current_peak = best_flip_peak
                improved = True
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_seq_copy = list(seq)

    return best_seq_copy
