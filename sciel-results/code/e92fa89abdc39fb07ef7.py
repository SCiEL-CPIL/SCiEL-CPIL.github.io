import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initial Coefficients (GRS truncated)
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=np.int8)
    
    # Setup Evaluation Points for Local Search (Medium Resolution)
    M_low = 2048
    theta_low = np.linspace(0, 2*np.pi, M_low, endpoint=False)
    k = np.arange(24)
    Z_low = np.exp(1j * np.outer(theta_low, k))
    
    # Helper to compute sup norm on low res grid
    def get_low_res_sup_norm(c_vec):
        P = Z_low @ c_vec.astype(np.float64)
        return np.max(np.abs(P))
    
    # Helper to get refined sup norm (High Resolution + Golden Section)
    def get_refined_sup_norm(c_vec):
        M_high = 8192
        theta_high = np.linspace(0, 2*np.pi, M_high, endpoint=False)
        Z_high = np.exp(1j * np.outer(theta_high, k))
        P_high = Z_high @ c_vec.astype(np.float64)
        abs_P_high = np.abs(P_high)
        
        # Find top 5 peaks
        idx = np.argpartition(abs_P_high, -5)[-5:]
        max_refined = 0.0
        for i in idx:
            delta = 2*np.pi / M_high
            a = theta_high[i] - delta
            b = theta_high[i] + delta
            gr = (np.sqrt(5) - 1) / 2
            c_g = b - gr * (b - a)
            d_g = a + gr * (b - a)
            
            def f(t):
                z = np.exp(1j * t)
                val = np.sum(c_vec * np.power(z, k))
                return np.abs(val)
                
            f_c = f(c_g)
            f_d = f(d_g)
            
            for _ in range(50):
                if f_c > f_d:
                    b = d_g
                    d_g = c_g
                    f_d = f_c
                    c_g = b - gr * (b - a)
                    f_c = f(c_g)
                else:
                    a = c_g
                    c_g = d_g
                    f_c = f_d
                    d_g = a + gr * (b - a)
                    f_d = f(d_g)
            
            local_max = max(f_c, f_d)
            if local_max > max_refined:
                max_refined = local_max
                
        return max_refined

    best_c = c.copy()
    best_score = get_refined_sup_norm(best_c)
    best_low_res_score = get_low_res_sup_norm(best_c)
    current_c = c.copy()
    
    # Local Search Function (Greedy 1/2 flips)
    def local_search(c_vec, P_vec, max_val):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 1-Flips
            for j in range(24):
                delta_P = -2 * c_vec[j] * Z_low[:, j]
                new_P = P_vec + delta_P
                new_max = np.max(np.abs(new_P))
                if new_max < max_val - 1e-12:
                    c_vec[j] = -c_vec[j]
                    P_vec = new_P
                    max_val = new_max
                    improved = True
                    break
            if improved:
                continue
                
            # 2-Flips
            for j1 in range(23):
                for j2 in range(j1 + 1, 24):
                    delta_P = -2 * (c_vec[j1] * Z_low[:, j1] + c_vec[j2] * Z_low[:, j2])
                    new_P = P_vec + delta_P
                    new_max = np.max(np.abs(new_P))
                    if new_max < max_val - 1e-12:
                        c_vec[j1] = -c_vec[j1]
                        c_vec[j2] = -c_vec[j2]
                        P_vec = new_P
                        max_val = new_max
                        improved = True
                        break
                if improved:
                    break
        return c_vec, P_vec, max_val

    # Iterated Local Search Loop
    while time.time() < deadline - 10:
        # Local search from current state
        P_low = Z_low @ current_c.astype(np.float64)
        max_low = np.max(np.abs(P_low))
        current_c, P_low, max_low = local_search(current_c, P_low, max_low)
        
        # Check if we found a new low-res minimum
        if max_low < best_low_res_score - 1e-12:
            # Refine this candidate
            current_score = get_refined_sup_norm(current_c)
            if current_score < best_score:
                best_score = current_score
                best_c = current_c.copy()
                best_low_res_score = max_low
            else:
                # Even if refined score is worse, update low-res best if it's a new low-res min?
                # No, we want to track the best refined. But low-res is a proxy.
                # If low-res is better but refined is worse, it's a "false" improvement due to resolution.
                # We should still update best_low_res_score to avoid re-refining the same low-res min?
                # Or keep best_low_res_score as the lowest low-res score seen, regardless of refined?
                # Let's update best_low_res_score to max_low to avoid re-checking.
                best_low_res_score = max_low
        
        # Perturbation: Flip 3-5 random coefficients
        n_flips = np.random.randint(3, 6)
        flip_idx = np.random.choice(24, n_flips, replace=False)
        current_c[flip_idx] = -current_c[flip_idx]
        
        # Occasionally restart from scratch if stuck?
        # ILS perturbation should handle it.
        
    # Final refinement of best_c
    # Ensure best_c is the one with the lowest refined score
    # We already track this.
    
    return [int(x) for x in best_c]
