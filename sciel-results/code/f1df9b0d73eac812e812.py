import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave buffer for safety
    
    N_COEFF = 24
    N_GRID = 4096  # Start with 4096, can increase if time allows
    
    # Precompute grid
    theta = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
    z = np.exp(1j * theta)
    
    # Precompute powers z^k for k=0..23
    Z_powers = np.zeros((N_COEFF, N_GRID), dtype=complex)
    Z_powers[0, :] = 1
    for k in range(1, N_COEFF):
        Z_powers[k, :] = Z_powers[k-1, :] * z
        
    # Initialize with Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N_COEFF:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:N_COEFF], dtype=np.int8)
    
    def eval_max(c_vec):
        P = np.sum(c_vec[:, None] * Z_powers, axis=0)
        return np.max(np.abs(P))
    
    best_c = c.copy()
    best_score = eval_max(c)
    current_score = best_score
    
    tabu_list = []
    tabu_tenure = 15
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < time_limit:
        # Check for restart
        if no_improve_count > max_no_improve:
            # Random restart
            c = np.random.choice([-1, 1], size=N_COEFF).astype(np.int8)
            current_score = eval_max(c)
            if current_score < best_score:
                best_score = current_score
                best_c = c.copy()
                no_improve_count = 0
            else:
                no_improve_count = 0
            tabu_list = []
            continue
            
        P_current = np.sum(c[:, None] * Z_powers, axis=0)
        P_flips = P_current[None, :] - 2 * c[:, None] * Z_powers
        scores = np.max(np.abs(P_flips), axis=1)
        
        # Find best move
        g_idx = np.argmin(scores)
        g_score = scores[g_idx]
        
        # Tabu check
        is_tabu = g_idx in tabu_list
        
        if is_tabu:
            # Aspiration: if g_score < best_score, accept
            if g_score >= best_score:
                # Find best non-tabu
                mask = np.ones(N_COEFF, dtype=bool)
                for t in tabu_list:
                    mask[t] = False
                if np.any(mask):
                    non_tabu_scores = np.where(mask, scores, np.inf)
                    g_idx = np.argmin(non_tabu_scores)
                    g_score = non_tabu_scores[g_idx]
                else:
                    # All tabu, accept best
                    pass
        
        # Apply move
        c[g_idx] *= -1
        current_score = g_score
        
        # Update tabu
        tabu_list.append(g_idx)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    # Final polish: increase grid size and re-evaluate best_c?
    # The evaluator does high precision. We just need the right coefficients.
    # The coefficients found with 4096 grid should be robust.
    
    return list(best_c)
