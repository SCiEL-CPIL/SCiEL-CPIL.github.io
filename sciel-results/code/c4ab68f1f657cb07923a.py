"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _broyden_2d(F, x0, deadline, max_iter=80, max_ls=4):
    """Broyden's method for 2D root finding. Uses rank-1 Jacobian updates
    so only 1 function eval per iteration after the first."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x, 1
    
    best_x = x.copy()
    best_norm = norm_F
    calls = 1
    
    # Initial Jacobian via finite differences (3 calls total so far)
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = F(x + e)
        calls += 1
        J[:, j] = (Fx_e - Fx) / h
    
    # Broyden state
    x_prev = x.copy()
    Fx_prev = Fx.copy()
    J_prev = J.copy()
    
    stagnation_count = 0
    max_stagnation = 3
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if calls >= 495:  # Leave margin for safety
            break
        
        # Broyden step: solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            try:
                dx = -np.linalg.solve(J + 1e-10 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                dx = -Fx * 0.01
        
        # Limit step
        step_norm = np.linalg.norm(dx)
        if step_norm > 2.0:
            dx = dx / step_norm * 2.0
        
        # Line search (minimal: try full, half, quarter)
        accepted = False
        for alpha in [1.0, 0.5, 0.25]:
            x_candidate = x + alpha * dx
            F_candidate = F(x_candidate)
            calls += 1
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate <= 1e-8:
                return x_candidate, calls
            
            if norm_candidate < best_norm:
                best_x = x_candidate.copy()
                best_norm = norm_candidate
            
            if norm_candidate < norm_F * 0.7:
                # Broyden rank-1 update
                s = x_candidate - x
                y = F_candidate - Fx
                J = J + np.outer(y - J @ s, s) / (s @ s)
                
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
                x_prev = x
                Fx_prev = Fx
                accepted = True
                break
        
        if not accepted:
            stagnation_count += 1
            if stagnation_count >= max_stagnation:
                break
        else:
            stagnation_count = 0
        
        if norm_F <= 1e-8:
            return x, calls
    
    return best_x, calls

def _newton_fd(F, x0, deadline, max_iter=60, max_ls=5):
    """Damped Newton with finite differences (baseline)."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    calls = 1
    
    if norm_F <= 1e-8:
        return x, calls
    
    best_x = x.copy()
    best_norm = norm_F
    h = 1e-5
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if calls >= 495:
            break
        
        # Compute Jacobian
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x + e)
            calls += 1
            J[:, j] = (Fx_e - Fx) / h
        
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            try:
                dx = -np.linalg.solve(J + 1e-8 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                dx = -Fx * 0.01
        
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
        
        in_polish = norm_F < 1e-4
        
        if in_polish:
            accepted = False
            for alpha in [1.0, 0.5, 0.25]:
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                calls += 1
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate <= 1e-8:
                    return x_candidate, calls
                
                if norm_candidate < best_norm:
                    best_x = x_candidate.copy()
                    best_norm = norm_candidate
                
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    accepted = True
                    break
            
            if not accepted:
                x_candidate = x - 0.1 * Fx
                F_candidate = F(x_candidate)
                calls += 1
                norm_candidate = np.linalg.norm(F_candidate)
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                else:
                    break
        else:
            alpha = 1.0
            accepted = False
            for ls_iter in range(max_ls):
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                calls += 1
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate <= 1e-8:
                    return x_candidate, calls
                
                if norm_candidate < best_norm:
                    best_x = x_candidate.copy()
                    best_norm = norm_candidate
                
                if norm_candidate < norm_F * 0.5:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                alpha = 0.1
                x_candidate = x - alpha * Fx
                F_candidate = F(x_candidate)
                calls += 1
                norm_candidate = np.linalg.norm(F_candidate)
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                else:
                    break
        
        if norm_F <= 1e-8:
            return x, calls
    
    return best_x, calls

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    x0 = np.array(x0, dtype=float)
    
    # Strategy: Try Broyden first (fewer calls per iteration).
    # If it fails or stagnates, fall back to Newton-FD.
    # If both fail, try Newton-FD from a perturbed start.
    
    # Phase 1: Broyden from x0
    x_sol, calls_1 = _broyden_2d(F, x0, deadline, max_iter=80)
    if np.linalg.norm(F(x_sol)) <= 1e-8:
        return x_sol
    if calls_1 >= 495:
        return x_sol
    
    # Phase 2: Newton-FD from x0 (fresh, doesn't share call budget conceptually
    # but we track total)
    # Actually, calls are counted globally by the evaluator, so we need to be careful.
    # Let's just use the best result if we're running low.
    remaining_budget = 495 - calls_1
    if remaining_budget < 10:
        return x_sol
    
    # Phase 2: Newton-FD from x0
    x_sol2, calls_2 = _newton_fd(F, x0, deadline, max_iter=40, max_ls=4)
    total_calls = calls_1 + calls_2
    if np.linalg.norm(F(x_sol2)) <= 1e-8:
        return x_sol2
    if total_calls >= 495:
        return x_sol2 if np.linalg.norm(F(x_sol2)) < np.linalg.norm(F(x_sol)) else x_sol
    
    # Phase 3: Newton-FD from perturbed x0
    if time.time() < deadline - 5:
        # Try a few perturbed starting points
        rng = np.random.RandomState(42)
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            if total_calls >= 490:
                break
            x_pert = x0 + rng.randn(2) * 0.5
            x_sol3, calls_3 = _newton_fd(F, x_pert, deadline, max_iter=30, max_ls=3)
            total_calls += calls_3
            if np.linalg.norm(F(x_sol3)) <= 1e-8:
                return x_sol3
            if np.linalg.norm(F(x_sol3)) < np.linalg.norm(F(x_sol)):
                x_sol = x_sol3
    
    return x_sol
# EVOLVE-BLOCK-END
