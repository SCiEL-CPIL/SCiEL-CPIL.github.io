"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Adaptive step size for finite differences
    h = 1e-5
    max_iter = 40  # Reduced to save calls for early abort
    max_ls = 4     # Reduced line search steps
    
    # Threshold to switch to fast polish mode
    polish_threshold = 1e-3
    
    # Tolerance for convergence check
    tol = 1e-8
    
    total_calls = 1  # initial F(x0)
    
    def update_best(x_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand
            
    def time_check():
        return time.time() > deadline

    # Main Newton loop
    for iteration in range(max_iter):
        if time_check() or total_calls > 150: # Hard stop if too many calls
            break
        
        in_polish = norm_F < polish_threshold
        
        # Compute Jacobian using forward differences
        h_jac = h
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h_jac
            Fx_e = F(x + e)
            J[:, j] = (Fx_e - Fx) / h_jac
            total_calls += 1
        
        # Try to solve J dx = -Fx
        try:
            # Check condition number roughly? Or just solve.
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                dx = -np.linalg.solve(J + 1e-6 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                dx = -Fx * 0.01
        
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
        if step_norm < 1e-15:
            break
        
        # Line Search
        accepted = False
        alpha = 1.0
        for ls_iter in range(max_ls):
            x_candidate = x + alpha * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            total_calls += 1
            
            if norm_candidate <= tol:
                return x_candidate
            
            update_best(x_candidate, norm_candidate)
            
            # Accept if we made significant progress
            # In polish mode, accept any decrease?
            if in_polish:
                if norm_candidate < norm_F:
                    accepted = True
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
            else:
                if norm_candidate < norm_F * 0.5:
                    accepted = True
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
                elif norm_candidate < norm_F and ls_iter == max_ls - 1:
                    # Accept small decrease on last try to avoid stalling
                    accepted = True
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
            alpha *= 0.5
            
        if not accepted:
            # Stuck, try gradient step
            x_candidate = x - 0.1 * Fx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            total_calls += 1
            if norm_candidate < norm_F:
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
                update_best(x, norm_F)
            else:
                break
        
        if norm_F <= tol:
            return x
        
        # Adaptive h
        if norm_F > 1.0:
            h = 1e-4
        elif norm_F < 1e-3:
            h = 1e-6

    # If main loop didn't converge, try a few random restarts
    if best_norm > tol and total_calls < 200 and not time_check():
        # Random restarts are more likely to find a different basin than deterministic grid
        for _ in range(10):
            if time_check() or total_calls > 250:
                break
            
            # Generate random point in a ball around x0
            r = np.random.uniform(0.1, 5.0)
            theta = np.random.uniform(0, 2 * np.pi)
            x_start = x0 + r * np.array([np.cos(theta), np.sin(theta)])
            
            try:
                F_start = F(x_start)
                total_calls += 1
            except Exception:
                continue
            
            norm_start = np.linalg.norm(F_start)
            if norm_start <= tol:
                return x_start
            update_best(x_start, norm_start)
            
            # Quick Newton from this start
            x_curr = x_start.copy()
            Fx_curr = F_start.copy()
            norm_curr = norm_start
            h_local = 1e-5
            
            for _ in range(10):
                if time_check() or total_calls > 280:
                    break
                
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h_local
                    Fx_e = F(x_curr + e)
                    J[:, j] = (Fx_e - Fx_curr) / h_local
                    total_calls += 1
                
                try:
                    dx = -np.linalg.solve(J, Fx_curr)
                except np.linalg.LinAlgError:
                    try:
                        dx = -np.linalg.solve(J + 1e-6 * np.eye(2), Fx_curr)
                    except np.linalg.LinAlgError:
                        dx = -Fx_curr * 0.01
                
                step_norm = np.linalg.norm(dx)
                if step_norm > 1.0:
                    dx = dx / step_norm
                if step_norm < 1e-15:
                    break
                
                # Simple line search for restarts
                alpha = 1.0
                accepted = False
                for _ in range(3):
                    x_candidate = x_curr + alpha * dx
                    F_candidate = F(x_candidate)
                    norm_candidate = np.linalg.norm(F_candidate)
                    total_calls += 1
                    
                    if norm_candidate <= tol:
                        return x_candidate
                    update_best(x_candidate, norm_candidate)
                    
                    if norm_candidate < norm_curr * 0.5:
                        accepted = True
                        x_curr = x_candidate
                        Fx_curr = F_candidate
                        norm_curr = norm_candidate
                        break
                    alpha *= 0.5
                
                if not accepted:
                    break
                
                if norm_curr <= tol:
                    return x_curr
                
                if norm_curr > 1.0:
                    h_local = 1e-4
                elif norm_curr < 1e-3:
                    h_local = 1e-6

    return best_x
# EVOLVE-BLOCK-END
