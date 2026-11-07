import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 130.0  # safety margin before 150s kill

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    def compute_jacobian_fd(x_pt):
        """Compute Jacobian via central finite differences (4 calls)."""
        h = 1e-5
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt - e1)
            Fx3 = F_count(x_pt + e2)
            Fx4 = F_count(x_pt - e2)
            J = np.array([(Fx1 - Fx2) / (2*h), (Fx3 - Fx4) / (2*h)]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Phase 1: Homotopy continuation
    # H(x, t) = t*F(x) + (1-t)*(x - x0)
    # At t=0, root is x0. At t=1, root is a root of F.
    
    x_h = x.copy()
    t = 0.0
    
    # Compute initial Jacobian at x0 for the homotopy
    # At t=0, dH/dx = (1-t)*I + t*J_F = I
    # So we start with J_H = I
    
    J_H = np.eye(2)
    
    # We need J_F to update J_H properly
    # But at t=0, we don't need J_F yet
    # Let's compute J_F at x0 using 4 calls
    J_F = compute_jacobian_fd(x_h)
    if J_F is None:
        J_F = np.eye(2)
    
    # Now J_H = t*J_F + (1-t)*I = I at t=0
    J_H = t * J_F + (1 - t) * np.eye(2)
    
    # Homotopy stepping
    dt = 0.1
    max_homotopy_steps = 30
    
    for step in range(max_homotopy_steps):
        if not time_ok() or call_count > 350:
            break
        
        if t >= 0.999:
            break
            
        t_new = min(1.0, t + dt)
        
        # Newton iterations at fixed t_new to solve H(x, t_new) = 0
        converged = False
        max_newton = 15
        
        for _ in range(max_newton):
            if not time_ok() or call_count > 380:
                break
                
            try:
                Fx_curr = F_count(x_h)
            except:
                break
            if not np.all(np.isfinite(Fx_curr)):
                break
            
            H = t_new * Fx_curr + (1.0 - t_new) * (x_h - x0)
            norm_H_sq = float(np.dot(H, H))
            
            if norm_H_sq < 1e-24:
                converged = True
                break
            
            # J_H = t_new * J_F + (1 - t_new) * I
            J_H = t_new * J_F + (1.0 - t_new) * np.eye(2)
            
            try:
                dx = np.linalg.solve(J_H, -H)
            except:
                J_H = J_H + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_H, -H)
                except:
                    break
            
            x_try = x_h + dx
            
            try:
                Fx_try = F_count(x_try)
            except:
                break
            if not np.all(np.isfinite(Fx_try)):
                break
            
            # Update J_F using Broyden's rank-1 update
            dF = Fx_try - Fx_curr
            J_F_dx = J_F @ dx
            residual = dF - J_F_dx
            dx_norm_sq = float(np.dot(dx, dx))
            if dx_norm_sq > 1e-30:
                J_F = J_F + np.outer(residual, dx) / dx_norm_sq
            
            H_new = t_new * Fx_try + (1.0 - t_new) * (x_try - x0)
            norm_H_new_sq = float(np.dot(H_new, H_new))
            
            if norm_H_new_sq < norm_H_sq * 0.5 or norm_H_new_sq < 1e-24:
                x_h = x_try
                Fx_curr = Fx_try
                if norm_H_new_sq < 1e-24:
                    converged = True
                    break
            else:
                # Try smaller step
                x_try = x_h + 0.5 * dx
                try:
                    Fx_try = F_count(x_try)
                except:
                    break
                if not np.all(np.isfinite(Fx_try)):
                    break
                H_new = t_new * Fx_try + (1.0 - t_new) * (x_try - x0)
                norm_H_new_sq = float(np.dot(H_new, H_new))
                if norm_H_new_sq < norm_H_sq:
                    x_h = x_try
                    Fx_curr = Fx_try
                    dF = Fx_try - Fx_curr
                    J_F_dx = J_F @ (0.5 * dx)
                    residual = dF - J_F_dx
                    dx_half = 0.5 * dx
                    dx_norm_sq = float(np.dot(dx_half, dx_half))
                    if dx_norm_sq > 1e-30:
                        J_F = J_F + np.outer(residual, dx_half) / dx_norm_sq
                else:
                    break
        
        if converged:
            t = t_new
            # Increase dt if converging fast
            dt = min(0.5, dt * 1.5)
            if t >= 0.999:
                break
        else:
            # Decrease dt
            dt *= 0.5
            if dt < 0.01:
                break
    
    # Phase 2: Polish with Broyden's method on F directly
    if t >= 0.999:
        x = x_h
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))
        if norm_sq < best_norm_sq:
            best_norm_sq = norm_sq
            best_x = x.copy()
        
        # Use J_F from homotopy as initial Jacobian
        J = J_F.copy()
        
        for _ in range(20):
            if not time_ok() or call_count > 450:
                break
            if norm_sq <= tol_sq:
                return x
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                J = J + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except:
                    break
            
            x_try = x + dx
            try:
                Fx_try = F_count(x_try)
            except:
                break
            if not np.all(np.isfinite(Fx_try)):
                break
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            
            dF = Fx_try - Fx
            Jdx = J @ dx
            residual = dF - Jdx
            dx_norm_sq = float(np.dot(dx, dx))
            if dx_norm_sq > 1e-30:
                J = J + np.outer(residual, dx) / dx_norm_sq
            
            if norm_sq_try < norm_sq:
                x = x_try
                Fx = Fx_try
                norm_sq = norm_sq_try
                if norm_sq < best_norm_sq:
                    best_norm_sq = norm_sq
                    best_x = x.copy()
            else:
                break
    
    # Phase 3: If homotopy didn't reach t=1, try direct Broyden from best_x
    if best_norm_sq > tol_sq and call_count < 480 and time_ok():
        x = best_x.copy()
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))
        
        J = compute_jacobian_fd(x)
        if J is None:
            J = np.eye(2)
        
        for _ in range(15):
            if not time_ok() or call_count > 490:
                break
            if norm_sq <= tol_sq:
                return x
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                J = J + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except:
                    break
            
            x_try = x + dx
            try:
                Fx_try = F_count(x_try)
            except:
                break
            if not np.all(np.isfinite(Fx_try)):
                break
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            
            dF = Fx_try - Fx
            Jdx = J @ dx
            residual = dF - Jdx
            dx_norm_sq = float(np.dot(dx, dx))
            if dx_norm_sq > 1e-30:
                J = J + np.outer(residual, dx) / dx_norm_sq
            
            if norm_sq_try < norm_sq:
                x = x_try
                Fx = Fx_try
                norm_sq = norm_sq_try
                if norm_sq < best_norm_sq:
                    best_norm_sq = norm_sq
                    best_x = x.copy()
            else:
                break
    
    return best_x
