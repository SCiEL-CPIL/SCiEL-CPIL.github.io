import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds a root x in R^2 such that ||F(x)|| <= 1e-8.
    Uses trust-region dogleg method on nonlinear least-squares formulation.
    """
    start_time = time.time()
    deadline = start_time + 100  # 50s safety margin for 150s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_F
    total_calls = 1
    
    # Constants
    max_iter = 50
    fd_h = 1e-6
    delta_min = 1e-8
    delta_max = 1e6
    gamma = 0.1  # Armijo-like parameter for trust region
    
    def update_best(x_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand

    # Compute Jacobian using central finite differences for better accuracy
    def compute_fd_jacobian(x_curr, F_curr):
        J_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = fd_h
            Fx_plus = F(x_curr + e)
            Fx_minus = F(x_curr - e)
            J_fd[:, j] = (Fx_plus - Fx_minus) / (2 * fd_h)
        return J_fd

    J = compute_fd_jacobian(x, Fx)
    total_calls += 4
    
    # Trust region radius
    delta = 1.0
    
    # Dogleg trust-region loop
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check for convergence
        if norm_F <= 1e-8:
            return x
        
        # Compute Gauss-Newton step: J^T J s = -J^T Fx
        JtJ = J.T @ J
        JtF = J.T @ Fx
        
        # Check if JtJ is well-conditioned
        try:
            eigvals = np.linalg.eigvalsh(JtJ)
            if np.min(eigvals) < 1e-14:
                # Add small regularization
                JtJ += 1e-10 * np.eye(2)
                eigvals = np.linalg.eigvalsh(JtJ)
                if np.min(eigvals) < 1e-14:
                    # Fall back to steepest descent
                    s_gn = -JtF / (np.linalg.norm(JtF) + 1e-14)
                    s_sd = s_gn
                else:
                    s_gn = -np.linalg.solve(JtJ, JtF)
                    s_sd = -JtF / (np.linalg.norm(JtF) + 1e-14)
            else:
                s_gn = -np.linalg.solve(JtJ, JtF)
                s_sd = -JtF / (np.linalg.norm(JtF) + 1e-14)
        except np.linalg.LinAlgError:
            # Fall back to steepest descent
            s_gn = -JtF / (np.linalg.norm(JtF) + 1e-14)
            s_sd = s_gn
        
        # Compute the dogleg path: from s_sd to s_gn
        # The dogleg step is either:
        # 1. Along steepest descent until it reaches the trust region boundary
        # 2. Or along the line from s_sd to s_gn if that intersects the trust region
        
        norm_sd = np.linalg.norm(s_sd)
        norm_gn = np.linalg.norm(s_gn)
        
        if norm_gn <= delta:
            # Gauss-Newton step fits in trust region
            s = s_gn
        elif norm_sd >= delta:
            # Steepest descent step is outside trust region, scale it
            s = s_sd * (delta / norm_sd)
        else:
            # Find intersection of line from s_sd to s_gn with trust region
            # Parametrize: s(t) = s_sd + t * (s_gn - s_sd), t in [0,1]
            # ||s(t)||^2 = delta^2
            # Solve quadratic
            d = s_gn - s_sd
            a = d @ d
            b = 2 * (s_sd @ d)
            c = s_sd @ s_sd - delta * delta
            
            disc = b * b - 4 * a * c
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / (2 * a)
                t2 = (-b + sqrt_disc) / (2 * a)
                
                # Pick the root in [0,1]
                if 0 <= t1 <= 1:
                    t = t1
                elif 0 <= t2 <= 1:
                    t = t2
                else:
                    # No valid intersection, use steepest descent scaled
                    t = 0
                    s = s_sd * (delta / norm_sd) if norm_sd > 0 else np.zeros(2)
                    # Update trust region and continue
                    if norm_F > best_norm * 0.99:
                        delta *= 0.5
                        if delta < delta_min:
                            delta = delta_min
                    continue
            else:
                # No real intersection, use steepest descent scaled
                s = s_sd * (delta / norm_sd) if norm_sd > 0 else np.zeros(2)
                if norm_F > best_norm * 0.99:
                    delta *= 0.5
                    if delta < delta_min:
                        delta = delta_min
                continue
            
            s = s_sd + t * d
        
        # Evaluate at trial point
        x_trial = x + s
        F_trial = F(x_trial)
        norm_trial = np.linalg.norm(F_trial)
        total_calls += 1
        
        if norm_trial <= 1e-8:
            return x_trial
        
        update_best(x_trial, norm_trial)
        
        # Compute actual reduction vs predicted reduction
        # Predicted: 0.5 * ||F||^2 reduction from linear model
        # Actual: 0.5 * (norm_F^2 - norm_trial^2)
        actual_reduction = 0.5 * (norm_F ** 2 - norm_trial ** 2)
        predicted_reduction = 0.5 * (norm_F ** 2 - np.linalg.norm(Fx + J @ s) ** 2)
        
        if predicted_reduction > 1e-14:
            rho = actual_reduction / predicted_reduction
        else:
            rho = 1.0 if actual_reduction > 0 else 0.0
        
        # Update trust region based on rho
        if rho > 0.75:
            # Good agreement, expand trust region
            delta = min(delta * 2.0, delta_max)
            x = x_trial
            Fx = F_trial
            norm_F = norm_trial
        elif rho > 0.25:
            # Acceptable, keep trust region
            x = x_trial
            Fx = F_trial
            norm_F = norm_trial
        else:
            # Poor agreement, shrink trust region
            delta *= 0.5
            if delta < delta_min:
                delta = delta_min
                # If we're stuck at minimum trust region, try to escape
                # by taking a small random step
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    rand_step = 1e-4 * np.random.randn(2)
                    x_rand = x + rand_step
                    F_rand = F(x_rand)
                    norm_rand = np.linalg.norm(F_rand)
                    total_calls += 1
                    if norm_rand < norm_F:
                        x = x_rand
                        Fx = F_rand
                        norm_F = norm_rand
                        update_best(x, norm_F)
                        delta = max(delta * 2.0, 1e-6)
                        break
                if delta < delta_min:
                    break
        
        # Recompute Jacobian periodically or if trust region changed significantly
        if it % 3 == 0 or delta > 1.0:
            J = compute_fd_jacobian(x, Fx)
            total_calls += 4
    
    # Final check
    if norm_F <= 1e-8:
        return x
    return best_x
