"""Root finder for 3 fixed 2D systems using Broyden's method with Newton polish, multi-start, and homotopy continuation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method
    with merit function line search, Newton polish, homotopy continuation, and multi-start.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 115.0
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    def compute_jacobian(x_pt, Fx_pt):
        """Compute Jacobian via forward differences."""
        h = 1e-7
        scale = np.max(np.abs(x_pt))
        if scale > 1.0:
            h = 1e-7 * scale
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_pt + e)
            J[:, j] = (Fx_e - Fx_pt) / h
        return J
    
    # Track best solution
    best_x = x.copy()
    best_norm_sq = float('inf')
    
    # Initial evaluation
    try:
        Fx0 = F(x)
        best_norm_sq = float(np.dot(Fx0, Fx0))
    except:
        return np.zeros(2)
    
    if best_norm_sq <= tol_sq:
        return best_x
    
    # Newton polish: use fresh Jacobians for high-precision convergence
    def newton_polish(x_start, Fx_start, max_iters=20):
        """
        Newton's method with fresh Jacobian each step for high-precision convergence.
        Returns (x, Fx, converged, iters_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, 0
        
        iters = 0
        for it in range(max_iters):
            iters = it + 1
            if time_left() < 1.0:
                break
            
            J = compute_jacobian(x_cur, Fx_cur)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-14:
                break
            
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            
            x_try = x_cur + s
            Fx_try = F(x_try)
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            
            if norm_sq_try <= tol_sq:
                return x_try, Fx_try, True, iters
            
            if norm_sq_try < norm_sq_cur:
                x_cur = x_try
                Fx_cur = Fx_try
                norm_sq_cur = norm_sq_try
            else:
                # Damped Newton step
                alpha = 0.5
                for _ in range(8):
                    x_try = x_cur + alpha * s
                    Fx_try = F(x_try)
                    norm_sq_try = float(np.dot(Fx_try, Fx_try))
                    if norm_sq_try < norm_sq_cur:
                        x_cur = x_try
                        Fx_cur = Fx_try
                        norm_sq_cur = norm_sq_try
                        break
                    alpha *= 0.5
                else:
                    break
        
        return x_cur, Fx_cur, False, iters
    
    # Broyden solver from a given starting point
    def broyden_solve(x_start, Fx_start, max_iters=40):
        """
        Broyden's good method with merit function line search.
        Returns (x, Fx, converged, iters_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, 0
        
        # Compute initial Jacobian via forward differences
        J = compute_jacobian(x_cur, Fx_cur)
        
        # Check if Jacobian is well-conditioned
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-14:
            return x_cur, Fx_cur, False, 0
        
        iters = 0
        max_step = 50.0
        
        for it in range(max_iters):
            iters = it + 1
            if time_left() < 1.0:
                break
            
            # Solve J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            
            # Limit step size
            if s_norm > max_step:
                s = s * (max_step / s_norm)
                s_norm = max_step
            
            # Line search on merit function phi(x) = 0.5 * ||F(x)||^2
            alpha = 1.0
            phi_prev = 0.5 * norm_sq_cur
            slope = float(np.dot(Fx_cur, J @ s))  # directional derivative of phi
            
            # If slope is not negative, use steepest descent direction
            if slope >= 0:
                s = -J.T @ Fx_cur
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                if s_norm > max_step:
                    s = s * (max_step / s_norm)
                slope = -float(np.dot(s, J.T @ Fx_cur))
                if slope >= 0:
                    break
            
            accepted = False
            c1 = 1e-4  # Armijo parameter
            
            for _ls in range(30):
                x_try = x_cur + alpha * s
                Fx_try = F(x_try)
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                phi_try = 0.5 * norm_sq_try
                
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, iters
                
                if phi_try <= phi_prev + c1 * alpha * slope:
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                break
            
            # Update Broyden: J <- J + (Fx_new - Fx - J*s)/||s||^2 * s^T
            delta_F = Fx_try - Fx_cur - J @ s
            s_norm_sq = np.dot(s, s)
            if s_norm_sq > 1e-16:
                J = J + np.outer(delta_F, s) / s_norm_sq
            
            # Update state
            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
        
        return x_cur, Fx_cur, False, iters
    
    # Homotopy continuation: solve F_t(x) = t*F(x) + (1-t)*(x - x0) for t from 0 to 1
    def homotopy_solve(x_start, max_t_steps=8, max_broyden_iters=30):
        """
        Homotopy/continuation method.
        Solves F_t(x) = t*F(x) + (1-t)*(x - x0) for increasing t.
        At t=0, the root is x0. At t=1, we have the root of F.
        Returns (x, Fx, converged).
        """
        x_cur = x_start.copy()
        x0_ref = x_start.copy()  # Reference point for homotopy
        
        # Try to evaluate F at current point
        try:
            Fx_cur = F(x_cur)
        except:
            return x_cur, None, False
        
        # If already close to root of F, return
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True
        
        # Homotopy parameter schedule
        t_schedule = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
        
        for t in t_schedule:
            if time_left() < 2.0:
                break
            
            # Define homotopy function G_t(x) = t*F(x) + (1-t)*(x - x0_ref)
            def G_t(x_pt):
                try:
                    Fx_pt = F(x_pt)
                except:
                    return np.array([1e10, 1e10])
                return t * Fx_pt + (1 - t) * (x_pt - x0_ref)
            
            # Evaluate G_t at current point
            try:
                Gx_cur = G_t(x_cur)
            except:
                break
            
            G_norm_sq = float(np.dot(Gx_cur, Gx_cur))
            
            # If t is close to 1 and G is small, check if F is small
            if t >= 0.95:
                norm_sq_check = float(np.dot(Fx_cur, Fx_cur))
                if norm_sq_check <= tol_sq:
                    return x_cur, Fx_cur, True
            
            # Solve G_t(x) = 0 using Broyden from x_cur
            # We need to be careful: G_t depends on F, so we can't use broyden_solve directly
            # Instead, use a modified Broyden that evaluates G_t
            
            # Compute Jacobian of G_t at x_cur
            # G_t(x) = t*F(x) + (1-t)*(x - x0_ref)
            # dG_t/dx = t*J_F + (1-t)*I
            
            # Use Broyden on G_t
            x_hom = x_cur.copy()
            try:
                Gx_hom = G_t(x_hom)
            except:
                break
            
            G_norm_sq_hom = float(np.dot(Gx_hom, Gx_hom))
            if G_norm_sq_hom <= tol_sq * tol_sq * 0.01:
                # G_t is small; check if F is small
                if t >= 0.9:
                    try:
                        Fx_hom = F(x_hom)
                    except:
                        break
                    norm_sq_hom = float(np.dot(Fx_hom, Fx_hom))
                    if norm_sq_hom <= tol_sq:
                        return x_hom, Fx_hom, True
                continue
            
            # Compute Jacobian of G_t via finite differences
            h = 1e-7
            scale = np.max(np.abs(x_hom))
            if scale > 1.0:
                h = 1e-7 * scale
            J_hom = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Gx_e = G_t(x_hom + e)
                except:
                    Gx_e = Gx_hom
                J_hom[:, j] = (Gx_e - Gx_hom) / h
            
            det_hom = J_hom[0, 0] * J_hom[1, 1] - J_hom[0, 1] * J_hom[1, 0]
            if abs(det_hom) < 1e-14:
                break
            
            # Broyden iterations on G_t
            for _it in range(max_broyden_iters):
                if time_left() < 1.0:
                    break
                
                try:
                    s = np.linalg.solve(J_hom, -Gx_hom)
                except np.linalg.LinAlgError:
                    break
                
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                
                # Limit step
                if s_norm > 10.0:
                    s = s * (10.0 / s_norm)
                
                # Line search
                alpha = 1.0
                phi_prev = 0.5 * G_norm_sq_hom
                accepted = False
                
                for _ls in range(20):
                    x_try = x_hom + alpha * s
                    try:
                        Gx_try = G_t(x_try)
                    except:
                        break
                    norm_sq_try = float(np.dot(Gx_try, Gx_try))
                    phi_try = 0.5 * norm_sq_try
                    
                    if norm_sq_try <= tol_sq * tol_sq * 0.01:
                        # G_t is very small
                        if t >= 0.9:
                            try:
                                Fx_try = F(x_try)
                            except:
                                break
                            norm_sq_F = float(np.dot(Fx_try, Fx_try))
                            if norm_sq_F <= tol_sq:
                                return x_try, Fx_try, True
                        x_cur = x_try
                        break
                    
                    if phi_try <= phi_prev + 1e-4 * alpha * (-float(np.dot(Gx_hom, J_hom @ s))):
                        accepted = True
                        break
                    
                    alpha *= 0.5
                    if alpha < 1e-10:
                        break
                
                if not accepted and G_norm_sq_hom > 1e-10:
                    # Accept anyway if we're making progress
                    x_try = x_hom + alpha * s
                    try:
                        Gx_try = G_t(x_try)
                    except:
                        break
                    norm_sq_try = float(np.dot(Gx_try, Gx_try))
                    if norm_sq_try < G_norm_sq_hom:
                        x_cur = x_try
                        break
                    else:
                        break
                
                # Update Broyden
                delta_G = Gx_try - Gx_hom - J_hom @ s
                s_norm_sq = np.dot(s, s)
                if s_norm_sq > 1e-16:
                    J_hom = J_hom + np.outer(delta_G, s) / s_norm_sq
                
                x_hom = x_try
                Gx_hom = Gx_try
                G_norm_sq_hom = norm_sq_try
            
            # After solving G_t, update x_cur
            x_cur = x_hom
            try:
                Fx_cur = F(x_cur)
            except:
                break
            
            norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
            if norm_sq_cur <= tol_sq:
                return x_cur, Fx_cur, True
            
            # Update best
            if norm_sq_cur < best_norm_sq:
                best_norm_sq = norm_sq_cur
                best_x = x_cur.copy()
        
        return x_cur, Fx_cur, False
    
    # Multi-start strategy with Broyden + Newton polish + Homotopy
    max_restarts = 20
    
    # First, try from x0 with Broyden
    x_sol, Fx_sol, converged, _ = broyden_solve(x, Fx0, max_iters=40)
    if converged:
        return x_sol
    
    # If Broyden got close but not exact, try Newton polish
    norm_sq_sol = float(np.dot(Fx_sol, Fx_sol))
    if norm_sq_sol < 1e-4:
        x_pol, Fx_pol, conv_pol, _ = newton_polish(x_sol, Fx_sol, max_iters=20)
        if conv_pol:
            return x_pol
        norm_sq_pol = float(np.dot(Fx_pol, Fx_pol))
        if norm_sq_pol < norm_sq_sol:
            x_sol, Fx_sol, norm_sq_sol = x_pol, Fx_pol, norm_sq_pol
    
    if norm_sq_sol < best_norm_sq:
        best_norm_sq = norm_sq_sol
        best_x = x_sol.copy()
    
    # Try homotopy from x0
    if time_left() > 5.0:
        x_hom, Fx_hom, conv_hom = homotopy_solve(x, max_t_steps=8, max_broyden_iters=25)
        if conv_hom:
            return x_hom
        if Fx_hom is not None:
            norm_sq_hom = float(np.dot(Fx_hom, Fx_hom))
            if norm_sq_hom < best_norm_sq:
                best_norm_sq = norm_sq_hom
                best_x = x_hom.copy()
    
    # If not converged, try from multiple starting points
    np.random.seed(42)
    
    for restart in range(max_restarts):
        if time_left() < 3.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x
        
        # Generate starting point
        if restart % 4 == 0:
            # Random in box around x0
            x_start = x + np.random.uniform(-10, 10, size=2)
        elif restart % 4 == 1:
            # Ring around best_x with growing radius
            radius = 1.0 + 2.0 * (restart // 4)
            angle = np.random.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart % 4 == 2:
            # Small perturbation around best_x
            x_start = best_x + np.random.randn(2) * 0.5
        else:
            # Larger random perturbation
            x_start = x + np.random.randn(2) * 5.0
        
        if not np.all(np.isfinite(x_start)):
            continue
        
        try:
            Fx_start = F(x_start)
        except:
            continue
        
        norm_sq_start = float(np.dot(Fx_start, Fx_start))
        if norm_sq_start < best_norm_sq:
            best_norm_sq = norm_sq_start
            best_x = x_start.copy()
        
        if norm_sq_start <= tol_sq:
            return x_start
        
        # Broyden from this start
        x_sol, Fx_sol, converged, _ = broyden_solve(x_start, Fx_start, max_iters=35)
        if converged:
            return x_sol
        
        # Newton polish if close
        norm_sq_sol = float(np.dot(Fx_sol, Fx_sol))
        if norm_sq_sol < 1e-4:
            x_pol, Fx_pol, conv_pol, _ = newton_polish(x_sol, Fx_sol, max_iters=20)
            if conv_pol:
                return x_pol
            norm_sq_pol = float(np.dot(Fx_pol, Fx_pol))
            if norm_sq_pol < norm_sq_sol:
                x_sol, Fx_sol, norm_sq_sol = x_pol, Fx_pol, norm_sq_pol
        
        if norm_sq_sol < best_norm_sq:
            best_norm_sq = norm_sq_sol
            best_x = x_sol.copy()
        
        # Try homotopy from this start if we have time
        if time_left() > 8.0 and restart < 8:
            x_hom, Fx_hom, conv_hom = homotopy_solve(x_start, max_t_steps=6, max_broyden_iters=20)
            if conv_hom:
                return x_hom
            if Fx_hom is not None:
                norm_sq_hom = float(np.dot(Fx_hom, Fx_hom))
                if norm_sq_hom < best_norm_sq:
                    best_norm_sq = norm_sq_hom
                    best_x = x_hom.copy()
    
    # Final Newton polish on best_x if close
    if best_norm_sq < 1e-4:
        try:
            Fx_best = F(best_x)
            x_pol, Fx_pol, conv_pol, _ = newton_polish(best_x, Fx_best, max_iters=20)
            if conv_pol:
                return x_pol
            norm_sq_pol = float(np.dot(Fx_pol, Fx_pol))
            if norm_sq_pol < best_norm_sq:
                best_x = x_pol.copy()
        except:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
