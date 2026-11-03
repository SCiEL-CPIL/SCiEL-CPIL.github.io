import numpy as np
import math

N = 150

def construct():
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_on_sphere(n, rng):
        v = rng.normal(0, 1, size=(n, 3))
        norms = np.linalg.norm(v, axis=1)
        v /= norms[:, None]
        return v
    
    def compute_energy(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        return -np.sum(np.log(np.sqrt(d2_iu)))
    
    def compute_grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2_safe = np.where(d2 > 1e-15, d2, 1e-15)
        
        diff = P[None, :, :] - P[:, None, :]
        terms = diff / d2_safe[:, :, None]
        G = np.sum(terms, axis=1)
        
        dots_GP = np.einsum('ij,ij->i', G, P)
        G_proj = G - dots_GP[:, None] * P
        return G_proj

    def riemannian_exp(P, V):
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def parallel_transport(X, Y, V):
        """
        Parallel transport vector V from X to Y on the unit sphere.
        X, Y: (N, 3) arrays of unit vectors
        V: (N, 3) array of tangent vectors at X (V[i] . X[i] = 0)
        Returns: (N, 3) array of tangent vectors at Y
        """
        # Compute angle theta between X and Y
        dots = np.sum(X * Y, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        theta = np.arccos(dots)
        
        # Handle cases where theta is very small or very close to pi
        small_theta = theta < 1e-10
        large_theta = theta > np.pi - 1e-10
        
        # For small theta, parallel transport is approximately identity
        # For large theta, we need careful handling
        
        # Standard parallel transport formula on sphere:
        # T_{X->Y}(V) = V + (1 - cos(theta)/theta) (X x Y) x V / |X x Y|^2 * |X x Y|^2 ... 
        # Actually, the correct formula uses the rotation that maps X to Y.
        
        # Let's use the explicit rotation matrix approach:
        # The rotation that maps X to Y can be constructed using cross product.
        # For each i, we want R_i such that R_i @ X_i = Y_i
        
        # Compute cross product u = X x Y
        u = np.cross(X, Y)
        u_norm = np.linalg.norm(u, axis=1)
        
        # For points where X and Y are nearly parallel (u_norm small), use identity
        # For points where X and Y are nearly antipodal, handle separately
        
        result = V.copy()
        
        # Normal case: X and Y are not parallel or antipodal
        mask = (u_norm > 1e-10) & (u_norm < np.pi - 1e-10)
        
        if np.any(mask):
            Xm = X[mask]
            Ym = Y[mask]
            Vm = V[mask]
            um = u[mask]
            um_norm = u_norm[mask]
            
            # sin_theta and cos_theta
            sin_theta = np.sin(theta[mask])
            cos_theta = np.cos(theta[mask])
            
            # Rotation matrix using Rodrigues' formula
            # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
            # where K is the cross-product matrix of u/|u|
            
            # K @ v = u_hat x v
            # K^2 @ v = u_hat x (u_hat x v) = u_hat (u_hat . v) - v (u_hat . u_hat) = u_hat (u_hat . v) - v
            
            u_hat = um / um_norm[:, None]
            
            # Compute K @ Vm: u_hat x Vm
            K_V = np.cross(u_hat, Vm)
            
            # Compute K^2 @ Vm: u_hat (u_hat . Vm) - Vm
            uhat_dot_V = np.sum(u_hat * Vm, axis=1)
            K2_V = u_hat * uhat_dot_V[:, None] - Vm
            
            # R @ Vm = Vm + sin(theta) * K_V + (1 - cos(theta)) * K2_V
            result[mask] = Vm + sin_theta[:, None] * K_V + (1.0 - cos_theta)[:, None] * K2_V
        
        # Handle small theta: parallel transport is approximately V projected to tangent at Y
        if np.any(small_theta):
            # Project V to tangent space at Y
            V_proj = V[small_theta]
            Ys = Y[small_theta]
            dot_VY = np.sum(V_proj * Ys, axis=1)
            result[small_theta] = V_proj - dot_VY[:, None] * Ys
        
        # Handle large theta (near antipodal): use a simple projection
        if np.any(large_theta):
            Vl = V[large_theta]
            Yl = Y[large_theta]
            dot_VY = np.sum(Vl * Yl, axis=1)
            result[large_theta] = Vl - dot_VY[:, None] * Yl
        
        # Ensure result is tangent to Y
        dot_result_Y = np.sum(result * Y, axis=1)
        result = result - dot_result_Y[:, None] * Y
        
        return result

    def line_search(P_cur, E_cur, G_cur, direction, eta0):
        """
        Backtracking line search with Armijo condition.
        direction: tangent vector at P_cur
        Returns: (P_new, E_new, eta_used)
        """
        eta = eta0
        c1 = 1e-4  # Armijo parameter
        rho = 0.5  # Backtracking factor
        
        for ls_iter in range(50):
            P_trial = riemannian_exp(P_cur, direction * eta)
            E_trial = compute_energy(P_trial)
            
            # Armijo condition: E_trial <= E_cur + c1 * eta * <G_cur, direction>
            inner = np.sum(G_cur * direction)
            if E_trial <= E_cur + c1 * eta * inner:
                return P_trial, E_trial, eta
            
            eta *= rho
        
        # If line search fails, return the last trial
        return P_trial, E_trial, eta

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initializations
    initializations = []
    initializations.append(fibonacci_spiral(N))
    for r in range(8):
        initializations.append(random_on_sphere(N, rng))
    
    for init_idx, P_init in enumerate(initializations):
        P_cur = P_init.copy()
        E_cur = compute_energy(P_cur)
        G_cur = compute_grad(P_cur)
        
        if E_cur < best_energy:
            best_energy = E_cur
            P_best = P_cur.copy()
        
        # Riemannian Conjugate Gradient
        max_iters = 3000
        eta0 = 0.01
        beta_method = 'PR'  # Polak-Ribiere
        
        # CG state
        G_prev = G_cur.copy()
        P_prev = P_cur.copy()
        D_cur = -G_cur.copy()  # Initial search direction = negative gradient
        
        best_in_restart = E_cur
        P_best_restart = P_cur.copy()
        patience = 0
        max_patience = 100
        
        for it in range(max_iters):
            # Compute gradient
            G_cur = compute_grad(P_cur)
            grad_norm = np.linalg.norm(G_cur)
            
            if grad_norm < 1e-9:
                break
            
            # Line search
            P_new, E_new, eta_used = line_search(P_cur, E_cur, G_cur, D_cur, eta0)
            
            if E_new < best_in_restart:
                best_in_restart = E_new
                P_best_restart = P_new.copy()
            
            # Update
            P_prev = P_cur
            G_prev = G_cur
            P_cur = P_new
            E_cur = E_new
            
            # Compute beta using Polak-Ribiere formula
            G_new = compute_grad(P_cur)
            
            # Transport G_prev from P_prev to P_cur
            G_prev_trans = parallel_transport(P_prev, P_cur, G_prev)
            
            # PR beta: beta = <G_new, G_new - G_prev_trans> / <G_prev_trans, G_prev_trans>
            num = np.sum(G_new * (G_new - G_prev_trans))
            den = np.sum(G_prev_trans * G_prev_trans)
            
            if den < 1e-15:
                beta = 0.0
            else:
                beta = max(0.0, num / den)
            
            # New search direction
            D_cur = -G_new + beta * parallel_transport(P_prev, P_cur, D_cur)
            
            # Ensure D_cur is tangent to P_cur
            dot_DP = np.sum(D_cur * P_cur, axis=1)
            D_cur = D_cur - dot_DP[:, None] * P_cur
            
            # Check for convergence or restart
            if E_cur < best_energy:
                best_energy = E_cur
                P_best = P_cur.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                # Restart CG
                D_cur = -G_new.copy()
                patience = 0
            
            # Early termination if progress is negligible
            if it > 100 and abs(E_cur - best_in_restart) < 1e-10:
                break
        
        if best_in_restart < best_energy:
            best_energy = best_in_restart
            P_best = P_best_restart.copy()
    
    return P_best
