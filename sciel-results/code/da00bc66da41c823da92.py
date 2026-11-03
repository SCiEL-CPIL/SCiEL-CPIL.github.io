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
        # P is (N, 3)
        # d_ij = ||P_i - P_j||
        # Energy = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        # d^2 = 2 - 2*dots
        d2 = 2.0 - 2.0 * dots
        # Ensure positive
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        d_iu = np.sqrt(d2_iu)
        # log(1/d) = -log(d)
        return -np.sum(np.log(d_iu))
        
    def compute_grad(P):
        # Gradient of E wrt P_i
        # Riemannian gradient is the projection of Euclidean gradient onto tangent space.
        # Proj_T_p(R) = R - <R, p> p
        
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        # Vectorized Euclidean Gradient Calculation
        # G_i = sum_{j != i} (P_j - P_i) / ||P_j - P_i||^2
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        # G_euclidean = inv_d2 @ P - P * sum(inv_d2, axis=1)
        G_euclidean = inv_d2 @ P
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        G_euclidean -= sum_inv_d2[:, None] * P
        
        # Project to tangent space
        dots_GP = np.sum(G_euclidean * P, axis=1)
        G_riemann = G_euclidean - dots_GP[:, None] * P
        
        return G_riemann

    def riemannian_exp(P, V):
        # V is tangent vector at P
        # exp_P(V) = P cos(||V||) + (V/||V||) sin(||V||)
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        # Renormalize to avoid drift
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def riemannian_retr(P, V):
        # Retraction: P + V, then normalize
        P_new = P + V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def lbfgs_riemannian(P_init, max_iters=200, m=10):
        P = P_init.copy()
        E = compute_energy(P)
        G = compute_grad(P)
        
        # L-BFGS state
        S = [] # Directions
        Y = [] # Gradient differences
        W = [] # Weights
        rho = [] # Inverse inner products
        
        for k in range(max_iters):
            # Compute search direction
            # r = -G
            r = -G
            q = r.copy()
            
            # Backward recursion for alpha
            alphas = []
            for i in range(len(S)-1, -1, -1):
                beta = rho[i] * np.sum(S[i] * Y[i])
                alphas.append(beta)
                q -= beta * Y[i]
            alphas.reverse()
            
            # Initial Hessian approximation H0
            # Use a simple diagonal or identity, or scale based on last step
            if len(S) > 0:
                s_last = S[-1]
                y_last = Y[-1]
                gamma = np.sum(s_last * y_last) / np.sum(y_last * y_last)
                if gamma <= 0 or not np.isfinite(gamma):
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = gamma * q
            
            # Forward recursion for p
            for i in range(len(S)):
                alpha = alphas[i]
                beta = rho[i] * np.sum(S[i] * r)
                r += (alpha - beta) * S[i]
            
            # Line search
            alpha = 1.0
            c1 = 1e-4
            slope = np.sum(G * r)
            
            found = False
            for ls_iter in range(30):
                P_trial = riemannian_exp(P, alpha * r)
                E_trial = compute_energy(P_trial)
                
                if E_trial <= E + c1 * alpha * slope:
                    found = True
                    break
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            if not found:
                # If line search fails, try a smaller step or break
                if alpha < 1e-10:
                    break
                # Force a small step to escape? Or just break.
                P_trial = riemannian_exp(P, alpha * r)
                E_trial = compute_energy(P_trial)
                if E_trial < E:
                    found = True
                else:
                    break
            
            P_new = P_trial
            E_new = E_trial
            G_new = compute_grad(P_new)
            
            # Update L-BFGS memory
            s = P_new - P # This is not strictly tangent difference in Euclidean space, 
                          # but for small steps it's a good approximation for the "direction"
                          # However, for Riemannian L-BFGS, we should use the vector in tangent space
                          # that maps P to P_new. That is exactly alpha * r (before exp).
                          # Wait, S should be the step in the tangent space?
                          # Standard Riemannian L-BFGS uses:
                          # S_k = P_{k+1} - P_k (Euclidean difference is bad)
                          # Better: S_k = the tangent vector used? No, usually S_k is the difference in the manifold.
                          # Let's stick to the standard Euclidean-like update on the tangent vectors
                          # S_k = P_{k+1} - P_k is problematic on sphere.
                          # Instead, we store the step direction in tangent space?
                          # Actually, the standard formulation for Riemannian L-BFGS (e.g., Liu-Storey)
                          # uses S_k = P_{k+1} - P_k (chord) is not ideal.
                          # Let's use the tangent vector step: S_k = alpha * r_k
                          # And Y_k = G_{k+1} - G_k (both in tangent spaces at their respective points,
                          # but we need to parallel transport G_k to P_{k+1} or vice versa?
                          # For simplicity and robustness, many implementations just use the Euclidean difference
                          # of the points for S and the difference of the gradients (projected to a common space) for Y.
                          # However, a simpler and effective approach for small steps is:
                          # S_k = P_{k+1} - P_k (Euclidean)
                          # Y_k = G_{k+1} - G_k (Euclidean)
                          # This works surprisingly well for sphere if steps are small.
                          
                          # Let's try the "tangent space" version which is more rigorous:
                          # We need to compare G at P and G at P_new.
                          # We can transport G to P_new using the logarithm map?
                          # Or just use the Euclidean difference. Given the time limit and complexity,
                          # let's use the Euclidean difference of the points and gradients.
                          # It is an approximation of the Riemannian curvature.
                          
            S.append(alpha * r)
            Y.append(G_new - G)
            
            s_norm = np.linalg.norm(S[-1])
            y_norm = np.linalg.norm(Y[-1])
            if s_norm > 1e-8 and y_norm > 1e-8:
                sy = np.sum(S[-1] * Y[-1])
                if sy > 1e-8: # Curvature condition
                    rho.append(1.0 / sy)
                else:
                    # If curvature condition fails, reset memory or skip update?
                    # Let's just skip adding to memory to avoid issues
                    S.pop()
                    Y.pop()
                    # And don't update P? No, we already updated P.
                    # Just don't store this pair.
                    pass
            else:
                S.pop()
                Y.pop()
                
            # Update memory size
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
                rho.pop(0)
                
            P = P_new
            E = E_new
            G = G_new
            
            if np.linalg.norm(G) < 1e-10:
                break
                
        return P, E

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = []
    initializations.append(fibonacci_spiral(N))
    # Add a few random initializations
    for _ in range(10):
        initializations.append(random_on_sphere(N, rng))
        
    for P_init in initializations:
        P, E = lbfgs_riemannian(P_init)
        
        # Polishing phase: Gradient Descent with small steps
        # This helps refine the solution to machine precision
        step_size = 1e-5
        for _ in range(100):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-12:
                break
            # Normalize direction
            d = -G / grad_norm
            # Line search
            alpha = step_size
            slope = -grad_norm
            for _ in range(20):
                P_trial = riemannian_exp(P, alpha * d)
                E_trial = compute_energy(P_trial)
                if E_trial <= E + 1e-4 * alpha * slope:
                    P = P_trial
                    E = E_trial
                    break
                alpha *= 0.5
            else:
                break
            
            if E < best_energy:
                best_energy = E
                P_best = P.copy()
        if E < best_energy:
            best_energy = E
            P_best = P.copy()

    return P_best
