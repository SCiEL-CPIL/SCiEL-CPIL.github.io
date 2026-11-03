import numpy as np
import math

N = 150

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Riemannian Gradient Descent with adaptive step size
    # Energy: E = sum_{i<j} log(1/|pi-pj|) = - sum_{i<j} log|pi-pj|
    
    def energy(P):
        # Compute pairwise distances
        # |pi - pj|^2 = 2 - 2 pi.pj
        dots = P @ P.T
        np.fill_diagonal(dots, 0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        return -np.sum(np.log(np.sqrt(d2_iu)))

    def grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0)
        d2 = 2.0 - 2.0 * dots
        # diff[k, j] = P_j - P_k
        diff = P[None, :, :] - P[:, None, :]
        d2_safe = np.where(d2 > 1e-15, d2, 1e-15)
        terms = diff / d2_safe[:, :, None]
        G = np.sum(terms, axis=1)
        
        # Project to tangent space
        dots_G_P = np.sum(G * P, axis=1)
        G_proj = G - dots_G_P[:, None] * P
        return G_proj

    # Use multiple restarts with different initial conditions
    best_energy = float('inf')
    P_best = None
    
    for restart in range(5):
        if restart == 0:
            P_cur = P.copy()
        else:
            # Add random perturbation
            rng = np.random.RandomState(restart * 42)
            P_cur = P + rng.normal(0, 0.1, size=(N, 3))
            # Normalize
            P_cur /= np.linalg.norm(P_cur, axis=1)[:, None]
        
        eta = 1e-4
        best_in_restart = energy(P_cur)
        P_best_in_restart = P_cur.copy()
        
        max_iters = 3000
        for it in range(max_iters):
            G = grad(P_cur)
            
            # Update using exponential map
            v = -eta * G
            v_norm = np.linalg.norm(v, axis=1)
            
            v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
            cos_term = np.cos(v_norm_safe)
            sin_term = np.sin(v_norm_safe) / v_norm_safe
            
            P_new = cos_term[:, None] * P_cur + sin_term[:, None] * v
            P_new /= np.linalg.norm(P_new, axis=1)[:, None]
            
            E_new = energy(P_new)
            
            if E_new < best_in_restart:
                best_in_restart = E_new
                P_best_in_restart = P_new.copy()
                # Increase step size slightly on improvement
                eta = min(eta * 1.1, 1e-2)
            else:
                # Decrease step size on rejection
                eta *= 0.7
                if eta < 1e-7:
                    break
                continue
            P_cur = P_new
        
        if best_in_restart < best_energy:
            best_energy = best_in_restart
            P_best = P_best_in_restart.copy()
    
    return P_best
