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
        # Use math.fsum for exact match with evaluator
        vals = np.log(1.0 / np.sqrt(d2_iu))
        return math.fsum(vals)
    
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

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(12345)
    
    initializations = []
    initializations.append(fibonacci_spiral(N))
    for r in range(10):
        initializations.append(random_on_sphere(N, rng))
    
    for init_idx, P_init in enumerate(initializations):
        P_cur = P_init.copy()
        E_cur = compute_energy(P_cur)
        G_cur = compute_grad(P_cur)
        
        if E_cur < best_energy:
            best_energy = E_cur
            P_best = P_cur.copy()
        
        eta = 1e-2
        momentum = 0.9
        m = np.zeros_like(P_cur)
        
        max_iters = 5000
        min_eta = 1e-10
        max_eta = 1.0
        patience = 0
        best_in_restart = E_cur
        P_best_restart = P_cur.copy()
        
        for it in range(max_iters):
            G_cur = compute_grad(P_cur)
            grad_norm = np.linalg.norm(G_cur)
            
            if grad_norm < 1e-10:
                break
            
            # Update momentum
            m = momentum * m - eta * G_cur
            
            # Line search
            improved = False
            for ls_iter in range(30):
                P_trial = riemannian_exp(P_cur, m)
                E_trial = compute_energy(P_trial)
                
                if E_trial < E_cur:
                    if E_trial < best_in_restart:
                         best_in_restart = E_trial
                         P_best_restart = P_trial.copy()
                    P_cur = P_trial
                    E_cur = E_trial
                    eta = min(eta * 1.1, max_eta)
                    improved = True
                    patience = 0
                    break
                else:
                    eta *= 0.8
            
            if not improved:
                patience += 1
                if patience > 50:
                    break
            
            if E_cur < best_energy:
                best_energy = E_cur
                P_best = P_cur.copy()
        
        if best_in_restart < best_energy:
            best_energy = best_in_restart
            P_best = P_best_restart.copy()
    
    return P_best
