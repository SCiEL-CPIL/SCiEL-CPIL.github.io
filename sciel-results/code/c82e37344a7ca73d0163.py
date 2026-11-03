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

    def optimize(P_init, max_iters=2000, tol=1e-12):
        P = P_init.copy()
        E = compute_energy(P)
        G = compute_grad(P)
        
        best_E = E
        best_P = P.copy()
        
        eta = 0.1
        c1 = 1e-4
        max_ls = 20
        
        for it in range(max_iters):
            grad_norm = np.linalg.norm(G)
            if grad_norm < tol:
                break
            
            # Backtracking Line Search
            step = eta
            improved = False
            for ls_iter in range(max_ls):
                V = -step * G
                P_trial = riemannian_exp(P, V)
                E_trial = compute_energy(P_trial)
                
                # Armijo condition
                if E_trial < E - c1 * step * grad_norm**2:
                    P = P_trial
                    E = E_trial
                    G = compute_grad(P)
                    
                    if E < best_E:
                        best_E = E
                        best_P = P.copy()
                    
                    # Increase step size if we didn't backtrack much
                    if ls_iter == 0:
                        eta = min(eta * 1.5, 1.0)
                    else:
                        eta = step
                    improved = True
                    break
                else:
                    step *= 0.5
            
            if not improved:
                # If line search fails, shrink eta drastically
                eta *= 0.1
                if eta < 1e-10:
                    break
            
            # Optional: check for stagnation
            # if it > 0 and abs(E - best_E) < 1e-10:
            #     break

        return best_P, best_E

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initializations
    inits = []
    inits.append(fibonacci_spiral(N))
    for i in range(5):
        inits.append(random_on_sphere(N, rng))
        
    # Optimize each initialization
    for P_init in inits:
        P_opt, E_opt = optimize(P_init)
        if E_opt < best_energy:
            best_energy = E_opt
            P_best = P_opt

    # Final polish: run a few more iterations on the best found
    if P_best is not None:
        P_final, E_final = optimize(P_best, max_iters=500, tol=1e-14)
        if E_final < best_energy:
            best_energy = E_final
            P_best = P_final

    return P_best
