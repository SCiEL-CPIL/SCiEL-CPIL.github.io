# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use small-support distributions (3-5 points) with analytical structure,
    then optimize weights using scipy's L-BFGS-B with exact gradient computation.
    """
    start_time = time.time()
    deadline = start_time + 595.0  # 5s safety margin
    
    def calc_prob_from_support_weights(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for i.i.d. X from discrete measure.
        """
        n = len(weights)
        w = weights / np.sum(weights)
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            inner += w[i] * w[j]
                P += w[k] * w[l] * inner
        return P
    
    def calc_prob_gradient(support, weights):
        """
        Compute P and its gradient with respect to weights (unnormalized).
        Returns (P, grad) where grad[i] = dP/dweights[i].
        """
        n = len(weights)
        wsum = np.sum(weights)
        w = weights / wsum
        
        P = 0.0
        grad = np.zeros(n)
        
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                # A[k,l] = sum_{i,j: s[i]+s[j] < threshold} w[i]*w[j]
                A = 0.0
                # Also compute dA/dw[m] for each m
                dA_dw = np.zeros(n)
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            A += w[i] * w[j]
                            # d(w[i]*w[j])/dw[m] = delta(m,i)*w[j] + delta(m,j)*w[i]
                            if m := None:  # placeholder
                                pass
                            for m in range(n):
                                if m == i:
                                    dA_dw[m] += w[j]
                                if m == j:
                                    dA_dw[m] += w[i]
                
                # P += w[k]*w[l]*A
                # dP/dw[m] += (dw[k]/dw[m])*w[l]*A + w[k]*(dw[l]/dw[m])*A + w[k]*w[l]*(dA/dw[m])
                # dw[k]/dw[m] = delta(k,m)/wsum - w[k]/wsum^2
                for m in range(n):
                    dwk_dwm = (1.0/wsum if m == k else 0.0) - w[k]/(wsum*wsum)
                    dwl_dwm = (1.0/wsum if m == l else 0.0) - w[l]/(wsum*wsum)
                    grad[m] += dwk_dwm * w[l] * A + w[k] * dwl_dwm * A + w[k] * w[l] * dA_dw[m]
                P += w[k] * w[l] * A
        
        return P, grad
    
    def objective_and_grad(x, support):
        """Objective function for scipy: minimize -P. x are unnormalized weights."""
        P, grad = calc_prob_gradient(support, x)
        return -P, -grad
    
    def optimize_weights(support, initial_weights, max_iter=500):
        """Optimize weights using L-BFGS-B."""
        n = len(support)
        w0 = np.array(initial_weights, dtype=np.float64)
        w0 = np.maximum(w0, 1e-10)
        w0 = w0 / np.sum(w0)
        w0 = w0 * 1.0  # unnormalized, sum = 1
        
        best_P = calc_prob_from_support_weights(support, w0)
        best_w = w0.copy()
        
        def obj(x):
            return -calc_prob_from_support_weights(support, x)
        
        def obj_grad(x):
            P, g = calc_prob_gradient(support, x)
            return -P, -g
        
        bounds = [(1e-10, None)] * n
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                       'jac': lambda x: np.ones(n)}]
        
        try:
            res = minimize(obj_grad, w0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-12})
            if res.success and res.fun < -best_P:
                best_P = -res.fun
                best_w = res.x.copy()
        except Exception:
            pass
        
        return best_P, best_w
    
    # Try multiple support configurations
    best_overall_P = 0.0
    best_overall_support = None
    best_overall_weights = None
    
    support_configs = [
        # (support_points, initial_weights)
        ([0.0, 1.0, 10.0], [0.3, 0.3, 0.4]),
        ([0.0, 1.0, 5.0], [0.3, 0.3, 0.4]),
        ([0.0, 1.0, 20.0], [0.2, 0.3, 0.5]),
        ([0.0, 0.5, 1.0, 10.0], [0.2, 0.2, 0.2, 0.4]),
        ([0.0, 0.5, 1.0, 5.0], [0.2, 0.2, 0.2, 0.4]),
        ([0.0, 0.25, 0.5, 1.0, 10.0], [0.15, 0.15, 0.15, 0.15, 0.4]),
        ([0.0, 0.5, 1.0, 1.5, 10.0], [0.12, 0.12, 0.12, 0.12, 0.52]),
        ([0.0, 1.0, 2.0, 10.0], [0.2, 0.2, 0.2, 0.4]),
        ([0.0, 0.3, 0.6, 1.0, 10.0], [0.15, 0.15, 0.15, 0.15, 0.4]),
    ]
    
    for support_pts, init_w in support_configs:
        if time.time() > deadline - 30:
            break
        support = np.array(support_pts, dtype=np.float64)
        n = len(support)
        
        # Coarse grid search to find good starting point
        best_P = 0.0
        best_w = np.array(init_w, dtype=np.float64)
        
        # Simple coordinate descent for initial optimization
        for iteration in range(20):
            if time.time() > deadline - 30:
                break
            improved = False
            for i in range(n):
                # Try to optimize weight i by varying it
                current_P = calc_prob_from_support_weights(support, best_w)
                for delta in [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]:
                    new_w = best_w.copy()
                    new_w[i] += delta
                    # Renormalize
                    if np.any(new_w < -1e-10):
                        continue
                    new_w = np.maximum(new_w, 0.0)
                    s = np.sum(new_w)
                    if s < 1e-10:
                        continue
                    new_w = new_w / s
                    P = calc_prob_from_support_weights(support, new_w)
                    if P > best_P + 1e-12:
                        best_P = P
                        best_w = new_w.copy()
                        improved = True
            if not improved:
                break
        
        # Now use scipy for fine optimization
        P_opt, w_opt = optimize_weights(support, best_w, max_iter=200)
        
        if P_opt > best_overall_P:
            best_overall_P = P_opt
            best_overall_support = support.copy()
            best_overall_weights = w_opt.copy()
    
    # If we didn't find anything, use a default
    if best_overall_support is None:
        best_overall_support = np.array([0.0, 1.0, 10.0], dtype=np.float64)
        best_overall_weights = np.array([0.3, 0.3, 0.4], dtype=np.float64)
    
    # Final verification
    final_P = calc_prob_from_support_weights(best_overall_support, best_overall_weights)
    
    return best_overall_support, best_overall_weights


# EVOLVE-BLOCK-END
