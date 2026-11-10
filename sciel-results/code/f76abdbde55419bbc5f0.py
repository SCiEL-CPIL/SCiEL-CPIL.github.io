import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    timeout = 600.0

    def compute_entropies(P):
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-9:
            return 0.0, 0.0, 0.0, 0.0
        P = P / s

        px = P.sum(axis=0)
        px = px[px > 0]
        H0 = -np.sum(px * np.log2(px))

        pzy = np.zeros(19)
        for y in range(n):
            for x in range(n):
                pzy[x + y] += P[y, x]
        pzy = pzy[pzy > 0]
        H1 = -np.sum(pzy * np.log2(pzy))

        pzy2 = np.zeros(28)
        for y in range(n):
            for x in range(n):
                pzy2[x + 2 * y] += P[y, x]
        pzy2 = pzy2[pzy2 > 0]
        H2 = -np.sum(pzy2 * np.log2(pzy2))

        pzy_minus = np.zeros(19)
        for y in range(n):
            for x in range(n):
                pzy_minus[x - y + 9] += P[y, x]
        pzy_minus = pzy_minus[pzy_minus > 0]
        Hm1 = -np.sum(pzy_minus * np.log2(pzy_minus))

        return H0, H1, H2, Hm1

    def soft_max(values, temperature=0.01):
        """Smooth approximation of max using log-sum-exp"""
        max_val = np.max(values)
        if max_val < 1e-9:
            return max_val
        exp_vals = np.exp((values - max_val) / temperature)
        return max_val + temperature * np.log(np.sum(exp_vals))

    def objective(params):
        r = 4
        U = params[:n*r].reshape(n, r)
        V = params[n*r:].reshape(n, r)
        P = U @ V.T
        # Use softplus to ensure non-negativity
        P = np.log1p(np.exp(P))  # softplus
        
        H0, H1, H2, Hm1 = compute_entropies(P)
        M = soft_max(np.array([H0, H1, H2]))
        if M < 1e-9:
            return 1e6
        ratio = Hm1 / M
        return -ratio

    best_P = np.ones((n, n)) / (n*n)
    best_score = 1.0

    # Try multiple random initializations with different ranks
    for i in range(20):
        if time.time() - start_time > timeout - 30:
            break
        
        np.random.seed(42 + i)
        rank = min(4, n)
        U = np.random.randn(n, rank) * 0.5
        V = np.random.randn(n, rank) * 0.5
        params0 = np.concatenate([U.flatten(), V.flatten()])
        
        # First pass: L-BFGS-B with moderate tolerance
        res = minimize(objective, params0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-7})
        
        if res.fun < -best_score + 1e-6:
            U_opt = res.x[:n*rank].reshape(n, rank)
            V_opt = res.x[n*rank:].reshape(n, rank)
            P_opt = U_opt @ V_opt.T
            P_opt = np.log1p(np.exp(P_opt))  # softplus
            P_opt = np.clip(P_opt, 0, None)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                H0, H1, H2, Hm1 = compute_entropies(P_opt)
                M = max(H0, H1, H2)
                if M > 1e-9:
                    score = Hm1 / M
                    if score > best_score:
                        best_score = score
                        best_P = P_opt.copy()
        
        # Second pass: polish with tighter tolerance if time allows
        if time.time() - start_time < timeout - 60:
            res2 = minimize(objective, res.x, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-10})
            if res2.fun < -best_score + 1e-8:
                U_opt = res2.x[:n*rank].reshape(n, rank)
                V_opt = res2.x[n*rank:].reshape(n, rank)
                P_opt = U_opt @ V_opt.T
                P_opt = np.log1p(np.exp(P_opt))
                P_opt = np.clip(P_opt, 0, None)
                s = P_opt.sum()
                if s > 1e-9:
                    P_opt /= s
                    H0, H1, H2, Hm1 = compute_entropies(P_opt)
                    M = max(H0, H1, H2)
                    if M > 1e-9:
                        score = Hm1 / M
                        if score > best_score:
                            best_score = score
                            best_P = P_opt.copy()

    # Final: try direct 100D optimization on the best P found as starting point
    if time.time() - start_time < timeout - 120:
        def objective_direct(P_flat):
            P = P_flat.reshape(n, n)
            P = np.clip(P, 0, None)
            s = P.sum()
            if s <= 1e-9:
                return 1e6
            P = P / s
            H0, H1, H2, Hm1 = compute_entropies(P)
            M = soft_max(np.array([H0, H1, H2]))
            if M < 1e-9:
                return 1e6
            return -Hm1 / M

        P_flat0 = best_P.flatten()
        res_direct = minimize(objective_direct, P_flat0, method='SLSQP',
                             bounds=[(0, None)] * (n*n),
                             options={'maxiter': 1000, 'ftol': 1e-12})
        
        P_direct = res_direct.x.reshape(n, n)
        P_direct = np.clip(P_direct, 0, None)
        s = P_direct.sum()
        if s > 1e-9:
            P_direct /= s
            H0, H1, H2, Hm1 = compute_entropies(P_direct)
            M = max(H0, H1, H2)
            if M > 1e-9:
                score = Hm1 / M
                if score > best_score:
                    best_score = score
                    best_P = P_direct.copy()

    # Ensure final validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P /= s
    else:
        best_P = np.ones((n, n)) / (n*n)

    return best_P
