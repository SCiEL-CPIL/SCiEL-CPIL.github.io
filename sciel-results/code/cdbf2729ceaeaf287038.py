import numpy as np

N = 100

def construct():
    target_sum = N / 2.0
    
    def calc_R(h):
        R = np.zeros(N)
        for k in range(N):
            R[k] = np.sum(h[k:] * h[:-k])
        return R

    def project_sum_box(y, S):
        lo = np.min(y) - 1.0
        hi = np.max(y) + 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            h = np.clip(y - mid, 0.0, 1.0)
            s = np.sum(h)
            if s > S:
                lo = mid
            else:
                hi = mid
        h = np.clip(y - (lo + hi) / 2.0, 0.0, 1.0)
        diff = S - np.sum(h)
        if abs(diff) > 1e-12:
            mask = (h > 1e-9) & (h < 1 - 1e-9)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        return h

    def pgd_solve(L, h_init, steps=1000, eta=0.05):
        h = h_init.copy()
        for i in range(steps):
            R = calc_R(h)
            violations = np.maximum(0.0, L - R)
            viol_sq_sum = np.sum(violations**2)
            
            if viol_sq_sum < 1e-12:
                break
            
            grad = np.zeros(N)
            active_k = np.where(violations > 0)[0]
            for k in active_k:
                v = violations[k]
                if k > 0:
                    grad[k:] -= 2.0 * v * h[:-k]
                if k < N:
                    grad[:N-k] -= 2.0 * v * h[k:]
            
            h = h - eta * grad
            h = project_sum_box(h, target_sum)
            
        return h, viol_sq_sum

    h0 = np.full(N, 0.5)
    
    # Binary search for max L
    L_lo = 0.0
    L_hi = 25.0
    best_L = L_lo
    best_h = h0
    current_h = h0
    
    for _ in range(25):
        L_mid = (L_lo + L_hi) / 2.0
        h_sol, viol = pgd_solve(L_mid, current_h, steps=500, eta=0.05)
        
        if viol < 1e-8:
            best_L = L_mid
            best_h = h_sol
            L_lo = L_mid
            current_h = h_sol
        else:
            L_hi = L_mid
            
    # Final polish
    h_final, _ = pgd_solve(best_L, best_h, steps=2000, eta=0.01)
    
    # Ensure sum
    diff = target_sum - np.sum(h_final)
    if abs(diff) > 1e-9:
        mask = (h_final > 0) & (h_final < 1)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0, 1)
            
    return h_final
