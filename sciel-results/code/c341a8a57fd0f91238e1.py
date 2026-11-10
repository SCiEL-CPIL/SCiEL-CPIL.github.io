import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    N = 256
    support = np.arange(N, dtype=np.float64)
    
    def compute_P(weights):
        P_Y = np.convolve(weights, weights, mode='full')
        CDF_Y = np.cumsum(P_Y)
        P_S3 = np.convolve(P_Y, weights, mode='full')
        CDF_S3 = np.cumsum(P_S3)
        
        total = 0.0
        for m in range(N):
            s_m = support[m]
            limit_q = int(2 * s_m)
            if limit_q > len(CDF_S3):
                q_val = CDF_S3[-1]
            elif limit_q <= 0:
                q_val = 0.0
            else:
                idx_q = min(limit_q - 1, len(CDF_S3) - 1)
                q_val = CDF_S3[idx_q]
            total += weights[m] * q_val
        return total
    
    def compute_grad(weights):
        P_Y = np.convolve(weights, weights, mode='full')
        CDF_Y = np.cumsum(P_Y)
        P_S3 = np.convolve(P_Y, weights, mode='full')
        CDF_S3 = np.cumsum(P_S3)
        
        grad = np.zeros(N)
        for m in range(N):
            s_m = support[m]
            limit_q = int(2 * s_m)
            if limit_q > len(CDF_S3):
                q_val = CDF_S3[-1]
            elif limit_q <= 0:
                q_val = 0.0
            else:
                idx_q = min(limit_q - 1, len(CDF_S3) - 1)
                q_val = CDF_S3[idx_q]
            
            r_val = 0.0
            for l in range(N):
                s_l = support[l]
                limit_r = int(2 * s_l - s_m)
                if limit_r > len(CDF_Y):
                    p_val = CDF_Y[-1]
                elif limit_r <= 0:
                    p_val = 0.0
                else:
                    idx_r = min(limit_r - 1, len(CDF_Y) - 1)
                    p_val = CDF_Y[idx_r]
                r_val += weights[l] * p_val
            
            grad[m] = q_val + 3.0 * r_val
        return grad
    
    def project_simplex(w):
        w = np.clip(w, 0, None)
        s = np.sum(w)
        if s > 0:
            w = w / s
        else:
            w = np.ones(N) / N
        return w
    
    best_weights = None
    best_P = -1.0
    
    initializations = [
        np.ones(N) / N,
        np.concatenate([np.array([0.7, 0.3]), np.zeros(N-2)]),
        np.concatenate([np.array([0.5, 0.3, 0.2]), np.zeros(N-3)]),
        np.concatenate([np.array([0.4]), np.zeros(N-2), np.array([0.6])]),
        np.concatenate([np.array([0.6, 0.2, 0.1, 0.1]), np.zeros(N-4)]),
        np.concatenate([np.array([0.3, 0.3, 0.2, 0.2]), np.zeros(N-4)]),
        np.concatenate([np.array([0.5, 0.5]), np.zeros(N-2)]),
        np.concatenate([np.array([0.8, 0.1, 0.1]), np.zeros(N-3)]),
    ]
    
    for init_w in initializations:
        weights = project_simplex(init_w.copy())
        step = 0.05
        for it in range(500):
            grad = compute_grad(weights)
            P_old = compute_P(weights)
            
            alpha = step
            improved = False
            for _ in range(20):
                new_w = project_simplex(weights + alpha * grad)
                P_new = compute_P(new_w)
                if P_new > P_old:
                    weights = new_w
                    improved = True
                    break
                alpha *= 0.5
            
            if improved:
                P_cur = compute_P(weights)
                if P_cur > best_P:
                    best_P = P_cur
                    best_weights = weights.copy()
                step = min(step * 1.2, 0.3)
            else:
                step = max(step * 0.5, 0.0005)
    
    return support, best_weights
