import numpy as np
import time
import scipy.optimize as opt

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n
    w0 = 1.0 / n
    
    # Coarse quadrature for optimization
    M_coarse = 100
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    w_coarse = (1.0 - y_coarse) / n
    a_y_coarse = np.outer(y_coarse, a)
    
    # Full quadrature for final evaluation
    M_full = 20000
    y_full = (np.arange(M_full) + 0.5) / M_full
    w_full = (1.0 - y_full) / n
    a_y_full = np.outer(y_full, a)

    def compute_union_length(L, w):
        """
        L: (M, n) array of left endpoints.
        w: (M,) array of widths.
        Returns union length for each row.
        """
        M, n = L.shape
        union_lengths = np.zeros(M)
        for m in range(M):
            l = np.sort(L[m])
            r = l + w[m]
            current_max_r = -np.inf
            total_len = 0.0
            for j in range(n):
                if l[j] > current_max_r:
                    total_len += r[j] - l[j]
                    current_max_r = r[j]
                else:
                    if r[j] > current_max_r:
                        total_len += r[j] - current_max_r
                        current_max_r = r[j]
            union_lengths[m] = total_len
        return union_lengths

    def area_coarse(x):
        L = x[None, :] + a_y_coarse
        return np.mean(compute_union_length(L, w_coarse))

    def grad_coarse(x):
        eps = 1e-5
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (area_coarse(x_plus) - area_coarse(x_minus)) / (2 * eps)
        return grad

    def area_full(x):
        L = x[None, :] + a_y_full
        return np.mean(compute_union_length(L, w_full))

    def grad_full(x):
        eps = 1e-6
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (area_full(x_plus) - area_full(x_minus)) / (2 * eps)
        return grad

    np.random.seed(42)
    best_x = None
    best_area = np.inf
    
    deadline = time.time() + 500
    
    restarts = 15
    
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            k = int(np.log2(n))
            j = np.arange(1, k + 1)
            x = []
            for i in range(n):
                eps = np.array(list(str(bin(i))[2:]), dtype=int)
                eps = np.pad(eps, (k - len(eps), 0), "constant")
                x.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
            x = np.array(x, dtype=np.float64)
        else:
            x = np.random.uniform(-1.0, 1.0, n)
            
        # Optimize with coarse area
        res = opt.minimize(area_coarse, x, jac=grad_coarse, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-5})
        
        x_opt = res.x
        
        # Polish with full area
        res_full = opt.minimize(area_full, x_opt, jac=grad_full, method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-4})
        
        x_final = res_full.x
        area_val = area_full(x_final)
        
        if area_val < best_area:
            best_area = area_val
            best_x = x_final.copy()

    if best_x is None:
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        best_x = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            best_x.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        best_x = np.array(best_x, dtype=np.float64)

    return best_x
