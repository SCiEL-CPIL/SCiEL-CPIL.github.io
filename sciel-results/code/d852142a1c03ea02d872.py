import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0
    
    # Use a small set of k to solve LP
    K_max = 50
    ks = np.arange(2, K_max + 1)
    m = len(ks)
    
    # Objective: maximize -sum f_k log(k)/k
    # c = -log(ks)/ks
    c = -np.log(ks) / ks
    
    # Constraints: M(x) = f(1)*floor(x/1) + sum_{k>=2} f(k)*floor(x/k) <= 1
    # where f(1) = -sum_{k>=2} f(k)/k
    # So M(x) = -x*sum_{k>=2} f(k)/k + sum_{k>=2} f(k)*floor(x/k)
    #         = sum_{k>=2} f(k) * (floor(x/k) - x/k)
    # Constraint: sum_{k>=2} f(k) * (floor(x/k) - x/k) <= 1
    c_obj = -c
    
    A_ub = []
    b_ub = []
    
    max_x = 10 * K_max
    for x in range(1, max_x + 1):
        row = np.array([float(x) // k - x / k for k in ks])
        A_ub.append(row)
        b_ub.append(1.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: -10 <= f_k <= 10
    bounds = [(-10, 10)] * m
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            f_vals = res.x
            # Construct domain and values
            # f(1) = -sum_{k>=2} f(k)/k
            f1 = -np.sum(f_vals / ks)
            
            domain = np.concatenate(([1], ks))
            values = np.concatenate(([f1], f_vals))
            
            # Verify: after clipping and f(1) replacement, M(x) <= 1
            # Clip values
            clipped = np.clip(values, -10, 10)
            S = np.sum(clipped / domain)
            # Replace f(1)
            idx1 = np.where(domain == 1)[0]
            if len(idx1) > 0:
                clipped[idx1[0]] = clipped[idx1[0]] - S
            else:
                # f(1) = 0 before replacement, so new f(1) = -S
                clipped = np.insert(clipped, 0, -S)
                domain = np.insert(domain, 0, 1)
            
            # Check M(x)
            feasible = True
            K_check = int(domain.max())
            for x in range(1, 10 * K_check + 1):
                Mx = np.sum(clipped * (x // domain))
                if Mx > 1.0001:
                    feasible = False
                    break
            
            if feasible:
                return domain, values
        else:
            pass
    except Exception:
        pass
    
    # Fallback to Mobius
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
