import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    N = 24
    # Use a dense grid for local search
    M = 2000 
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * thetas)
    # Precompute powers of z
    powers = np.vander(z, N, increasing=True).T
    
    best_coeffs = None
    best_sup = np.inf
    
    def eval_sup(coeffs):
        vals = powers @ coeffs
        return np.max(np.abs(vals))
    
    def local_search(start_coeffs):
        coeffs = start_coeffs.copy()
        sup = eval_sup(coeffs)
        improved = True
        while improved:
            improved = False
            vals = powers @ coeffs
            # Calculate change if flipping bit j
            # new_val = val - 2 * c_j * z^j
            flips = powers * (2 * coeffs)[:, None]
            all_vals = vals[None, :, :] - flips
            all_sups = np.max(np.abs(all_vals), axis=1)
            
            best_j = np.argmin(all_sups)
            new_sup = all_sups[best_j]
            
            if new_sup < sup - 1e-12:
                coeffs[best_j] *= -1
                sup = new_sup
                improved = True
                
            if time.time() > deadline:
                break
        return coeffs, sup

    # 1. Try GRS sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:N], dtype=np.float64)
    c, s = local_search(grs)
    if s < best_sup:
        best_sup = s
        best_coeffs = c
        
    if time.time() > deadline:
        return best_coeffs.astype(int).tolist()

    # 2. Random restarts
    while time.time() < deadline:
        random_coeffs = np.random.choice([-1, 1], size=N)
        c, s = local_search(random_coeffs)
        if s < best_sup:
            best_sup = s
            best_coeffs = c
            
    # 3. Try a few structured patterns
    patterns = [
        np.ones(N),
        np.ones(N) * np.array([1 if i%2==0 else -1 for i in range(N)]),
        np.array([1 if i < N//2 else -1 for i in range(N)]),
        np.array([1 if (i//2)%2==0 else -1 for i in range(N)])
    ]
    for pat in patterns:
        if time.time() > deadline: break
        c, s = local_search(pat)
        if s < best_sup:
            best_sup = s
            best_coeffs = c

    return best_coeffs.astype(int).tolist()
