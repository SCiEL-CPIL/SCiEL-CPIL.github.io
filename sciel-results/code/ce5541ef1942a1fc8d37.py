import numpy as np
import time

def construct():
    deadline = time.time() + 140
    
    n_points = 10000
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    z = np.exp(1j * angles)
    powers = np.empty((24, n_points), dtype=complex)
    powers[0] = 1.0
    for k in range(1, 24):
        powers[k] = powers[k-1] * z
    
    def fast_eval(coeffs):
        vals = coeffs @ powers
        return np.max(np.abs(vals))
    
    def full_eval(coeffs):
        base = fast_eval(coeffs)
        abs_vals = np.abs(coeffs @ powers)
        top_idx = np.argsort(abs_vals)[-5:]
        
        for idx in top_idx:
            a = angles[(idx - 1) % n_points]
            b = angles[(idx + 1) % n_points]
            if a > b:
                a, b = b, a
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            for _ in range(50):
                if time.time() > deadline:
                    break
                fc = abs(np.exp(1j*c) ** np.arange(24) @ coeffs)
                fd = abs(np.exp(1j*d) ** np.arange(24) @ coeffs)
                if fc < fd:
                    b = d
                    d = c
                    c = b - gr * (b - a)
                else:
                    a = c
                    c = d
                    d = a + gr * (b - a)
            mid = (a + b) / 2
            val = abs(np.exp(1j*mid) ** np.arange(24) @ coeffs)
            if val > base:
                base = val
        return base
    
    def local_search(coeffs):
        """Flip each bit if it improves the fast eval."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            cur_val = fast_eval(coeffs)
            for pos in range(24):
                new_coeffs = coeffs.copy()
                new_coeffs[pos] *= -1
                new_val = fast_eval(new_coeffs)
                if new_val < cur_val:
                    coeffs = new_coeffs
                    cur_val = new_val
                    improved = True
        return coeffs, fast_eval(coeffs)
    
    # Golay-Rudin-Shapiro initial
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    
    best_coeffs = None
    best_val = float('inf')
    
    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            coeffs = np.array(p[:24], dtype=float)
        else:
            coeffs = np.where(np.random.RandomState(10000 + restart).rand(24) < 0.5, 1.0, -1.0)
        
        # Local search first
        coeffs, cur_val = local_search(coeffs)
        
        if cur_val < best_val:
            full_val = full_eval(coeffs)
            if full_val < best_val:
                best_val = full_val
                best_coeffs = coeffs.copy()
        
        # Simulated annealing
        T = 0.05
        T_min = 0.0005
        cooling = 0.9998
        rng = np.random.RandomState(20000 + restart)
        
        while T > T_min and time.time() < deadline - 10:
            pos = rng.randint(24)
            new_coeffs = coeffs.copy()
            new_coeffs[pos] *= -1
            
            new_val = fast_eval(new_coeffs)
            
            if new_val < best_val:
                full_new_val = full_eval(new_coeffs)
                if full_new_val < best_val:
                    best_val = full_new_val
                    best_coeffs = new_coeffs.copy()
            
            delta = new_val - cur_val
            if delta < 0 or rng.random() < np.exp(-delta / T):
                coeffs = new_coeffs
                cur_val = new_val
            
            T *= cooling
        
        # Final local search
        coeffs, cur_val = local_search(coeffs)
        if cur_val < best_val:
            full_val = full_eval(coeffs)
            if full_val < best_val:
                best_val = full_val
                best_coeffs = coeffs.copy()
    
    if best_coeffs is None:
        best_coeffs = np.array(p[:24], dtype=float)
    
    return best_coeffs.tolist()
