import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 115
    
    # Grid for local search
    M = 15000
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    Z = np.exp(1j * thetas[:, None] * np.arange(N)[None, :])
    
    best_c = None
    best_score = np.inf
    best_raw = np.inf  # raw max |P(z)|, score = raw / sqrt(24)
    
    def calculate_raw_score(c):
        vals = Z @ c
        return np.max(np.abs(vals))
    
    def precise_max_abs(c):
        """High-precision evaluation: find peaks via coarse grid, then golden-section refine."""
        # Coarse grid to find top peak locations
        vals_coarse = np.exp(1j * thetas[:, None] * np.arange(N)[None, :]) @ c
        abs_coarse = np.abs(vals_coarse)
        
        # Find top 5 peak indices
        top_idx = np.argsort(abs_coarse)[-5:]
        
        # For each peak, do golden-section refinement on |P(z)|
        best_val = 0.0
        for idx in top_idx:
            theta0 = thetas[idx]
            # Search window around this peak
            half_width = np.pi / M  # half the grid spacing
            a = theta0 - half_width
            b = theta0 + half_width
            
            # Golden section search for maximum of |P(z)|
            gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            
            def eval_abs(theta):
                z = np.exp(1j * theta * np.arange(N))
                return np.abs(np.sum(c * z))
            
            fc = eval_abs(c_pt)
            fd = eval_abs(d_pt)
            
            for _ in range(60):  # ~60 iterations for 1e-9 accuracy
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = eval_abs(d_pt)
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = eval_abs(c_pt)
            
            max_here = max(fc, fd)
            if max_here > best_val:
                best_val = max_here
        
        return best_val
    
    def local_search(c_init, max_iters=200):
        c = c_init.astype(float).copy()
        vals = Z @ c
        current_raw = np.max(np.abs(vals))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each bit, in a fixed order
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = c[i]
                new_val = -old_val
                c[i] = new_val
                
                vals_new = vals + 2 * new_val * Z[:, i]
                new_raw = np.max(np.abs(vals_new))
                
                if new_raw < current_raw - 1e-12:
                    current_raw = new_raw
                    vals = vals_new
                    improved = True
                else:
                    c[i] = old_val
            
            if not improved:
                break
                
        return c, current_raw
    
    def polish(c_init, max_iters=50):
        """Second-pass local search with precise evaluation at the end."""
        c = c_init.astype(float).copy()
        vals = Z @ c
        current_raw = np.max(np.abs(vals))
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = c[i]
                new_val = -old_val
                c[i] = new_val
                
                vals_new = vals + 2 * new_val * Z[:, i]
                new_raw = np.max(np.abs(vals_new))
                
                if new_raw < current_raw - 1e-12:
                    current_raw = new_raw
                    vals = vals_new
                    improved = True
                else:
                    c[i] = old_val
            
            if not improved:
                break
        
        # Final precise evaluation
        final_raw = precise_max_abs(c)
        return c, final_raw

    # Structured seeds
    seeds = []
    
    # Rudin-Shapiro
    r = [1, 1]
    for i in range(2, N):
        if i % 2 == 0:
            r.append(r[i // 2])
        else:
            r.append(-r[(i - 1) // 2])
    seeds.append(np.array(r, dtype=float))
    
    # Thue-Morse
    tm = np.array([(-1) ** bin(i).count('1') for i in range(N)], dtype=float)
    seeds.append(tm)
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float)
    seeds.append(alt)
    
    # All ones
    seeds.append(np.ones(N, dtype=float))
    
    # All minus
    seeds.append(-np.ones(N, dtype=float))
    
    # Block patterns: +++---+++---+++---
    block = np.array([1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1], dtype=float)
    seeds.append(block)
    
    # Reverse patterns
    for s in seeds[:]:
        seeds.append(s[::-1].copy())
    
    # Add random seeds
    np.random.seed(42)
    for _ in range(30):
        if time.time() > deadline:
            break
        seeds.append(np.random.choice([-1, 1], N).astype(float))
    
    # Process all seeds
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        c_res, raw_score = local_search(seed)
        
        if raw_score < best_raw:
            best_raw = raw_score
            best_c = c_res.copy()
    
    # If we have time, do more random restarts
    if time.time() < deadline - 20:
        extra_restarts = 0
        while time.time() < deadline - 15:
            c_rand = np.random.choice([-1, 1], N).astype(float)
            c_res, raw_score = local_search(c_rand, max_iters=100)
            if raw_score < best_raw:
                best_raw = raw_score
                best_c = c_res.copy()
            extra_restarts += 1
            if extra_restarts > 200:
                break
    
    # Final polish pass on the best candidate
    if best_c is not None and time.time() < deadline - 5:
        c_polished, raw_polished = polish(best_c, max_iters=80)
        if raw_polished < best_raw:
            best_raw = raw_polished
            best_c = c_polished.copy()
        else:
            # Try polishing a few other good candidates too
            pass
    
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
