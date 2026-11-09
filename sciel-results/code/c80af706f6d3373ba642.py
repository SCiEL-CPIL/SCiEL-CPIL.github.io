# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin

    # 1. Generate diverse initial population
    # We use log-probability parameters z, P = softmax(z)
    pop_size = 60
    pop = []
    best_z = None
    best_val = -1e9
    
    def init_pop():
        p = []
        # Uniform
        p.append(np.zeros(100))
        
        # Heuristic starts based on geometry
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        
        # Try to concentrate on structures that maximize D/M
        # D is anti-diagonals (x-y). M is max(col, diag, slope2)
        # We want x-y to be spread, x, x+y, x+2y to be concentrated.
        
        # Pattern 1: A single "thick" anti-diagonal? No, D needs spread.
        # Maybe a pattern that covers many anti-diagonals but few columns?
        # Impossible: if x varies, columns vary.
        # If we fix x, D is spread (x-y = c-y), but H_x = 0.
        # If H_x = 0, M = max(0, H_{x+y}, H_{x+2y}).
        # If x is fixed, x+y = x+y, x+2y = x+2y.
        # As y varies, x+y and x+2y vary.
        # Let's test x=0.
        # P[y, 0] = 1/10.
        # D: x-y = -y. y in 0..9. Values 0, -1, ..., -9. All distinct. H_D = log2(10).
        # M: H_x = 0. H_{x+y} = H_y = log2(10). H_{x+2y} = H_{2y} = log2(10).
        # M = log2(10). Ratio = 1.0.
        
        # Can we do better?
        # We need H_D > H_M.
        # H_D is max log2(10) if uniform on anti-diags.
        # If we can make H_M < log2(10), we win.
        # H_M is max of entropies of x, x+y, x+2y.
        # To make H_x small, concentrate x.
        # To make H_{x+y} small, concentrate x+y.
        # To make H_{x+2y} small, concentrate x+2y.
        
        # If we concentrate on a single point (x0, y0):
        # H_D = 0, H_M = 0. Ratio undefined/0.
        
        # If we concentrate on a line x=x0:
        # H_D = log2(10) (if y uniform).
        # H_M = log2(10).
        # Ratio = 1.
        
        # What if we use a non-uniform distribution on the line x=x0?
        # Let p(y) be the distribution.
        # H_D = H_{-y} = H(p).
        # H_x = 0.
        # H_{x+y} = H(p).
        # H_{x+2y} = H_{2y}. If p is symmetric/uniform, H_{2y} < H(p)?
        # No, if y is uniform 0..9, 2y is 0,2,..18. 10 distinct values. H=log2(10).
        # If we make p(y) non-uniform, H(p) decreases.
        # But H_{2y} might decrease differently.
        
        # Let's try a distribution on x=0 that is not uniform.
        # If p(y) is peaked, H(p) is small.
        # But D/M might be > 1 if H_{2y} drops faster than H(p)?
        # Actually, H_{2y} is the entropy of the transformed variable.
        # If y is discrete, 2y is injective on 0..9. So H_{2y} = H_y.
        # So on a vertical line, H_D = H_M. Ratio = 1.
        
        # We need a 2D structure.
        # Consider a "block" of cells.
        # If we pick cells such that x-y takes many values, but x, x+y, x+2y take few.
        # This is a combinatorial design problem.
        
        # Let's just use random starts and let the GA find the structure.
        
        # Add random starts
        rng = np.random.default_rng(42)
        for _ in range(pop_size - 2):
            p.append(rng.normal(0, 2.0, 100))
            
        # Add some structured starts
        # Start with vertical lines
        for x0 in [0, 4, 9]:
             z = np.full(100, -100.0)
             for y in range(10):
                 z[y*10 + x0] = 0.0
             p.append(z)
             
        return p

    pop = init_pop()
    
    # Evaluate initial population
    vals = []
    for z in pop:
        P = _softmax(z).reshape(10, 10)
        v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
        vals.append(v)
        if v > best_val:
            best_val = v
            best_z = z.copy()
            
    # Genetic Algorithm Loop
    generation = 0
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 10.0:
        generation += 1
        new_pop = []
        new_vals = []
        
        # Elitism: keep best
        idx_best = np.argmax(vals)
        new_pop.append(pop[idx_best].copy())
        new_vals.append(vals[idx_best])
        
        # Crossover and Mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            i = np.random.choice(pop_size)
            j = np.random.choice(pop_size)
            if vals[i] >= vals[j]:
                parent1 = pop[i]
            else:
                parent1 = pop[j]
                
            k = np.random.choice(pop_size)
            l = np.random.choice(pop_size)
            if vals[k] >= vals[l]:
                parent2 = pop[k]
            else:
                parent2 = pop[l]
                
            # Uniform Crossover
            mask = np.random.rand(100) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            # Gaussian noise
            noise = np.random.normal(0, temp, 100)
            child = child + noise
            
            # Occasionally, do a "restart" mutation: blend with a random start
            if np.random.rand() < 0.1:
                random_z = np.random.normal(0, 2.0, 100)
                child = 0.5 * child + 0.5 * random_z
            
            new_pop.append(child)
            
            # Evaluate
            P = _softmax(child).reshape(10, 10)
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            new_vals.append(v)
            
            if v > best_val:
                best_val = v
                best_z = child.copy()
                
        pop = new_pop
        vals = new_vals
        temp *= cooling
        
        # Every 100 generations, try to polish the best with a local step
        if generation % 100 == 0 and time.time() < deadline - 50.0:
             # Simple local search: try small perturbations of best_z
             for _ in range(20):
                 pert = best_z + np.random.normal(0, 0.1, 100)
                 P = _softmax(pert).reshape(10, 10)
                 v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
                 if v > best_val:
                     best_val = v
                     best_z = pert.copy()

    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P

# EVOLVE-BLOCK-END
