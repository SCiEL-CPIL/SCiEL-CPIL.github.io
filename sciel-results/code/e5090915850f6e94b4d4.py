import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _objective(h):
    """
    Objective function for minimization (we want to maximize score).
    """
    h = np.clip(h, 0, None)
    s = _score_raw(h)
    return -s

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initial best
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Define bounds for DE
    # Heights are non-negative. We can bound them to [0, 1] since we normalize or scale doesn't matter.
    # Actually, if we fix sum=1, bounds are [0, 1].
    # But DE works better if we don't force sum=1 in the bounds, but in the objective?
    # No, let's just bound [0, 10] and normalize in objective?
    # If we normalize in objective, the landscape is flat in scale direction.
    # Better to fix sum=1? DE doesn't handle equality constraints well.
    # Let's just optimize h in [0, 1] and normalize in _score_raw?
    # _score_raw is scale invariant. So optimizing h in [0, 1] is fine.
    
    bounds = [(0.0, 1.0)] * N
    
    # Strategy: 
    # 1. Run DE with diverse initial population.
    # We can generate a custom initial population.
    
    def generate_initial_population(popsize):
        pop = []
        # 1. Uniform
        pop.append(np.ones(N) / N)
        
        # 2. Ramps
        for start in range(0, N-10, 5):
            for length in range(10, N-start, 5):
                h = np.zeros(N)
                h[start:start+length] = np.linspace(0.1, 1.0, length)
                if np.sum(h) > 0:
                    h /= np.sum(h)
                    pop.append(h)
                    
        # 3. Asymmetric Gaussians
        for center in range(5, N-5, 5):
            for sigma in [2, 5, 10]:
                for asym in [0.5, 1.0, 2.0]:
                    x = np.arange(N)
                    # Skewed gaussian
                    h = np.exp(-0.5 * ((x - center) / (sigma * (1 + asym * (x > center))))**2)
                    # Simpler: just shift center
                    h = np.exp(-0.5 * ((x - center) / sigma)**2)
                    # Add a tail
                    if asym != 1.0:
                         h[x > center] *= asym
                    if np.sum(h) > 0:
                        h /= np.sum(h)
                        pop.append(h)
                        
        # 4. Random Sparse
        np.random.seed(42)
        while len(pop) < popsize:
            k = np.random.randint(3, 15)
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = np.random.rand(k)
            if np.sum(h) > 0:
                h /= np.sum(h)
                pop.append(h)
                
        return pop[:popsize]

    popsize = 50
    init_pop = generate_initial_population(popsize)
    
    try:
        de_res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=popsize,
            tol=0.001,
            init=init_pop, # Pass custom init
            polish=False
        )
        
        if de_res.success or de_res.fun < 0: # fun is negative score
            h_de = np.clip(de_res.x, 0, None)
            if np.sum(h_de) > 1e-15:
                h_de /= np.sum(h_de)
                s_de = _score_raw(h_de)
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de.copy()
    except Exception as e:
        pass
        
    # Refine with L-BFGS-B
    # We need a differentiable objective. 
    # Approximate max with softmax.
    def _objective_smooth(params):
        h = np.clip(params, 0, None)
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        # Softmax max
        temp = 1.0
        max_conv = np.log(np.sum(np.exp(h_conv / temp))) * temp
        if max_conv <= 1e-15:
            return 0.0
        s = l2_sq / (DELTA * max_conv)
        return -s

    # Start refinement from best_h and a few others
    starts = [best_h.copy()]
    # Add random smooth starts
    for _ in range(5):
        h_rand = np.random.rand(N)
        h_rand /= np.sum(h_rand)
        starts.append(h_rand)
        
    for h_start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _objective_smooth, 
                h_start, 
                method='L-BFGS-B', 
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.success:
                h_opt = np.clip(res.x, 0, None)
                if np.sum(h_opt) > 1e-15:
                    h_opt /= np.sum(h_opt)
                    s_opt = _score_raw(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
            
    return list(best_h)
