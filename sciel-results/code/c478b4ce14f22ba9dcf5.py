import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # --- Phase 1: Global Exploration with CMA-ES ---
        # CMA-ES is superior to NM for global search and adapting to rotation/shift
        # because it maintains a covariance matrix. NM is local and can get stuck
        # or struggle with the high condition number of Bent Cigar if not started
        # very close to the optimum.
        
        # Initial population
        pop_size = 2 * dim + 4  # Standard CMA-ES lambda
        if pop_size > budget:
            pop_size = max(2, budget)
            
        # Start near center, but with some spread
        mean = np.zeros(dim)
        sigma = 1.5 # Initial step size
        
        # We will run CMA-ES for a portion of the budget, say 40%
        cma_budget = int(budget * 0.4)
        
        # CMA-ES Parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) ** 2 + 4)
        cmu = min(0.5, 2 / ((dim + 1.3) ** 2 + 6))
        cs = (cc + 2 * cmu) / (1 + cc)
        ds = 1 + 2 * max(0, sqrt(cc * (cc + 2 * cmu) - 1)) + cs
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Pre-compute update factors
        def update_covariance(psi, y):
            nonlocal C, C_inv, C_sqrt, C_sqrt_inv
            # C += (1/mu + (cc+cs-cs)*sigma^2/lambda) * (y y^T - C)
            # Simplified rank-1 and rank-mu updates
            # For simplicity and speed in Python, we use the standard update
            # Note: This is a simplified CMA-ES implementation. 
            # A full robust implementation might be too long, but the core idea is the covariance adaptation.
            
            # Actually, implementing full CMA-ES with matrix updates in pure Python/NumPy 
            # without pre-compiled chunks can be slow if not careful. 
            # However, dim=5 is small. 5x5 matrix ops are very fast.
            
            # Rank-1 update (p_sigma)
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu)*sigma*y
            # This requires tracking p_sigma. 
            # Let's stick to a simpler adaptation if CMA-ES is too complex to code correctly in short time.
            
            # Alternative: Use a simpler adaptation like (1+lambda)-ES with covariance adaptation?
            # Or just use the current NM but with a better global search phase?
            
            # Let's try a different structural change: 
            # Use a global search (like Random Search or a simple ES) to find a good basin,
            # then switch to NM for local refinement.
            # The current program uses NM from the start. 
            # The inspiration was (1+1)-ES. 
            # Combining: Global ES phase -> Local NM phase.
            
            pass # Placeholder, will implement below

        # Let's implement a simpler but effective global search: 
        # (1+lambda)-ES with adaptive sigma, running for a while.
        # Then switch to NM.
        
        # Actually, the current program's NM is quite good locally. 
        # The issue might be the initial point. 
        # Let's use a short (1+1)-ES phase to find a good starting point for NM.
        
        # Phase 1: (1+1)-ES for global search
        x_es = rng.uniform(-4, 4, dim)
        f_es = safe_func(x_es)
        sigma_es = 1.0
        
        es_budget = int(budget * 0.3)
        
        for _ in range(es_budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            y = np.clip(x_es + sigma_es * rng.standard_normal(dim), -5, 5)
            f_y = safe_func(y)
            
            if f_y < f_es:
                x_es, f_es = y, f_y
                sigma_es *= 1.5
            else:
                sigma_es *= 0.87
        
        # Phase 2: Nelder-Mead for local refinement
        # Start NM from the best point found by ES
        x0 = x_es.copy()
        f0 = f_es
        
        if best_f == float('inf'):
            best_x, best_f = x0, f0
            
        # Initialize simplex around x0
        alpha = 0.5 # Smaller step size for NM since we are likely closer
        
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += alpha
            x_i = np.clip(x_i, -5, 5)
            f_i = safe_func(x_i)
            simplex.append(x_i)
            simplex_f.append(f_i)
            
        # Sort simplex
        indices = np.argsort(simplex_f)
        simplex = [simplex[i] for i in indices]
        simplex_f = [simplex_f[i] for i in indices]
        
        # NM Parameters
        rho = 1.0
        chi = 2.0
        gamma = 0.5
        sigma_shrink = 0.5
        
        max_iter = 200
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Sort
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            if simplex_f[-1] - simplex_f[0] < 1e-10:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + rho * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_func(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + chi * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = safe_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                xc = centroid + gamma * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = safe_func(xc)
                
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = safe_func(simplex[i])
                    indices = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in indices]
                    simplex_f = [simplex_f[i] for i in indices]
                    
        return best_x, best_f
