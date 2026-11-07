"""BBOB f14 (Different powers) optimizer.

Concurrent multi-start Nelder-Mead ensemble using Latin Hypercube sampling
for diverse initializations. This provides immediate local convergence from
multiple basins, maximizing early AOCC progress.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.asarray(x, dtype=float))

        # Number of concurrent Nelder-Mead agents
        n_agents = 5
        
        # Latin Hypercube Sampling for diverse initial points
        # Each dimension is divided into n_agents strata
        lhs_points = []
        for d in range(dim):
            # Create strata boundaries
            strata = np.linspace(-5, 5, n_agents + 1)
            # Sample one point per stratum
            for i in range(n_agents):
                lhs_points.append(rng.uniform(strata[i], strata[i+1]))
        
        # Reshape to (n_agents, dim)
        lhs_points = np.array(lhs_points).reshape(n_agents, dim)
        
        # Initialize agents with LHS points
        agents = []
        for i in range(n_agents):
            x0 = lhs_points[i]
            f0 = call(x0)
            record(x0, f0)
            agents.append({'x': x0, 'f': f0})
        
        # If we have budget left, start Nelder-Mead on each agent
        # Allocate remaining budget among agents
        remaining = budget - calls
        if remaining > 0:
            budget_per_agent = max(1, remaining // n_agents)
            
            for agent_idx in range(n_agents):
                if not ok():
                    break
                
                # Run Nelder-Mead for this agent
                x0 = agents[agent_idx]['x']
                f0 = agents[agent_idx]['f']
                
                # Build simplex around x0
                simp = [x0.copy()]
                fs = [f0]
                
                step = 0.5  # Initial simplex size
                for d in range(dim):
                    if not ok():
                        break
                    xi = x0.copy()
                    xi[d] += step
                    xi = np.clip(xi, -5, 5)
                    fi = call(xi)
                    record(xi, fi)
                    simp.append(xi.copy())
                    fs.append(fi)
                
                # Nelder-Mead parameters
                gamma = 1.0   # reflection
                rho = 2.0     # expansion
                sig = 0.5     # contraction
                shr = 0.5     # shrink
                
                # Run Nelder-Mead iterations
                max_iter = budget_per_agent
                for it in range(max_iter):
                    if not ok():
                        break
                    
                    # Sort simplex
                    order = np.argsort(fs)
                    simp = [simp[i] for i in order]
                    fs = [fs[i] for i in order]
                    
                    x_best = simp[0]
                    f_best = fs[0]
                    x_worst = simp[-1]
                    f_worst = fs[-1]
                    f_second_worst = fs[-2]
                    centroid = np.mean(simp[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + gamma * (centroid - x_worst)
                    xr = np.clip(xr, -5, 5)
                    if not ok():
                        break
                    fr = call(xr)
                    record(xr, fr)
                    
                    if fr < f_best:
                        # Expansion
                        xe = centroid + rho * (xr - centroid)
                        xe = np.clip(xe, -5, 5)
                        if not ok():
                            break
                        fe = call(xe)
                        record(xe, fe)
                        if fe < fr:
                            simp[-1] = xe
                            fs[-1] = fe
                        else:
                            simp[-1] = xr
                            fs[-1] = fr
                    elif fr < f_second_worst:
                        # Accept reflection
                        simp[-1] = xr
                        fs[-1] = fr
                    else:
                        # Contraction
                        if fr < f_worst:
                            xc = centroid + sig * (xr - centroid)
                        else:
                            xc = centroid - sig * (x_worst - centroid)
                        xc = np.clip(xc, -5, 5)
                        if not ok():
                            break
                        fc = call(xc)
                        record(xc, fc)
                        if fc < min(fr, f_worst):
                            simp[-1] = xc
                            fs[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simp)):
                                if not ok():
                                    break
                                simp[i] = x_best + shr * (simp[i] - x_best)
                                simp[i] = np.clip(simp[i], -5, 5)
                                fs[i] = call(simp[i])
                                record(simp[i], fs[i])
                    
                    # Convergence check
                    if np.max(fs) - np.min(fs) < 1e-10:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
