import numpy as np
import time

def solve(a, b):
        n = len(a)
        start_time = time.time()
        deadline = 110.0 # Safety margin
        
        # Initial solution: Rank Matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
        best_p = p.copy()
        best_cost = np.sum(a * b[np.ix_(p, p)])
        
        # Helper to calculate cost
        def get_cost(p):
            return np.sum(a * b[np.ix_(p, p)])
            
        # Helper to calculate delta for swap i, j
        def calc_delta(p, i, j):
            k = p[i]
            l = p[j]
            # Vectorized delta
            # Sum over u of (a[i,u] - a[j,u]) * (b[k,p[u]] - b[l,p[u]])
            # Since b is symmetric and a is symmetric, and diag is 0:
            da = a[i, :] - a[j, :]
            db = b[k, p] - b[l, p]
            return np.dot(da, db)
            
        # Simulated Annealing
        T = 1000.0 # Initial temp
        T_min = 1.0
        alpha = 0.999 # Cooling rate
        
        current_p = p.copy()
        current_cost = best_cost
        
        # Main SA loop
        while time.time() < deadline:
            # Check time more frequently if needed, but 120s is long enough for many loops
            # Random swap
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            delta = calc_delta(current_p, i, j)
            
            # Acceptance
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Swap
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                    
            # Cooling
            T *= alpha
            if T < T_min:
                # Restart
                T = 100.0 # Lower restart temp
                # Perturb best solution
                current_p = best_p.copy()
                current_cost = best_cost
                # Random perturb
                for _ in range(5):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        current_p[i], current_p[j] = current_p[j], current_p[i]
                # Recalc cost to be safe (or track delta, but perturb is multi-step)
                current_cost = get_cost(current_p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                    
        return best_p.tolist()
