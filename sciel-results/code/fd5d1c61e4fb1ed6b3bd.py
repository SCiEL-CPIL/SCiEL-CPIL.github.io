import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    
    # Check for symmetry and zero diagonal to enable fast delta
    # QAPLIB instances are almost always symmetric with 0 diagonal
    # We will assume this for performance, as checking costs O(N^2) once
    is_symmetric = np.allclose(a, a.T) and np.allclose(b, b.T)
    zero_diag = np.allclose(np.diag(a), 0) and np.allclose(np.diag(b), 0)
    
    # Initialization: Rank matching
    # Facilities with high flow sums go to locations with low distance sums (central)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        # sum_{i,j} a[i,j] b[p[i], p[j]]
        # Efficient: np.einsum or matrix multiplication
        # b_p = b[np.ix_(p, p)]
        # cost = np.sum(a * b_p)
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing parameters
    start_time = time.time()
    timeout = 115.0 # Safety margin for 120s limit
    
    # Temperature schedule
    # Start high enough to accept bad moves, end low to polish
    t_max = 1000.0
    t_min = 0.01
    # Number of iterations estimate? 
    # We will just run until time limit or T < t_min
    # Geometric cooling
    # We need to estimate how many steps we can do.
    # Let's just use a fixed number of steps per temperature or time-based cooling.
    # Time-based: T = t_max * (t_min/t_max) ** (elapsed / total_time)
    # This is robust.
    
    # Pre-compute row sums? Not needed for the delta formula derived above.
    
    # Helper for delta
    def get_delta(i, j, p):
        if i == j:
            return 0
        u = p[i]
        v = p[j]
        if is_symmetric and zero_diag:
            # delta = 2 * sum_{k != i,j} (a[i,k] + a[j,k]) (b[v, p[k]] - b[u, p[k]])
            # sum_all = dot(a[i]+a[j], b[v,p] - b[u,p])
            # Subtract k=i and k=j terms
            vec_diff = b[v, p] - b[u, p]
            weights = a[i, :] + a[j, :]
            sum_all = np.dot(weights, vec_diff)
            
            # term for k=i: (a[i,i]+a[j,i]) * (b[v, p[i]] - b[u, p[i]])
            # p[i] is u
            term_i = (a[i, i] + a[j, i]) * (b[v, u] - b[u, u])
            # term for k=j: (a[i,j]+a[j,j]) * (b[v, p[j]] - b[u, p[j]])
            # p[j] is v
            term_j = (a[i, j] + a[j, j]) * (b[v, v] - b[u, v])
            
            delta = 2.0 * (sum_all - term_i - term_j)
            return delta
        else:
            # Fallback: slower exact calculation
            # This is rarely used if inputs are standard
            p_new = p.copy()
            p_new[i], p_new[j] = p_new[j], p_new[i]
            return calc_cost(p_new) - current_cost

    # Main SA Loop
    # We will use a fixed number of iterations per temperature step to allow
    # adaptive cooling based on time.
    
    # Estimate max iterations possible?
    # Let's just run until timeout.
    # We need to update T.
    # Let's assume total time T_total = 115s.
    # T(t) = t_max * (t_min / t_max) ** (t / T_total)
    
    t_current = t_max
    
    while time.time() - start_time < timeout:
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = get_delta(i, j, p)
        
        # Acceptance criterion
        if delta < 0 or np.random.rand() < np.exp(-delta / t_current):
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        # Update temperature
        # Simple geometric cooling per step is hard to tune without knowing step count.
        # Time-based cooling is better.
        elapsed = time.time() - start_time
        # Ensure we don't divide by zero or go out of bounds
        if elapsed >= timeout:
            break
        # T = t_max * (t_min/t_max)^(elapsed/timeout)
        # Using log for stability?
        # t_current = t_max * np.exp(np.log(t_min/t_max) * (elapsed / timeout))
        t_current = t_max * (t_min / t_max) ** (elapsed / timeout)
        
        if t_current < t_min:
             # We can break or continue with t_min
             # Continuing with t_min allows for final polishing
             pass
             
    return best_p.tolist()
