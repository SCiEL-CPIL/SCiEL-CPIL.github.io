"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    p_arr = np.asarray(p)
    total = 0
    for i in range(len(a)):
        total += np.dot(a[i], b[p_arr[i]])
    return total

def _compute_delta(a, b, p, i, j):
    """Compute the change in cost if we swap p[i] and p[j]."""
    n = len(p)
    u = p[i]
    v = p[j]
    
    # Terms involving i and j
    # delta = sum_{k != i,j} a[i,k](b[v,p[k]] - b[u,p[k]]) + a[j,k](b[u,p[k]] - b[v,p[k]])
    #         + a[k,i](b[p[k],v] - b[p[k],u]) + a[k,j](b[p[k],u] - b[p[k],v])
    #         + a[i,j](b[v,u] - b[u,v]) + a[j,i](b[u,v] - b[v,u])
    #         + a[i,i](b[v,v] - b[u,u]) + a[j,j](b[u,u] - b[v,v])
    
    # Vectorized computation for k != i, j
    # Create mask for k != i, j
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    pks = p[ks]
    
    # part 1: sum_{k in ks} a[i,k] * (b[v, p[k]] - b[u, p[k]])
    # part 2: sum_{k in ks} a[j,k] * (b[u, p[k]] - b[v, p[k]])
    # part 3: sum_{k in ks} a[k,i] * (b[p[k], v] - b[p[k], u])
    # part 4: sum_{k in ks} a[k,j] * (b[p[k], u] - b[p[k], v])
    
    # Note: b is symmetric usually, but we handle general case.
    # b[v, pks] is vector of shape (len(ks),)
    b_v_pks = b[v, pks]
    b_u_pks = b[u, pks]
    b_pks_v = b[pks, v]
    b_pks_u = b[pks, u]
    
    ai_k = a[i, ks]
    aj_k = a[j, ks]
    ak_i = a[ks, i]
    ak_j = a[ks, j]
    
    delta_vec = ai_k * (b_v_pks - b_u_pks) + aj_k * (b_u_pks - b_v_pks) + \
                ak_i * (b_pks_v - b_pks_u) + ak_j * (b_pks_u - b_pks_v)
    
    delta = np.sum(delta_vec)
    
    # Add terms for i,j pairs and self terms
    delta += a[i, j] * (b[v, u] - b[u, v])
    delta += a[j, i] * (b[u, v] - b[v, u])
    delta += a[i, i] * (b[v, v] - b[u, u])
    delta += a[j, j] * (b[u, u] - b[v, v])
    
    return delta

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Initial solution: Random
    p = list(range(n))
    random.shuffle(p)
    current_cost = _qap_cost(a, b, p)
    
    best_p = p[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1000.0 # Initial temperature
    T_min = 1.0
    cooling_rate = 0.9995
    
    # We need to estimate the scale of costs to set T properly
    # Let's just run a few random swaps to estimate average delta magnitude
    avg_delta = 0
    for _ in range(100):
        i, j = random.sample(range(n), 2)
        d = _compute_delta(a, b, p, i, j)
        avg_delta += abs(d)
    avg_delta /= 100
    if avg_delta == 0:
        avg_delta = 1.0
    T = 10 * avg_delta
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick random swap
        i, j = random.sample(range(n), 2)
        
        delta = _compute_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p[:]
        else:
            # Accept with probability exp(-delta / T)
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost = new_cost
        
        T *= cooling_rate
        
    return best_p
# EVOLVE-BLOCK-END
