import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0
    
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_sums, kind='stable')
    central = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    T = 5000.0
    cooling = 0.9999
    
    batch_size = 10000
    
    while time.time() < deadline:
        # Generate batch of moves
        # 2 * batch_size random indices
        moves = np.random.randint(n, size=(2, batch_size))
        
        for k in range(batch_size):
            i = moves[0, k]
            j = moves[1, k]
            if i == j:
                continue
                
            u = p[i]
            v = p[j]
            
            V1 = a[i] - a[j]
            V2 = b[v] - b[u]
            
            dot_prod = np.dot(V1, V2[p])
            term_i = V1[i] * V2[u]
            term_j = V1[j] * V2[v]
            
            delta = dot_prod - term_i - term_j
            
            if delta < 0:
                accept = True
            else:
                exponent = -delta / T
                if exponent > 0:
                    accept = True
                else:
                    accept = np.random.rand() < np.exp(exponent)
            
            if accept:
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= cooling
            if T < 1e-4:
                T = 100.0

    return best_p.tolist()
