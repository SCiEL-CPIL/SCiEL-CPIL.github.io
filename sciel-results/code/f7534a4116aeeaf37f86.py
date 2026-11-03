import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    def calc_cost(p):
        # Sum a[i, j] * b[p[i], p[j]]
        # Vectorized
        bp = b_int[p[:, None], p[None, :]]
        return np.sum(a_int * bp)
    
    current_cost = calc_cost(p)
    
    # Simulated Annealing
    T = 1000.0 # Initial temperature
    cooling = 0.99
    min_T = 1.0
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    while T > min_T and time.time() < deadline:
        # Randomize order of pairs to avoid bias
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        np.random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            
            # Calculate delta for swapping u and v
            # u' = p[u], v' = p[v]
            u_loc = p[u]
            v_loc = p[v]
            
            # D_old is implicitly b[p[:, None], p[None, :]]
            # We need D_old[u, :], D_old[:, u], D_old[v, :], D_old[:, v]
            
            # D_old[u, k] = b[u_loc, p[k]]
            # D_old[k, u] = b[p[k], u_loc]
            
            # T1 = sum_k a[u, k] * (D_new[u, k] - D_old[u, k])
            # D_new[u, k] = D_old[v, k] for k != u, v
            # D_new[u, u] = D_old[v, v]
            # D_new[u, v] = D_old[v, u]
            
            # Let's compute the sums directly
            # Vector of D_old[v, :]
            D_v_row = b_int[v_loc, p]
            D_u_row = b_int[u_loc, p]
            D_v_col = b_int[p, v_loc]
            D_u_col = b_int[p, u_loc]
            
            # a_u = a[u, :]
            a_u = a_int[u]
            a_v = a_int[v]
            a_uT = a_int[:, u]
            a_vT = a_int[:, v]
            
            # T1
            # sum_k a_u[k] * (D_new[u, k] - D_old[u, k])
            # D_new[u, k] is D_v_row[k] except at k=u, v
            # D_old[u, k] is D_u_row[k]
            
            # Base sum: sum_k a_u[k] * (D_v_row[k] - D_u_row[k])
            base1 = np.dot(a_u, D_v_row - D_u_row)
            # Corrections for k=u, v
            # k=u: D_new[u, u] = D_old[v, v] = D_v_row[v] (since p[v]=v_loc? No, p[v] is index, b[v_loc, v_loc])
            # D_v_row[v] = b[v_loc, p[v]] = b[v_loc, v_loc]. Correct.
            # D_old[u, u] = D_u_row[u] = b[u_loc, p[u]] = b[u_loc, u_loc].
            # Term: a_u[u] * (D_v_row[v] - D_u_row[u])
            # But base1 used D_v_row[u] - D_u_row[u].
            # Diff: a_u[u] * (D_v_row[v] - D_v_row[u])
            
            # k=v: D_new[u, v] = D_old[v, u] = D_v_col[u] = b[p[u], v_loc] = b[u_loc, v_loc].
            # D_v_row[v] = b[v_loc, v_loc].
            # Base1 used D_v_row[v] - D_u_row[v].
            # D_old[u, v] = D_u_row[v] = b[u_loc, p[v]] = b[u_loc, v_loc].
            # So base1 term for v was a_u[v] * (D_v_row[v] - D_u_row[v]).
            # Actual term: a_u[v] * (D_new[u, v] - D_old[u, v]) = a_u[v] * (D_v_col[u] - D_u_row[v]).
            # Diff: a_u[v] * (D_v_col[u] - D_v_row[v])
            
            T1 = base1 + a_u[u] * (D_v_row[v] - D_v_row[u]) + a_u[v] * (D_v_col[u] - D_v_row[v])
            
            # T2
            # sum_k a_uT[k] * (D_new[k, u] - D_old[k, u])
            # D_new[k, u] = D_old[k, v] for k != u, v
            # D_new[u, u] = D_old[v, v]
            # D_new[v, u] = D_old[u, v]
            
            base2 = np.dot(a_uT, D_v_col - D_u_col)
            # k=u: D_new[u, u] = D_v_row[v] (which is D_old[v, v]). D_u_col[u] = D_old[u, u].
            # Base2 used D_v_col[u] - D_u_col[u]. D_v_col[u] = D_old[u, v].
            # Diff: a_uT[u] * (D_v_row[v] - D_v_col[u])
            
            # k=v: D_new[v, u] = D_old[u, v] = D_u_col[v].
            # Base2 used D_v_col[v] - D_u_col[v]. D_v_col[v] = D_old[v, v].
            # Diff: a_uT[v] * (D_u_col[v] - D_v_col[v])
            
            T2 = base2 + a_uT[u] * (D_v_row[v] - D_v_col[u]) + a_uT[v] * (D_u_col[v] - D_v_col[v])
            
            # T3
            # sum_k a_v[k] * (D_new[v, k] - D_old[v, k])
            # D_new[v, k] = D_old[u, k] for k != u, v
            # D_new[v, v] = D_old[u, u]
            # D_new[v, u] = D_old[u, v]
            
            base3 = np.dot(a_v, D_u_row - D_v_row)
            # k=v: D_new[v, v] = D_u_row[u]. Base3 used D_u_row[v] - D_v_row[v].
            # Diff: a_v[v] * (D_u_row[u] - D_u_row[v])
            
            # k=u: D_new[v, u] = D_u_col[v]. Base3 used D_u_row[u] - D_v_row[u].
            # Diff: a_v[u] * (D_u_col[v] - D_u_row[u])
            
            T3 = base3 + a_v[v] * (D_u_row[u] - D_u_row[v]) + a_v[u] * (D_u_col[v] - D_u_row[u])
            
            # T4
            # sum_k a_vT[k] * (D_new[k, v] - D_old[k, v])
            # D_new[k, v] = D_old[k, u] for k != u, v
            # D_new[v, v] = D_old[u, u]
            # D_new[u, v] = D_old[u, v]
            
            base4 = np.dot(a_vT, D_u_col - D_v_col)
            # k=v: D_new[v, v] = D_u_col[u]. Base4 used D_u_col[v] - D_v_col[v].
            # Diff: a_vT[v] * (D_u_col[u] - D_u_col[v])
            
            # k=u: D_new[u, v] = D_u_col[v]. Base4 used D_u_col[u] - D_v_col[u].
            # Diff: a_vT[u] * (D_u_col[v] - D_u_col[u])
            
            T4 = base4 + a_vT[v] * (D_u_col[u] - D_u_col[v]) + a_vT[u] * (D_u_col[v] - D_u_col[u])
            
            delta = T1 + T2 + T3 + T4
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept
                p[u], p[v] = p[v], p[u]
                current_cost += delta
        
        T *= cooling

    return p.tolist()
