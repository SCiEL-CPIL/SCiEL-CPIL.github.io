"""Optimized program for 2x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor: T[u,v,w] = 1 where u=i*m+j, v=j*p+k, w=k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition using einsum."""
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)
    
    def try_peeling(max_rank, seed=42, max_iter=50):
        """Peeling approach: iteratively find best rank-1 approximation to residual."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        residual = T_target.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for rank_idx in range(max_rank):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random vectors
            u = rng.randn(nm)
            v = rng.randn(mp)
            w = rng.randn(pn)
            
            # Alternating maximization to find best rank-1 approx
            for it in range(max_iter):
                S_u = np.einsum('ijk,j,k->i', residual, v, w)
                u_new = S_u / (np.linalg.norm(S_u) + 1e-15)
                
                S_v = np.einsum('ijk,i,k->j', residual, u_new, w)
                v_new = S_v / (np.linalg.norm(S_v) + 1e-15)
                
                S_w = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                w_new = S_w / (np.linalg.norm(S_w) + 1e-15)
                
                if np.linalg.norm(u_new - u) + np.linalg.norm(v_new - v) + np.linalg.norm(w_new - w) < 1e-10:
                    break
                u, v, w = u_new, v_new, w_new
            
            # Compute coefficient
            alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
            if abs(alpha) < 1e-12:
                break
            
            residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
            U_list.append(alpha * u)
            V_list.append(v)
            W_list.append(w)
            
            if np.linalg.norm(residual) < 1e-10:
                break
        
        R = len(U_list)
        if R > max_rank:
            return None
        
        # Verify
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        if verify(U, V, W, tol=1e-6):
            return (U, V, W, R)
        return None
    
    def try_structured_search(max_rank, time_budget=20.0):
        """Try structured approaches to find rank-max_rank decomposition."""
        if time.time() > deadline - 5:
            return None
        
        start_search = time.time()
        
        # Try peeling with multiple seeds
        for seed in range(50):
            if time.time() > deadline - 5 or time.time() - start_search > time_budget:
                break
            
            result = try_peeling(max_rank, seed=seed + 42, max_iter=40)
            if result is not None:
                U, V, W, R = result
                if R <= max_rank:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
        
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Time budget per rank attempt
        time_budget = min(15.0, max(5.0, (deadline - time.time()) - 10) / max(1, best_rank - rank))
        
        result = try_structured_search(rank, time_budget=time_budget)
        if result is not None:
            best_result = result
            best_rank = result["rank"]
            if best_rank == lower_bound:
                break
    
    return best_result
# EVOLVE-BLOCK-END
