"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin before 120s kill
    
    # Build the list of (a,b,c) triples where the tensor T=1
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    num_triples = len(triples)  # n*m*p = 30
    dim_U = n * m  # 6
    dim_V = m * p  # 15
    dim_W = p * n  # 10
    
    # Precompute the structure: for each triple index t, we have (a_t, b_t, c_t)
    a_arr = np.array([t[0] for t in triples])
    b_arr = np.array([t[1] for t in triples])
    c_arr = np.array([t[2] for t in triples])
    
    # Schoolbook fallback (rank 30)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(dim_U)]
    V_school = [[0] * rank_school for _ in range(dim_V)]
    W_school = [[0] * rank_school for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify_and_round(U, V, W, r, tol=1e-9):
        """Verify the decomposition and round to nearest half-integer."""
        # First verify with floating point
        for (a, b, c) in triples:
            val = 0.0
            for s in range(r):
                val += U[a][s] * V[b][s] * W[c][s]
            if abs(val - 1.0) > tol:
                return None
        
        # Round to nearest half-integer
        U_rounded = [[float(round(x * 2) / 2) for x in row] for row in U]
        V_rounded = [[float(round(x * 2) / 2) for x in row] for row in V]
        W_rounded = [[float(round(x * 2) / 2) for x in row] for row in W]
        
        # Verify rounded version
        for (a, b, c) in triples:
            val = 0.0
            for s in range(r):
                val += U_rounded[a][s] * V_rounded[b][s] * W_rounded[c][s]
            if abs(val - 1.0) > 1e-6:
                return None
        
        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    def try_rank(r, max_iter=100, num_restarts=10):
        """Try to find a rank-r decomposition using alternating least squares."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            U = np.random.randn(dim_U, r) * 0.5
            V = np.random.randn(dim_V, r) * 0.5
            W = np.random.randn(dim_W, r) * 0.5
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U: solve for U[a,s] given V and W
                # For each triple t: sum_s U[a_t, s] * V[b_t, s] * W[c_t, s] = 1
                # This is a linear system in the variables U[*, :]
                A_U = np.zeros((num_triples, dim_U * r))
                for t in range(num_triples):
                    a_t, b_t, c_t = triples[t]
                    for s in range(r):
                        A_U[t, a_t * r + s] = V[b_t, s] * W[c_t, s]
                
                try:
                    sol_U, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_triples), rcond=None)
                    U = sol_U.reshape(dim_U, r)
                except:
                    break
                
                # Update V
                A_V = np.zeros((num_triples, dim_V * r))
                for t in range(num_triples):
                    a_t, b_t, c_t = triples[t]
                    for s in range(r):
                        A_V[t, b_t * r + s] = U[a_t, s] * W[c_t, s]
                
                try:
                    sol_V, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_triples), rcond=None)
                    V = sol_V.reshape(dim_V, r)
                except:
                    break
                
                # Update W
                A_W = np.zeros((num_triples, dim_W * r))
                for t in range(num_triples):
                    a_t, b_t, c_t = triples[t]
                    for s in range(r):
                        A_W[t, c_t * r + s] = U[a_t, s] * V[b_t, s]
                
                try:
                    sol_W, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_triples), rcond=None)
                    W = sol_W.reshape(dim_W, r)
                except:
                    break
                
                # Check convergence
                err = 0.0
                for (a, b, c) in triples:
                    val = 0.0
                    for s in range(r):
                        val += U[a, s] * V[b, s] * W[c, s]
                    err += (val - 1.0) ** 2
                
                if err < 1e-12:
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    result = verify_and_round(U_list, V_list, W_list, r)
                    if result is not None:
                        return result
            
            # Check best solution from this restart
            err = 0.0
            for (a, b, c) in triples:
                val = 0.0
                for s in range(r):
                    val += U[a, s] * V[b, s] * W[c, s]
                err += (val - 1.0) ** 2
            
            if err < 1e-8:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                result = verify_and_round(U_list, V_list, W_list, r)
                if result is not None:
                    return result
        
        return None
    
    # Try ranks from small to large
    # Known lower bounds suggest rank might be less than 30
    # For 2x3*3x5, try ranks from 1 up to 29
    for r in range(1, rank_school):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
