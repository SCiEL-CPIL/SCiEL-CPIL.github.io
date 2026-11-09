"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses structured block decomposition with ALS refinement.
    The key structural idea: exploit that C[i,k] = sum_j A[i,j]*B[j,k] can be
    seen as outer products of row-vectors of A and column-vectors of B.
    """
    start_time = time.time()
    deadline = 110  # Keep 10s safety margin before 120s limit
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    def check_solution(U, V, W, rank):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_structured_initialization(rank, seed):
        """
        Build initial U, V, W using the block structure.
        For each rank component r, we can think of it as:
          U_row_i[r] contributes to row i of A
          V_col_k[r] contributes to column k of B
        The structured form: U[i*m+j, r] = alpha[i, r] * beta[j, r]
                            V[j*p+k, r] = gamma[j, r] * delta[k, r]
                            W[k*n+i, r] = epsilon[k, r] * zeta[i, r]
        But this is overly constrained. Instead, use a simpler structured init:
        Initialize based on the schoolbook pattern but with random perturbations
        that respect the block structure.
        """
        rng = np.random.RandomState(seed)
        
        # Start from schoolbook-like structure but with randomization
        # For rank r, we want to "merge" some of the 60 schoolbook components
        # Strategy: group the (i,j,k) triples into rank groups
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # For each rank component, assign a subset of (i,j,k) triples
        # Use a structured assignment based on blocks
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    triples.append((i, j, k))
        
        # Shuffle and assign to rank components
        rng.shuffle(triples)
        
        for r in range(rank):
            # Each rank component handles roughly n*m*p/r triples
            start = r * (n*m*p) // rank
            end = (r+1) * (n*m*p) // rank
            if start >= len(triples):
                break
            subset = triples[start:end]
            
            for (i, j, k) in subset:
                # Add contribution to this rank component
                # Use structured values: U depends on (i,j), V on (j,k), W on (k,i)
                U[i*m+j, r] += 1.0
                V[j*p+k, r] += 1.0
                W[k*n+i, r] += 1.0
        
        # Add small random perturbation to break symmetry
        U += rng.randn(nm, rank) * 0.1
        V += rng.randn(mp, rank) * 0.1
        W += rng.randn(pn, rank) * 0.1
        
        return U, V, W

    def solve_als_structured(U, V, W, rank, max_iters=2000, time_budget=20.0):
        """
        ALS with structured updates. For each variable, solve a least squares
        problem. The key insight: when updating U[i*m+j, :], the system is:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        This is linear in U[i*m+j, :] given V and W.
        """
        reg = 1e-8
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 3:
                break
            
            # Update U: For each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    # A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A_u = np.zeros((p, rank))
                    for k in range(p):
                        v_idx = j*p + k
                        w_idx = k*n + i
                        A_u[k, :] = V[v_idx, :rank] * W[w_idx, :rank]
                    b_u = np.ones(p)
                    try:
                        U[u_idx, :rank] = np.linalg.lstsq(A_u, b_u, rcond=None)[0]
                    except:
                        pass
            
            # Update V: For each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    A_v = np.zeros((n, rank))
                    for i in range(n):
                        u_idx = i*m + j
                        w_idx = k*n + i
                        A_v[i, :] = U[u_idx, :rank] * W[w_idx, :rank]
                    b_v = np.ones(n)
                    try:
                        V[v_idx, :rank] = np.linalg.lstsq(A_v, b_v, rcond=None)[0]
                    except:
                        pass
            
            # Update W: For each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    A_w = np.zeros((m, rank))
                    for j in range(m):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        A_w[j, :] = U[u_idx, :rank] * V[v_idx, :rank]
                    b_w = np.ones(m)
                    try:
                        W[w_idx, :rank] = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
                    except:
                        pass
            
            # Check error periodically
            if it % 10 == 0:
                curr_err = check_solution(U, V, W, rank)
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-12:
                    break
        
        return best_err, best_U, best_V, best_W

    # Schoolbook baseline (Rank 60)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from low to high, using structured initialization
    # Theoretical considerations: for n×m × m×p, rank is at least max(n*m, m*p, p*n)
    # but can be lower. For 3×4×5, let's try a range.
    ranks_to_try = [15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50, 55]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline - 5:
            break
        
        found = False
        for seed in [42, 123, 999, 2024, 7777, 31415]:
            if time.time() - start_time > deadline - 5:
                break
            
            # Structured initialization
            U_init, V_init, W_init = build_structured_initialization(rank, seed)
            
            # ALS refinement
            time_left = deadline - (time.time() - start_time) - 3
            if time_left < 2:
                break
            
            err, U_c, V_c, W_c = solve_als_structured(
                U_init, V_init, W_init, rank, 
                max_iters=1000, 
                time_budget=min(15, time_left/2.0)
            )
            
            if err < 1e-10:
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        
        if found:
            break
    
    # Final verification
    if best_rank < rank_schoolbook:
        final_err = check_solution(best_U, best_V, best_W, best_rank)
        if final_err > 1e-8:
            # Fallback
            best_rank = rank_schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
