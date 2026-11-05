"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # We aim to find a decomposition with rank < 30.
    # Known rank for 2x3x5 is around 19-20.
    # We will use a greedy peeling algorithm with rational factors.
    
    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_rank_to_try = 25 # Try to find a solution with rank <= 25
    
    # Define a set of candidate vectors with small rational entries.
    # We use a grid of values: 0, 0.5, 1.0, -0.5, -1.0
    # To keep the search space manageable, we will generate sparse candidates.
    
    def generate_candidates(dim, sparsity=2, max_val=1.0, step=0.5):
        """Generate sparse candidate vectors with rational entries."""
        candidates = []
        # Include the standard basis vectors
        for i in range(dim):
            vec = np.zeros(dim)
            vec[i] = 1.0
            candidates.append(vec)
        
        # Include some random sparse vectors with rational entries
        # We use a deterministic seed for reproducibility
        np.random.seed(42)
        num_candidates = 200
        for _ in range(num_candidates):
            vec = np.zeros(dim)
            # Choose 1 to sparsity indices
            num_nonzeros = np.random.randint(1, sparsity + 1)
            indices = np.random.choice(dim, num_nonzeros, replace=False)
            for idx in indices:
                # Choose a value from the rational grid
                val = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                vec[idx] = val
            candidates.append(vec)
        return candidates

    cand_U = generate_candidates(n*m, sparsity=3)
    cand_V = generate_candidates(m*p, sparsity=3)
    cand_W = generate_candidates(p*n, sparsity=3)

    rank = 0
    found_solution = False
    
    while rank < max_rank_to_try:
        best_score = -1e10
        best_u = None
        best_v = None
        best_w = None
        
        # Search for the best rank-1 approximation
        # To make this faster, we can iterate over a subset of candidates
        # or use a smarter selection. For now, let's try all combinations if the list is small enough.
        # 200^3 is 8,000,000 which might be too slow for 25 iterations.
        # Let's reduce the candidate pool or use a heuristic.
        
        # Heuristic: Pick the candidate u that maximizes the norm of the contraction
        # Then pick v, w that maximize the score.
        
        # Step 1: Find best u
        best_u_score = -1e10
        best_u_temp = None
        for u in cand_U:
            # Contract residual with u to get a matrix M_vw
            # M_vw[j, k] = sum_i residual[i, j, k] * u[i]
            M_vw = np.einsum('ijk,i->jk', residual, u)
            # The best rank-1 approximation to M_vw has score ||M_vw||_F^2 (if we could choose any v,w)
            # But we are restricted to cand_V and cand_W.
            # Let's just use the Frobenius norm as a proxy for u's quality
            score_u = np.linalg.norm(M_vw, 'fro')
            if score_u > best_u_score:
                best_u_score = score_u
                best_u_temp = u
        
        if best_u_temp is None:
            break
            
        # Step 2: Given best_u_temp, find best v, w
        M_vw = np.einsum('ijk,i->jk', residual, best_u_temp)
        
        # Now we want to find v in cand_V and w in cand_W such that
        # sum_{j,k} M_vw[j,k] * v[j] * w[k] is maximized.
        # This is equivalent to maximizing v^T M_vw w.
        
        # We can iterate over cand_V and for each v, find the best w.
        # w_best = argmax_w (M_vw^T v)^T w
        # Let x = M_vw^T v. Then we want to maximize x^T w.
        # This is a dot product. The best w from cand_W is the one with the largest dot product with x.
        
        best_total_score = -1e10
        best_v_final = None
        best_w_final = None
        
        for v in cand_V:
            x = M_vw.T.dot(v) # Shape (p*n,)
            # Find best w in cand_W
            # scores_w = [np.dot(x, w) for w in cand_W]
            # This is still O(N_V * N_W * Dim)
            # Let's optimize:
            # We can precompute nothing, but N_V, N_W are ~200. 200*200 = 40000 dot products of size 10.
            # This is fast.
            max_dot = -1e10
            best_w_temp = None
            for w in cand_W:
                dot = np.dot(x, w)
                if dot > max_dot:
                    max_dot = dot
                    best_w_temp = w
            
            if max_dot > best_total_score:
                best_total_score = max_dot
                best_v_final = v
                best_w_final = best_w_temp
        
        if best_total_score <= 1e-10:
            break
            
        # Subtract the rank-1 tensor
        residual -= np.einsum('i,j,k->ijk', best_u_temp, best_v_final, best_w_final)
        
        U_list.append(best_u_temp)
        V_list.append(best_v_final)
        W_list.append(best_w_final)
        rank += 1
        
        # Check if residual is zero
        if np.linalg.norm(residual) < 1e-10:
            found_solution = True
            break

    if not found_solution:
        # Fall back to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the factor matrices
    U_mat = np.array(U_list).T  # shape (n*m, rank)
    V_mat = np.array(V_list).T  # shape (m*p, rank)
    W_mat = np.array(W_list).T  # shape (p*n, rank)
    
    # Convert to list of lists
    # Ensure values are exact fractions if possible, or just float
    U_frac = [[float(x) for x in row] for row in U_mat]
    V_frac = [[float(x) for x in row] for row in V_mat]
    W_frac = [[float(x) for x in row] for row in W_mat]
    
    # Verify the decomposition
    U_arr = np.array(U_frac)
    V_arr = np.array(V_frac)
    W_arr = np.array(W_frac)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    
    # Check non-diagonal entries are zero
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 != i2 or j1 != j2 or k1 != k2:
                                val = np.sum(U_arr[i1*m+j1] * V_arr[j2*p+k2] * W_arr[k2*n+i2])
                                err = abs(val)
                                max_err = max(max_err, err)
    
    if max_err > 1e-6:
        # Fall back to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_frac, "V": V_frac, "W": W_frac}
