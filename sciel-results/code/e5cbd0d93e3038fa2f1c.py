import time
import numpy as np

def construct():
    n, m, p = 2, 3, 9
    
    # 1. Construct the 50-rank Strassen-based decomposition
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Strassen 2x2x2 terms
    # u, v, w are dicts mapping (i,j) or (j,k) or (k,i) to coeff
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    # Part 1: j in {0,1}, k in {0..8}
    # 4 pairs of k: (0,1), (2,3), (4,5), (6,7)
    for k1, k2 in [(0,1), (2,3), (4,5), (6,7)]:
        for u_d, v_d, w_d in strassen_terms:
            U_t = {}
            V_t = {}
            W_t = {}
            for (i, j), c in u_d.items():
                U_t[(i, j)] = c
            for (j, k_loc), c in v_d.items():
                V_t[(j, k1 + k_loc)] = c
            for (i, k_loc), c in w_d.items():
                W_t[(i, k1 + k_loc)] = c
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
            
    # 1 single k: 8
    for i in range(2):
        for j in range(2):
            U_terms.append({(i, j): 1})
            V_terms.append({(j, 8): 1})
            W_terms.append({(i, 8): 1})
            
    # Part 2: j in {2}, k in {0..8}
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
            
    rank = len(U_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), c in U_terms[col].items():
            U[i * m + j][col] = c
        for (j, k), c in V_terms[col].items():
            V[j * p + k][col] = c
        for (i, k), c in W_terms[col].items():
            W[k * n + i][col] = c
            
    # 2. Try to reduce rank by checking linear dependencies
    # Flatten the tensor terms into vectors
    # Vector size: n*m * m*p * p*n
    vec_dim = n * m * m * p * p * n
    tensor_vecs = []
    for col in range(rank):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for (i, j), c in U_terms[col].items():
            u_vec[i * m + j] = c
        for (j, k), c in V_terms[col].items():
            v_vec[j * p + k] = c
        for (i, k), c in W_terms[col].items():
            w_vec[k * n + i] = c
            
        # Kronecker product u v w
        # This is expensive for 50 vectors of dim 2916?
        # 2916 * 50 = 145800 entries.
        # np.kron is fine.
        vec = np.kron(np.kron(np.array(u_vec), np.array(v_vec)), np.array(w_vec))
        tensor_vecs.append(vec)
        
    # Stack into matrix
    M = np.array(tensor_vecs).T # Shape (vec_dim, rank)
    
    # Check rank
    try:
        _, s, vh = np.linalg.svd(M, full_matrices=False)
        # Find number of non-zero singular values
        tol = 1e-10
        r = np.sum(s > tol)
        
        if r < rank:
            # We can eliminate (rank - r) terms
            # The null space of M (or rather, the dependencies)
            # We need to find which columns are dependent.
            # SVD gives us the basis.
            # If r < rank, the last (rank-r) columns in vh are null space vectors.
            # Actually, we want to find a subset of columns to remove such that the remaining are independent.
            # A simple heuristic: remove columns corresponding to the smallest singular values?
            # No, that's not how it works for column dependencies.
            
            # Better: use QR with column pivoting to find a basis.
            # Or just iterate: try removing each column and check if rank drops.
            # This is O(rank^2 * SVD cost).
            
            # Since rank is small (50), we can try removing each column i and checking if the rank of the remaining 49 is still 49.
            # If removing i drops the rank to 48, then i is redundant? No, if removing i keeps rank 49, then i was redundant?
            # If the total rank is r, and we have rank columns.
            # If we remove a column and the rank of the remaining matrix is still r, then the removed column was in the span of the others.
            
            removed = []
            current_rank = rank
            for i in range(rank):
                if i in removed:
                    continue
                # Try removing i
                mask = [j for j in range(rank) if j != i and j not in removed]
                if len(mask) == 0:
                    break
                M_sub = M[:, mask]
                # Check rank of M_sub
                # If rank(M_sub) == r, then column i is redundant.
                s_sub = np.linalg.svd(M_sub, full_matrices=False)[1]
                r_sub = np.sum(s_sub > tol)
                
                if r_sub == r:
                    removed.append(i)
                    
            # Reconstruct U, V, W without removed columns
            if removed:
                keep = [i for i in range(rank) if i not in removed]
                new_rank = len(keep)
                
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                for new_col, old_col in enumerate(keep):
                    for (i, j), c in U_terms[old_col].items():
                        new_U[i * m + j][new_col] = c
                    for (j, k), c in V_terms[old_col].items():
                        new_V[j * p + k][new_col] = c
                    for (i, k), c in W_terms[old_col].items():
                        new_W[k * n + i][new_col] = c
                        
                return {"U": new_U, "V": new_V, "W": new_W}
                
    except Exception:
        pass
        
    return {"U": U, "V": V, "W": W}
