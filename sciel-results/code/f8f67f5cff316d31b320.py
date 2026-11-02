import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def build_strassen_solution():
        R = 46
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        
        def add_strassen_block(k0, col_offset):
            prods = [
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1}, {(0,0):1}),
                ({(0,0):1}, {(0,1):1,(1,1):-1}),
                ({(1,1):1}, {(1,0):1,(0,0):-1}),
                ({(0,0):1,(0,1):1}, {(1,1):1}),
                ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
                ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
            ]
            for t, (a_coeff, b_coeff) in enumerate(prods):
                col = col_offset + t
                for (i, j), c in a_coeff.items():
                    U[A_idx(i, j), col] += c
                for (j, kl), c in b_coeff.items():
                    V[B_idx(j, k0 + kl), col] += c
            c_combos = [
                (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
                (0, k0+1,   [(2,1),(4,1)]),
                (1, k0,     [(1,1),(3,1)]),
                (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
            ]
            for (i, k, terms) in c_combos:
                for (t, coeff) in terms:
                    W[C_idx(i, k), col_offset + t] += coeff
        
        def add_single_block(k, col_offset):
            prods = [
                ((0,0),(0,k)),
                ((0,1),(1,k)),
                ((1,0),(0,k)),
                ((1,1),(1,k)),
            ]
            for t, ((i,j),(jj,kk)) in enumerate(prods):
                col = col_offset + t
                U[A_idx(i, j), col] += 1
                V[B_idx(jj, kk), col] += 1
            W[C_idx(0, k), col_offset+0] += 1
            W[C_idx(0, k), col_offset+1] += 1
            W[C_idx(1, k), col_offset+2] += 1
            W[C_idx(1, k), col_offset+3] += 1
        
        col = 0
        for k0 in range(0, 12, 2):
            add_strassen_block(k0, col)
            col += 7
        add_single_block(12, col)
        col += 4
        assert col == R
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i * m + j, j * p + k, k * n + i))
        target_arr = np.array(target_indices)
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        U_t = U[a_idx, :]
        V_t = V[b_idx, :]
        W_t = W[c_idx, :]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < tol)
    
    # Start with the known rank-46 solution
    U, V, W = build_strassen_solution()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 46
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to reduce rank by finding linear dependencies among the 46 rank-1 tensors
    if time.time() < deadline - 30:
        # Flatten each rank-1 tensor u_r ⊗ v_r ⊗ w_r into a vector of length nm*mp*pn
        # and check if the 46 vectors are linearly independent.
        num_tensors = 46
        flat_dim = nm * mp * pn  # 4 * 26 * 26 = 2704
        
        tensor_vecs = np.zeros((flat_dim, num_tensors), dtype=float)
        for r in range(num_tensors):
            outer_vw = np.outer(V[:, r], W[:, r])  # mp x pn
            full = np.outer(U[:, r], outer_vw)  # nm x mp x pn
            tensor_vecs[:, r] = full.flatten()
        
        # Check rank of the 2704 x 46 matrix
        rank_of_tensors = np.linalg.matrix_rank(tensor_vecs, tol=1e-10)
        
        if rank_of_tensors < num_tensors:
            # There is a linear dependence. Find the null space vector.
            _, S, Vt = np.linalg.svd(tensor_vecs, full_matrices=True)
            # The last row(s) of Vt correspond to null space directions
            null_vec = Vt[-1, :]  # one null space vector (for rank deficiency 1)
            
            # Find the index with the largest absolute value in null_vec
            # and try to eliminate that column
            elim_idx = np.argmax(np.abs(null_vec))
            
            # If we can express tensor[elim_idx] as a combination of others,
            # we can drop it. But we need to verify the remaining 45 still
            # span the same tensor.
            
            # Create a new solution without column elim_idx
            keep = [i for i in range(num_tensors) if i != elim_idx]
            new_U = U[:, keep]
            new_V = V[:, keep]
            new_W = W[:, keep]
            
            if verify(new_U, new_V, new_W, tol=1e-8):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = 45
                print(f"Found rank-45 solution by eliminating column {elim_idx}")
    
    # Snap to half-integers for clean output
    U = np.round(best_U * 2) / 2
    V = np.round(best_V * 2) / 2
    W = np.round(best_W * 2) / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
