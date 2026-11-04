import time
import numpy as np
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    # Tensor T[i, j, k] = 1 if i, j, k are valid indices for C[i, k] = sum_j A[i, j] B[j, k]
    # In the problem statement, the output factor is W[k*n+i, r].
    # The tensor is M[i, j, k] where i is row of A, j is col of A / row of B, k is col of B.
    # The decomposition is M = sum_r U[:, r] V[:, r] W[:, r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The entry M[i, j, k] corresponds to U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We want M[i, j, k] = 1 for all i, j, k.

    def verify(U, V, W, tol=1e-9):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = U.shape[1]
        # Construct the tensor M from the decomposition
        # M[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can do this efficiently with broadcasting or loops.
        # Since n, m, p are small (5), a loop is fine, but vectorized is better.
        
        M = np.zeros((n, m, p))
        for r in range(rank):
            u_vec = U[:, r]  # (n*m,)
            v_vec = V[:, r]  # (m*p,)
            w_vec = W[:, r]  # (p*n,)
            
            # Reshape u_vec to (n, m)
            u_mat = u_vec.reshape(n, m)
            # Reshape v_vec to (m, p)
            v_mat = v_vec.reshape(m, p)
            # Reshape w_vec to (p, n)
            w_mat = w_vec.reshape(p, n)
            
            # M[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This is not a simple matrix multiplication.
            # Let's just use the loop for clarity and speed given small size.
            # Actually, we can precompute the outer products.
            pass
        
        # Faster verification:
        # For each i, j, k, check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        # We can vectorize over r.
        # Create indices
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # U_vals shape: (n, m, p, rank)
        U_vals = U[u_idx, :]
        V_vals = V[v_idx, :]
        W_vals = W[w_idx, :]
        
        # Sum over rank
        total = np.sum(U_vals * V_vals * W_vals, axis=-1)
        
        return np.all(np.abs(total - 1.0) < tol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_block_decomposition(partition):
        """
        partition: list of (start, end) for rows, cols, and k-dim.
        We use the same partition for all 3 dimensions for simplicity in this helper.
        """
        # This function is a placeholder. We will implement a more direct one below.
        pass

    def add_strassen_2x2x2(terms, r1, c1, k1):
        """
        terms: list of (U_dict, V_dict, W_dict)
        """
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        
        # Strassen formulas for 2x2 matrices
        # M1 = (A00 + A11) (B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00) (B00 + B01)
        # M7 = (A01 - A11) (B10 + B11)
        
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 - M6 + M7
        
        # Term 1: M1
        terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1}))
        # Term 2: M2
        terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1}))
        # Term 3: M3
        terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1}))
        # Term 4: M4
        terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1}))
        # Term 5: M5
        terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1}))
        # Term 6: M6
        terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1}))
        # Term 7: M7
        terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1}))

    def build_block_decomp(partition):
        terms = []
        # partition is a list of (start, end)
        for r1, r2 in partition:
            for c1, c2 in partition:
                for k1, k2 in partition:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    if ar == 2 and ac == 2 and bc == 2:
                        add_strassen_2x2x2(terms, r1, c1, k1)
                    else:
                        for ii in range(r1, r2):
                            for jj in range(c1, c2):
                                for kk in range(k1, k2):
                                    u_idx = ii * m + jj
                                    v_idx = jj * p + kk
                                    w_idx = kk * n + ii
                                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx, r] = val
            for idx, val in V_dict.items():
                V[idx, r] = val
            for idx, val in W_dict.items():
                W[idx, r] = val
        
        return U, V, W, terms

    # Try different block partitions
    partitions = [
        [(0, 2), (2, 4), (4, 5)],  # [2, 2, 1]
        [(0, 4), (4, 5)],          # [4, 1]
        [(0, 3), (3, 5)],          # [3, 2]
        [(0, 1), (1, 3), (3, 5)],  # [1, 2, 2]
        [(0, 2), (2, 5)],          # [2, 3]
        [(0, 5)],                  # [5] (schoolbook)
    ]
    
    best_U, best_V, best_W = get_schoolbook()
    best_rank = best_U.shape[1]
    
    for partition in partitions:
        if time.time() > deadline - 20.0:
            break
        U, V, W, terms = build_block_decomp(partition)
        rank = U.shape[1]
        if verify(U, V, W):
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # Local search: Greedy merging
    if best_rank < n * m * p and time.time() < deadline - 30.0:
        # Convert best_U, best_V, best_W to terms list
        terms = []
        for r in range(best_rank):
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for idx in range(n * m):
                if best_U[idx, r] != 0:
                    U_dict[idx] = best_U[idx, r]
            for idx in range(m * p):
                if best_V[idx, r] != 0:
                    V_dict[idx] = best_V[idx, r]
            for idx in range(p * n):
                if best_W[idx, r] != 0:
                    W_dict[idx] = best_W[idx, r]
            terms.append((U_dict, V_dict, W_dict))
        
        def can_merge(term1, term2):
            """
            Check if term1 + term2 is rank-1.
            term1, term2: (U_dict, V_dict, W_dict)
            """
            # Construct the sum tensor T
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Here i is in U_dict keys, etc.
            # Since the terms are sparse, we can just iterate over the non-zero entries.
            
            T = {}
            for u_idx, u_val in term1[0].items():
                for v_idx, v_val in term1[1].items():
                    for w_idx, w_val in term1[2].items():
                        key = (u_idx, v_idx, w_idx)
                        T[key] = T.get(key, 0) + u_val * v_val * w_val
            
            for u_idx, u_val in term2[0].items():
                for v_idx, v_val in term2[1].items():
                    for w_idx, w_val in term2[2].items():
                        key = (u_idx, v_idx, w_idx)
                        T[key] = T.get(key, 0) + u_val * v_val * w_val
            
            # Check if T is rank-1
            # A tensor T is rank-1 if T[i, j, k] = a_i b_j c_k
            # This implies that for any i1, i2, j1, j2, k1, k2:
            # T[i1, j1, k1] T[i2, j2, k2] = T[i1, j2, k1] T[i2, j1, k2]
            # And similar for other pairs.
            
            u_indices = sorted(set(k[0] for k in T.keys()))
            v_indices = sorted(set(k[1] for k in T.keys()))
            w_indices = sorted(set(k[2] for k in T.keys()))
            
            if not u_indices or not v_indices or not w_indices:
                return True
            
            # Pick a reference element
            ref_key = None
            for key, val in T.items():
                if abs(val) > 1e-9:
                    ref_key = key
                    break
            
            if ref_key is None:
                return True
            
            # Check 2x2 minors
            for i1 in u_indices:
                for i2 in u_indices:
                    if i1 >= i2: continue
                    for j1 in v_indices:
                        for j2 in v_indices:
                            if j1 >= j2: continue
                            for k1 in w_indices:
                                for k2 in w_indices:
                                    if k1 >= k2: continue
                                    
                                    a = T.get((i1, j1, k1), 0)
                                    b = T.get((i1, j2, k1), 0)
                                    c = T.get((i2, j1, k1), 0)
                                    d = T.get((i2, j2, k1), 0)
                                    if abs(a * d - b * c) > 1e-9:
                                        return False
                                    
                                    a = T.get((i1, j1, k1), 0)
                                    b = T.get((i1, j1, k2), 0)
                                    c = T.get((i2, j1, k1), 0)
                                    d = T.get((i2, j1, k2), 0)
                                    if abs(a * d - b * c) > 1e-9:
                                        return False
                                    
                                    a = T.get((i1, j1, k1), 0)
                                    b = T.get((i1, j1, k2), 0)
                                    c = T.get((i1, j2, k1), 0)
                                    d = T.get((i1, j2, k2), 0)
                                    if abs(a * d - b * c) > 1e-9:
                                        return False
            return True

        # Greedy merge loop
        merged = True
        while merged and time.time() < deadline - 10.0:
            merged = False
            new_terms = list(terms)
            for i in range(len(new_terms)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, len(new_terms)):
                    if time.time() > deadline - 10.0:
                        break
                    if can_merge(new_terms[i], new_terms[j]):
                        # Merge
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for idx, val in new_terms[i][0].items():
                            U_dict[idx] = U_dict.get(idx, 0) + val
                        for idx, val in new_terms[j][0].items():
                            U_dict[idx] = U_dict.get(idx, 0) + val
                        for idx, val in new_terms[i][1].items():
                            V_dict[idx] = V_dict.get(idx, 0) + val
                        for idx, val in new_terms[j][1].items():
                            V_dict[idx] = V_dict.get(idx, 0) + val
                        for idx, val in new_terms[i][2].items():
                            W_dict[idx] = W_dict.get(idx, 0) + val
                        for idx, val in new_terms[j][2].items():
                            W_dict[idx] = W_dict.get(idx, 0) + val
                        
                        # Remove zeros
                        U_dict = {k: v for k, v in U_dict.items() if abs(v) > 1e-12}
                        V_dict = {k: v for k, v in V_dict.items() if abs(v) > 1e-12}
                        W_dict = {k: v for k, v in W_dict.items() if abs(v) > 1e-12}
                        
                        new_terms = [t for idx, t in enumerate(new_terms) if idx != i and idx != j]
                        new_terms.append((U_dict, V_dict, W_dict))
                        terms = new_terms
                        merged = True
                        break
                if merged:
                    break
        
        # Rebuild matrices
        rank = len(terms)
        if rank < best_rank:
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for r in range(rank):
                U_dict, V_dict, W_dict = terms[r]
                for idx, val in U_dict.items():
                    U[idx, r] = val
                for idx, val in V_dict.items():
                    V[idx, r] = val
                for idx, val in W_dict.items():
                    W[idx, r] = val
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # Convert numpy arrays to lists of lists as required by the interface
    # The interface expects {'U': U, 'V': V, 'W': W} with shapes (n*m, rank), etc.
    # And entries are integer/half-integer.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
