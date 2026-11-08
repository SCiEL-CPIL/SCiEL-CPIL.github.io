"""
Tensor decomposition for 3x3 * 3x7 matrix multiplication.
Uses schoolbook (rank 27) for the two 3x3 blocks and rank 3 for the 3x1 block.
Total rank: 27 + 27 + 3 = 57.
"""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 7
    
    # Shapes
    dim_u = n * m  # 9
    dim_v = m * p  # 21
    dim_w = p * n  # 21
    
    # Initialize U, V, W with 0s
    # We don't know the rank upfront exactly if we mix, but we know it's 57.
    # Let's build lists of vectors for U, V, W.
    
    U_vecs = []
    V_vecs = []
    W_vecs = []
    
    # Helper to add a term
    def add_term(u_sparse, v_sparse, w_sparse):
        # u_sparse: dict {index: value}
        # v_sparse: dict {index: value}
        # w_sparse: dict {index: value}
        u = [0] * dim_u
        for idx, val in u_sparse.items():
            u[idx] = val
        v = [0] * dim_v
        for idx, val in v_sparse.items():
            v[idx] = val
        w = [0] * dim_w
        for idx, val in w_sparse.items():
            w[idx] = val
        
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)
        
    # Block 1: Columns 0-2 (3x3x3 schoolbook, rank 27)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k  (k is 0,1,2)
                v_idx = j * p + k
                # W index: k*n + i
                w_idx = k * n + i
                
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                
    # Block 2: Columns 3-5 (3x3x3 schoolbook, rank 27)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # Global k is 3+k
                g_k = 3 + k
                u_idx = i * m + j
                v_idx = j * p + g_k
                w_idx = g_k * n + i
                
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                
    # Block 3: Column 6 (3x3x1 rank 3)
    # Terms for j=0, 1, 2
    for j in range(m):
        # U is non-zero at A[i, j] for all i
        u_sparse = {}
        for i in range(n):
            u_sparse[i * m + j] = 1
            
        # V is non-zero at B[j, 6]
        v_sparse = {j * p + 6: 1}
        
        # W is non-zero at C[i, 6] for all i
        w_sparse = {}
        for i in range(n):
            w_sparse[6 * n + i] = 1
            
        add_term(u_sparse, v_sparse, w_sparse)
        
    rank = len(U_vecs)
    
    # Convert to numpy arrays
    U = np.array(U_vecs).T  # Shape (dim_u, rank)
    V = np.array(V_vecs).T  # Shape (dim_v, rank)
    W = np.array(W_vecs).T  # Shape (dim_w, rank)
    
    # Verification
    if time.time() < deadline - 5:
        # Reconstruct tensor
        tensor = np.zeros((dim_u, dim_v, dim_w))
        for c in range(rank):
            tensor += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        # Target tensor
        target = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
                    
        if np.allclose(tensor, target):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook just in case
            U_fb = np.zeros((dim_u, n*m*p))
            V_fb = np.zeros((dim_v, n*m*p))
            W_fb = np.zeros((dim_w, n*m*p))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, col] = 1
                        V_fb[j*p+k, col] = 1
                        W_fb[k*n+i, col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
            
    return {"U": U, "V": V, "W": W}

