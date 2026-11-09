import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen's 2x2x2 decomposition
    # U, V, W are vectors of size 4
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 1, -1, 0]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0]
    ]
    
    # Block 1: M[4,4,4] using Kronecker product of Strassen
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # A is 4x4, B is 4x4, C is 4x4
    # We map the 4x4 matrices to 2x2 blocks of 2x2.
    # A[i,j] where i=2*i1+i2, j=2*j1+j2
    # The Kronecker product of tensors T1 x T2 corresponds to 
    # U = U1 x U2, V = V1 x V2, W = W1 x W2
    # We need to expand these into the full 20-dim (4*5) space.
    
    for t in range(7):
        for s in range(7):
            u_ks = np.kron(np.array(strassen_U[t]), np.array(strassen_U[s]))
            v_ks = np.kron(np.array(strassen_V[t]), np.array(strassen_V[s]))
            w_ks = np.kron(np.array(strassen_W[t]), np.array(strassen_W[s]))
            
            # Expand to full dimensions
            u_full = np.zeros(n * m)
            v_full = np.zeros(m * p)
            w_full = np.zeros(p * n)
            
            # u_ks is size 4 (2*2). It corresponds to A[0:4, 0:4]
            # A is 4x5. Indices 0..15 correspond to A[0:4, 0:4]
            # A[i,j] index is i*5 + j.
            # u_ks index x corresponds to A[i2, j2] in the 2x2 block?
            # Kronecker product of [a,b] and [c,d] is [ac, ad, bc, bd].
            # This matches the row-major order of the 4x4 matrix if we view it as 2x2 blocks.
            # A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 in 2x2.
            # In 4x4 (row major):
            # 0,0 -> 0
            # 0,1 -> 1
            # 1,0 -> 2
            # 1,1 -> 3
            # So u_ks[0] is coeff of A[0,0], u_ks[1] of A[0,1], etc.
            # These map to global indices:
            # A[0,0] -> 0*5+0 = 0
            # A[0,1] -> 0*5+1 = 1
            # A[1,0] -> 1*5+0 = 5
            # A[1,1] -> 1*5+1 = 6
            # Wait, Kronecker product order:
            # np.kron([a,b], [c,d]) = [ac, ad, bc, bd]
            # This corresponds to:
            # (i1=0, i2=0) -> a*c -> A[0,0]
            # (i1=0, i2=1) -> a*d -> A[0,1]
            # (i1=1, i2=0) -> b*c -> A[1,0]
            # (i1=1, i2=1) -> b*d -> A[1,1]
            # So u_ks[0] -> A[0,0]
            # u_ks[1] -> A[0,1]
            # u_ks[2] -> A[1,0]
            # u_ks[3] -> A[1,1]
            
            # Map to global indices
            # A[i,j] index = i*m + j
            # i = 2*i1 + i2, j = 2*j1 + j2
            # We need to map the 4 values of u_ks to the correct global indices.
            # u_ks[0] (i1=0, i2=0, j1=0, j2=0) -> i=0, j=0 -> idx 0
            # u_ks[1] (i1=0, i2=0, j1=0, j2=1) -> i=0, j=1 -> idx 1
            # u_ks[2] (i1=0, i2=1, j1=0, j2=0) -> i=1, j=0 -> idx 5
            # u_ks[3] (i1=0, i2=1, j1=0, j2=1) -> i=1, j=1 -> idx 6
            # Wait, np.kron([a,b], [c,d]) is [ac, ad, bc, bd].
            # This is (i1=0, j1=0) with (i2=0, j2=0) -> ac
            # (i1=0, j1=0) with (i2=0, j2=1) -> ad
            # (i1=0, j1=0) with (i2=1, j2=0) -> bc
            # (i1=0, j1=0) with (i2=1, j2=1) -> bd
            # This assumes the first vector is the "outer" index.
            # In our 2x2 Strassen vectors, the index is i*2+j.
            # So u[0]=A00, u[1]=A01, u[2]=A10, u[3]=A11.
            # Kronecker product of u and v:
            # u[0]v[0], u[0]v[1], u[1]v[0], u[1]v[1]
            # This corresponds to:
            # (A00 * B00), (A00 * B01), (A10 * B00), (A10 * B01)
            # This is NOT the standard Kronecker product of matrices.
            # Standard Kronecker A x B has entries A[i,j] B[k,l] at position (ik+j, jl).
            # Here we are multiplying tensors.
            # The tensor product of T1 and T2 has rank 1 terms (u1 x u2) (v1 x v2) (w1 x w2).
            # The index of u1 x u2 is (i1, i2) x (j1, j2) -> (i1*2+i2, j1*2+j2).
            # This matches the row-major order of the 4x4 matrix.
            # So u_ks[k] corresponds to the coefficient of A[i,j] where k = i*4+j?
            # No, k is the index in the 4-dim vector.
            # The 4-dim vector represents the 2x2 block.
            # The Kronecker product of two 4-dim vectors is 16-dim.
            # But we only have 4x4 = 16 entries in A[0:4, 0:4].
            # So u_ks should be expanded to 16 entries?
            # No, u_ks is the result of np.kron of two 4-vectors, so it is 16-dim.
            # My previous code: u_ks = np.kron(...). This is 16-dim.
            # I need to map these 16 values to the 20-dim global U vector.
            # The 16 values correspond to A[0:4, 0:4].
            # Global index for A[i,j] is i*5 + j.
            # The 16-dim vector u_ks has index l.
            # l = i*4 + j.
            # We need to map l to i*5 + j.
            
            for l in range(16):
                i = l // 4
                j = l % 4
                global_idx = i * m + j
                u_full[global_idx] = u_ks[l]
                
            # Similarly for V (4x4 part of 5x5)
            for l in range(16):
                j = l // 4
                k = l % 4
                global_idx = j * p + k
                v_full[global_idx] = v_ks[l]
                
            # Similarly for W (4x4 part of 5x4)
            # W is p x n = 5 x 4.
            # Index is k*n + i.
            # The 16-dim vector w_ks has index l = i*4 + k?
            # In Strassen W, index is i*2 + k.
            # Kronecker product: (i1*2+i2) * (k1*2+k2) -> (i1*4+i2*2+i? No)
            # The index in the 16-dim vector is l = i*4 + k.
            # We need to map l to k*n + i.
            for l in range(16):
                i = l // 4
                k = l % 4
                global_idx = k * n + i
                w_full[global_idx] = w_ks[l]
                
            U_cols.append(u_full)
            V_cols.append(v_full)
            W_cols.append(w_full)
            
    # Block 2: M[4,4,1]
    # A[0:4, 0:4] * B[0:4, 4:5] -> C[0:4, 4:5]
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(1):
                a_idx = i * m + j
                b_idx = j * p + (4 + k)
                c_idx = (4 + k) * n + i
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[a_idx] = 1.0
                v[b_idx] = 1.0
                w[c_idx] = 1.0
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    # Block 3: M[4,1,4]
    # A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for j in range(1):
            for k in range(4):
                a_idx = i * m + (4 + j)
                b_idx = (4 + j) * p + k
                c_idx = k * n + i
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[a_idx] = 1.0
                v[b_idx] = 1.0
                w[c_idx] = 1.0
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    # Block 4: M[4,1,1]
    # A[0:4, 4:5] * B[4:5, 4:5] -> C[0:4, 4:5]
    for i in range(4):
        for j in range(1):
            for k in range(1):
                a_idx = i * m + (4 + j)
                b_idx = (4 + j) * p + (4 + k)
                c_idx = (4 + k) * n + i
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[a_idx] = 1.0
                v[b_idx] = 1.0
                w[c_idx] = 1.0
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
