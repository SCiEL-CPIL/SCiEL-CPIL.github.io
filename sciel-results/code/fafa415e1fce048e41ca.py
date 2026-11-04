import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # We aim for rank 28 using Strassen for 2x2 blocks + schoolbook for the rest
    # 2x3 * 3x5
    # Split into:
    #   A_left (2x2) * B_left (2x2) -> C_left (2x2)  [Strassen, rank 7]
    #   A_left (2x2) * B_mid  (2x2) -> C_mid  (2x2)  [Strassen, rank 7]
    #   A_left (2x2) * B_right(2x1) -> C_right(2x1)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_left (2x2) -> C_left2(2x2)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_mid  (2x2) -> C_mid2 (2x2)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_right(2x1) -> C_right2(2x1) [Schoolbook, rank 2]
    # Total: 7+7+4+4+4+2 = 28

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        # C00 += M1, C11 += M1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        # C10 += M2, C11 -= M2
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        # C01 += M3, C11 += M3
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        # C00 += M4, C10 += M4
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        # C00 -= M5, C01 += M5
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        # C11 += M6
        W[k2*n+i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        # C00 += M7
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)

    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)

    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [0, 1], [4])

    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [0, 1])

    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [2, 3])

    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5] (Schoolbook, rank 2)
    add_schoolbook([0, 1], [2], [4])

    if col != rank:
        # Should not happen, but if it does, we have a bug.
        # We can't easily fix it on the fly without knowing what went wrong.
        # Let's just proceed with verification.
        pass

    # Verify exactness using vectorized NumPy operations for speed
    # T_{ijk} = sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # For each r, we want to add u[:,:,r] * v[:,:,r] * w[:,:,r] to tensor
    # We can do this with einsum or broadcasting
    # tensor[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Efficient way:
    # 1. Compute product of U and V for each r: (n, m, rank) * (m, p, rank) -> not direct
    # Let's use the property that this is a tensor contraction.
    # Actually, the most efficient simple way for small dims is:
    # Iterate over rank, but use vectorized ops inside.
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # T_{ijk} += u[i,j] * v[j,k] * w[k,i]
        # We can construct this as:
        # temp[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # This is not a standard matmul.
        # Let's stick to the loop but it's only 28 * 2 * 3 * 5 = 840 operations, very fast.
        # Or we can do:
        # term = np.einsum('ij,jk,ki->ijk', u, v, w)
        tensor += np.einsum('ij,jk,ki->ijk', u, v, w)

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback: full schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
