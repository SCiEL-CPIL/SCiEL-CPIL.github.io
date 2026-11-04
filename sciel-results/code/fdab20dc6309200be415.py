import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    Nm = n * m  # 20
    Mp = m * p  # 25
    Pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    # Build target indices for 4x5x5
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    target_set = set(target_indices)
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9, full_check=True):
        """Check if U, V, W decompose the target tensor exactly"""
        R = U_mat.shape[1]
        max_err = 0.0
        
        # Check all target positions (should be 1)
        for (a, b, c) in target_indices:
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
            if max_err > tol:
                return False, max_err
        
        # Check non-target positions (should be 0)
        if full_check:
            for a in range(Nm):
                for b in range(Mp):
                    for c in range(Pn):
                        if (a, b, c) in target_set:
                            continue
                        val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                        err = abs(val)
                        if err > max_err:
                            max_err = err
                        if max_err > tol:
                            return False, max_err
        else:
            np.random.seed(12345)
            for _ in range(1000):
                a = np.random.randint(Nm)
                b = np.random.randint(Mp)
                c = np.random.randint(Pn)
                if (a, b, c) in target_set:
                    continue
                val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
        
        return max_err < tol, max_err
    
    def add_term(U, V, W, col, a_set, b_set, c_set, u_val, v_val, w_val):
        """Add a rank-1 term with given values at specified indices"""
        for a in a_set:
            U[a, col] += u_val
        for b in b_set:
            V[b, col] += v_val
        for c in c_set:
            W[c, col] += w_val
    
    def schoolbook_block(U, V, W, col_offset, i1, i2, j1, j2, k1, k2):
        """Add schoolbook terms for block A[i1:i2, j1:j2] x B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]"""
        col = col_offset
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return col
    
    def strassen_block(U, V, W, col_offset, i1, i2, j1, j2, k1, k2):
        """Add Strassen terms for 2x2x2 block: A[i1:i2, j1:j2] x B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
        Requires i2-i1=2, j2-j1=2, k2-k1=2"""
        assert i2 - i1 == 2 and j2 - j1 == 2 and k2 - k1 == 2
        i0, i1b = i1, i1 + 1
        j0, j1b = j1, j1 + 1
        k0, k1b = k1, k1 + 1
        
        # A indices: a = i*m + j
        # B indices: b = j*p + k
        # C indices: c = k*n + i
        
        # Strassen 7 products for 2x2x2:
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        
        # C[i0,k0] = M1 + M4 - M5 + M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M6
        
        col = col_offset
        
        # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        a_indices = [i0*m + j0, i1b*m + j1b]
        b_indices = [j0*p + k0, j1b*p + k1b]
        c_indices = [k0*n + i0, k1b*n + i1b]  # contributes to both C[i0,k0] and C[i1,k1]
        U[a_indices[0], col] = 1.0
        U[a_indices[1], col] = 1.0
        V[b_indices[0], col] = 1.0
        V[b_indices[1], col] = 1.0
        W[c_indices[0], col] = 1.0
        W[c_indices[1], col] = 1.0
        col += 1
        
        # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        a_indices = [i1b*m + j0, i1b*m + j1b]
        b_indices = [j0*p + k0]
        c_indices = [k0*n + i1b]  # C[i1,k0]
        U[a_indices[0], col] = 1.0
        U[a_indices[1], col] = 1.0
        V[b_indices[0], col] = 1.0
        W[c_indices[0], col] = 1.0
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
        a_indices = [i0*m + j0]
        b_indices = [j0*p + k1b, j1b*p + k1b]
        c_indices = [k1b*n + i0]  # C[i0,k1]
        U[a_indices[0], col] = 1.0
        V[b_indices[0], col] = 1.0
        V[b_indices[1], col] = -1.0
        W[c_indices[0], col] = 1.0
        col += 1
        
        # M4: A[i1,j1] * (B[j1,k0]-B[j0,k0])
        a_indices = [i1b*m + j1b]
        b_indices = [j1b*p + k0, j0*p + k0]
        c_indices = [k0*n + i0, k0*n + i1b]  # C[i0,k0] and C[i1,k0]
        U[a_indices[0], col] = 1.0
        V[b_indices[0], col] = 1.0
        V[b_indices[1], col] = -1.0
        W[c_indices[0], col] = 1.0
        W[c_indices[1], col] = 1.0
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        a_indices = [i0*m + j0, i0*m + j1b]
        b_indices = [j1b*p + k1b]
        c_indices = [k0*n + i0, k1b*n + i0]  # -C[i0,k0] and +C[i0,k1]
        U[a_indices[0], col] = 1.0
        U[a_indices[1], col] = 1.0
        V[b_indices[0], col] = 1.0
        W[c_indices[0], col] = -1.0
        W[c_indices[1], col] = 1.0
        col += 1
        
        # M6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        a_indices = [i1b*m + j0, i0*m + j0]
        b_indices = [j0*p + k0, j0*p + k1b]
        c_indices = [k1b*n + i1b]  # C[i1,k1]
        U[a_indices[0], col] = 1.0
        U[a_indices[1], col] = -1.0
        V[b_indices[0], col] = 1.0
        V[b_indices[1], col] = 1.0
        W[c_indices[0], col] = 1.0
        col += 1
        
        # M7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        a_indices = [i0*m + j1b, i1b*m + j1b]
        b_indices = [j1b*p + k0, j1b*p + k1b]
        c_indices = [k0*n + i0]  # C[i0,k0]
        U[a_indices[0], col] = 1.0
        U[a_indices[1], col] = -1.0
        V[b_indices[0], col] = 1.0
        V[b_indices[1], col] = 1.0
        W[c_indices[0], col] = 1.0
        col += 1
        
        return col
    
    # Block decomposition: 4x5 x 5x5
    # Split: n=2+2, m=2+3, p=2+3
    # Blocks:
    #   (0,0): A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2], size 2x2x2, Strassen rank 7
    #   (0,1): A[0:2,0:2] x B[0:2,2:5] -> C[0:2,2:5], size 2x2x3, schoolbook rank 12
    #   (0,2): A[0:2,2:5] x B[2:5,0:2] -> C[0:2,0:2], size 2x3x2, schoolbook rank 12
    #   (0,3): A[0:2,2:5] x B[2:5,2:5] -> C[0:2,2:5], size 2x3x3, schoolbook rank 18
    #   (1,0): A[2:4,0:2] x B[0:2,0:2] -> C[2:4,0:2], size 2x2x2, Strassen rank 7
    #   (1,1): A[2:4,0:2] x B[0:2,2:5] -> C[2:4,2:5], size 2x2x3, schoolbook rank 12
    #   (1,2): A[2:4,2:5] x B[2:5,0:2] -> C[2:4,0:2], size 2x3x2, schoolbook rank 12
    #   (1,3): A[2:4,2:5] x B[2:5,2:5] -> C[2:4,2:5], size 2x3x3, schoolbook rank 18
    # Total: 7+12+12+18+7+12+12+18 = 98
    
    total_rank = 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18  # 98
    U = np.zeros((Nm, total_rank))
    V = np.zeros((Mp, total_rank))
    W = np.zeros((Pn, total_rank))
    
    col = 0
    # Block (0,0): Strassen 2x2x2
    col = strassen_block(U, V, W, col, 0, 2, 0, 2, 0, 2)
    # Block (0,1): schoolbook 2x2x3
    col = schoolbook_block(U, V, W, col, 0, 2, 0, 2, 2, 5)
    # Block (0,2): schoolbook 2x3x2
    col = schoolbook_block(U, V, W, col, 0, 2, 2, 5, 0, 2)
    # Block (0,3): schoolbook 2x3x3
    col = schoolbook_block(U, V, W, col, 0, 2, 2, 5, 2, 5)
    # Block (1,0): Strassen 2x2x2
    col = strassen_block(U, V, W, col, 2, 4, 0, 2, 0, 2)
    # Block (1,1): schoolbook 2x2x3
    col = schoolbook_block(U, V, W, col, 2, 4, 0, 2, 2, 5)
    # Block (1,2): schoolbook 2x3x2
    col = schoolbook_block(U, V, W, col, 2, 4, 2, 5, 0, 2)
    # Block (1,3): schoolbook 2x3x3
    col = schoolbook_block(U, V, W, col, 2, 4, 2, 5, 2, 5)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify the block decomposition
    is_valid, err = verify(U, V, W, full_check=True)
    
    if is_valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if block decomposition fails
    rank_sb = n * m * p  # 100
    U_sb = np.zeros((Nm, rank_sb))
    V_sb = np.zeros((Mp, rank_sb))
    W_sb = np.zeros((Pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
