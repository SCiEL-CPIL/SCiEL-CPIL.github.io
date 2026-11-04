import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Verify: for each (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] == W[k*n+i, t]
    # Actually, the tensor decomposition is:
    # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # But for matrix multiplication, T[i,j,k] = delta_{i,j} * delta_{j,k} is not right.
    # The multiplication tensor is T[i,j,k] = 1 if i,j,k form a valid product, else 0.
    # Specifically, (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k (it's a rank-1 tensor in the sense that
    # the multiplication tensor has entries T[i,j,k] = 1 if we're computing (AB)[i,k] using A[i,j]*B[j,k])
    # Actually, the standard convention is that the multiplication tensor M_{i,j,k} = 1 if i,j,k are valid indices
    # for the product, i.e., M is a 3D tensor where M[i,j,k] = 1.
    # The decomposition is: M[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    # Let's verify: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal 1 if this is a valid product term
    # Actually, for matrix multiplication, every (i,j,k) triple is "valid" in the sense that
    # A[i,j] * B[j,k] contributes to C[i,k]. So the tensor has all 1s? No, that's not right.
    
    # The multiplication tensor for (n x m) * (m x p) -> (n x p) is defined as:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition satisfies: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    # So we need: for all i,j,k, sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 7
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    # x_sub, y_sub are 2x2 dicts, c_sub is 2x2 dict for output
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # For 4x4x4, the c_block for each outer term m_idx
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            # Compute U coefficients
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            # Compute V coefficients
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            # Compute W coefficients
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    # 2x2x3: Strassen 2x2x2 (7 terms) + 4 schoolbook terms for col 3
    
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        # Strassen 2x2x2 for columns 4,5 (local cols 0,1)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj  # global cols 4,5
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj  # global cols 4,5
                W[k_final * n + i_final, r] += cc
        
        # Schoolbook for column 6 (local col 2)
        # C[i][6] = A[i][0]*B[0][6] + A[i][1]*B[1][6]
        # Local A: rows 2ar, 2ar+1; cols 2ac, 2ac+1
        # Local B: rows 2br, 2br+1; col 6
        # Local C: rows 2cr, 2cr+1; col 6
        
        # Term 1: A[0,0] * B[0,2] -> C[0,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        # Term 2: A[0,1] * B[1,2] -> C[0,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        # Term 3: A[1,0] * B[0,2] -> C[1,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        # Term 4: A[1,1] * B[1,2] -> C[1,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    # The four 2x2x3 blocks:
    # A_00 (ar=0, ac=0) * B_top (br=0) -> C_top (cr=0)
    add_2x2x3(0, 0, 0, 0)
    # A_01 (ar=0, ac=1) * B_bot (br=1) -> C_top (cr=0)
    add_2x2x3(0, 1, 1, 0)
    # A_10 (ar=1, ac=0) * B_top (br=0) -> C_bot (cr=1)
    add_2x2x3(1, 0, 0, 1)
    # A_11 (ar=1, ac=1) * B_bot (br=1) -> C_bot (cr=1)
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
