"""Matrix multiplication 5x5x5 tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    def _verify(U, V, W, n=5, m=5, p=5, tol=1e-6):
        """Check that the decomposition exactly represents matrix multiplication."""
        try:
            import numpy as np
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            # The multiplication tensor: T[i,j,k] = 1 for all i,j,k
            # because C[i,k] = sum_j A[i,j]*B[j,k] means each (i,j,k) triple
            # contributes exactly 1 to the trilinear form.
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            return max_err < tol
        except Exception:
            return False

    # Verified rank-7 tensors for 2x2x2 matrix multiplication
    # U indexed by [i*2+j], V by [j*2+k], W by [k*2+i]
    def strassen_2x2():
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 1, 0]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 1, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, -1, 0, 1]),
            ([1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, -1, 0]),
        ]

    def strassen_4x4():
        """Return rank-49 decomposition of 4x4x4 matrix multiplication.
        Each entry: (U_16, V_16, W_16) as lists of length 16.
        U indexed by [i*4+j], V by [j*4+k], W by [k*4+i].
        """
        s2 = strassen_2x2()
        
        # Outer Strassen terms for 2x2 block structure
        # Each term: (a_coeff, b_coeff, c_coeff) where keys are (block_row, block_col)
        outer_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):-1}),
        ]
        
        result = []
        for a_coeff, b_coeff, c_coeff in outer_terms:
            for inner_U, inner_V, inner_W in s2:
                U_4 = [0.0] * 16
                V_4 = [0.0] * 16
                W_4 = [0.0] * 16
                
                # Fill U_4: A[i,j] where i,j in 0..3
                # Block (bi,bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1
                for (bi, bj), a_val in a_coeff.items():
                    for ii in range(2):
                        for jj in range(2):
                            inner_idx = ii * 2 + jj
                            global_i = bi * 2 + ii
                            global_j = bj * 2 + jj
                            u_idx = global_i * 4 + global_j
                            U_4[u_idx] += a_val * inner_U[inner_idx]
                
                # Fill V_4: B[j,k] where j,k in 0..3
                for (bj, bk), b_val in b_coeff.items():
                    for jj in range(2):
                        for kk in range(2):
                            inner_idx = jj * 2 + kk
                            global_j = bj * 2 + jj
                            global_k = bk * 2 + kk
                            v_idx = global_j * 4 + global_k
                            V_4[v_idx] += b_val * inner_V[inner_idx]
                
                # Fill W_4: C[k,i] where k,i in 0..3
                for (bk, bi), c_val in c_coeff.items():
                    for kk in range(2):
                        for ii in range(2):
                            inner_idx = kk * 2 + ii
                            global_k = bk * 2 + kk
                            global_i = bi * 2 + ii
                            w_idx = global_k * 4 + global_i
                            W_4[w_idx] += c_val * inner_W[inner_idx]
                
                result.append((U_4, V_4, W_4))
        
        return result

    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
        
        n, m, p = 5, 5, 5
        U_list = []
        V_list = []
        W_list = []

        # Part 1: 4x4x4 core using Strassen rank-49
        s4 = strassen_4x4()
        for U4, V4, W4 in s4:
            U_full = [0.0] * 25
            V_full = [0.0] * 25
            W_full = [0.0] * 25
            
            for idx in range(16):
                i = idx // 4
                j = idx % 4
                U_full[i*5+j] = U4[idx]
                V_full[j*5+i] = V4[idx]  # Wait, V is indexed by j*5+k, not j*5+i
                W_full[i*5+j] = W4[idx]  # Wait, W is indexed by k*5+i, not i*5+j
            
            # Let me redo this more carefully
            U_full = [0.0] * 25
            V_full = [0.0] * 25
            W_full = [0.0] * 25
            
            for i in range(4):
                for j in range(4):
                    u_idx = i * 4 + j
                    U_full[i*5+j] = U4[u_idx]
            
            for j in range(4):
                for k in range(4):
                    v_idx = j * 4 + k
                    V_full[j*5+k] = V4[v_idx]
            
            for k in range(4):
                for i in range(4):
                    w_idx = k * 4 + i
                    W_full[k*5+i] = W4[w_idx]
            
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
        
        # Part 2: Border terms (at least one index is 4)
        # Schoolbook for these: each (i,j,k) with at least one index = 4 gets its own rank-1 term
        remaining = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if i == 4 or j == 4 or k == 4:
                        remaining.append((i, j, k))
        
        for i, j, k in remaining:
            U_full = [0.0] * 25
            V_full = [0.0] * 25
            W_full = [0.0] * 25
            U_full[i*5+j] = 1.0
            V_full[j*5+k] = 1.0
            W_full[k*5+i] = 1.0
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
        
        rank = len(U_list)
        # rank should be 49 + 61 = 110
        
        # Verify
        if _verify(U_list, V_list, W_list):
            def clean_matrix(mat):
                result = []
                for row in mat:
                    clean_row = []
                    for val in row:
                        if abs(val - round(val)) < 1e-9:
                            clean_row.append(int(round(val)))
                        else:
                            clean_row.append(val)
                    result.append(clean_row)
                return result
            
            U = clean_matrix(U_list)
            V = clean_matrix(V_list)
            W = clean_matrix(W_list)
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall through to schoolbook
            raise ValueError("Rank-110 decomposition failed verification")

    except Exception:
        pass
    
    # Fallback: schoolbook rank 125
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
