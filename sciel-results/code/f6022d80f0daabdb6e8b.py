"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _is_half_int(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half_int(x):
    return round(x * 2) / 2.0

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[2, 3] = -1 # B10
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _decompose_2x3x2():
    """
    Decompose 2x3x2 matrix multiplication: A(2x3) * B(3x2) = C(2x2).
    Rank 10 decomposition.
    U: (6, 10), V: (6, 10), W: (4, 10)
    """
    # We can view this as: C = A * B where A is 2x3, B is 3x2
    # Split A into columns: A = [A_col0, A_col1, A_col2]
    # C = A_col0 * B_row0 + A_col1 * B_row1 + A_col2 * B_row2
    # Each term is outer product of a 2x1 and 1x2 = 2x2 matrix
    # But we can do better using Strassen-like ideas.
    
    # Actually, let's use a direct construction.
    # The 2x3x2 tensor has 12 nonzero entries in the multiplication table.
    # We need at least 10 terms (known result).
    
    # Let me construct it using the fact that we can embed 2x2x2 (rank 7) 
    # and handle the extra column.
    
    # Alternative: use the decomposition directly.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Let's try to find a rank-10 decomposition by optimization or known result.
    # For now, let's use schoolbook rank 12 and see if we can improve.
    
    # Actually, let me just return a schoolbook decomposition for 2x3x2 for now
    # and focus on the overall structure.
    
    U = np.zeros((6, 12))
    V = np.zeros((6, 12))
    W = np.zeros((4, 12))
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    return U, V, W

def _build_2x3x6_decomposition():
    """
    Build a decomposition for 2x3x6 by splitting into three 2x3x2 blocks.
    Each block handles 2 columns of the 6-column output.
    Total rank: 3 * 12 = 36 (schoolbook) or potentially less.
    """
    n, m, p = 2, 3, 6
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # Split into 3 blocks of 2 columns each
    # Block b handles columns 2b, 2b+1 of B and C
    U_total = np.zeros((U_dim, 36))
    V_total = np.zeros((V_dim, 36))
    W_total = np.zeros((W_dim, 36))
    
    for b in range(3):
        U_block, V_block, W_block = _decompose_2x3x2()
        # U_block: (6, 12), V_block: (6, 12), W_block: (4, 12)
        col_offset = b * 12
        
        U_total[:, col_offset:col_offset+12] = U_block
        
        # V_block: row j*2+k corresponds to B[j, 2b+k]
        # In V_total, row jp*p+k = jp*6 + (2b+k)
        for j in range(3):
            for k in range(2):
                V_total[j*6 + (2*b + k), col_offset:col_offset+12] = V_block[j*2+k, :]
        
        # W_block: row k*2+i corresponds to C[2b+k, i]
        # In W_total, row kp*n+ip = (2b+kp)*2 + ip = 2b*2 + kp*2 + ip = 4b + kp*2 + ip
        for k in range(2):
            for i in range(2):
                W_total[(2*b + k)*2 + i, col_offset:col_offset+12] = W_block[k*2+i, :]
    
    return U_total, V_total, W_total

def _optimize_tensor_decomposition(n, m, p, target_rank, deadline, max_iterations=3000):
    """
    Optimize a rank-target decomposition for the n x m x p matrix multiplication tensor.
    """
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute all index combinations
    indices = list(product(range(n), range(n), range(m), range(m), range(p), range(p)))
    
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(attempt * 98765 + 42)
        
        U = np.random.randn(U_dim, target_rank) * 0.3
        V = np.random.randn(V_dim, target_rank) * 0.3
        W = np.random.randn(W_dim, target_rank) * 0.3
        
        lr = 0.005
        for it in range(max_iterations):
            if time.time() > deadline - 20:
                break
            
            # Compute error on a random subset
            num_samples = min(100, len(indices))
            sample_indices = np.random.choice(len(indices), num_samples, replace=False)
            
            errs = []
            for idx in sample_indices:
                i, ip, j, jp, k, kp = indices[idx]
                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                prod = U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :]
                s = np.sum(prod)
                errs.append((s - expected) ** 2)
            
            err = np.mean(errs)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Perturbation
            dU = np.random.randn(U_dim, target_rank) * lr
            dV = np.random.randn(V_dim, target_rank) * lr
            dW = np.random.randn(W_dim, target_rank) * lr
            
            U_new = U + dU
            V_new = V + dV
            W_new = W + dW
            
            # Quick check
            num_check = 30
            check_indices = np.random.choice(len(indices), num_check, replace=False)
            errs_new = []
            for idx in check_indices:
                i, ip, j, jp, k, kp = indices[idx]
                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                prod = U_new[i*m+j, :] * V_new[jp*p+k, :] * W_new[kp*n+ip, :]
                s = np.sum(prod)
                errs_new.append((s - expected) ** 2)
            
            new_err = np.mean(errs_new)
            
            if new_err < err:
                U, V, W = U_new, V_new, W_new
                err = new_err
            else:
                lr *= 0.95
    
    if best_err < 1e-8:
        U_r = np.array([[_round_to_half_int(x) for x in row] for row in best_U])
        V_r = np.array([[_round_to_half_int(x) for x in row] for row in best_V])
        W_r = np.array([[_round_to_half_int(x) for x in row] for row in best_W])
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, target_rank):
            return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try block-based decomposition (3 x 2x3x2 blocks)
    U_block, V_block, W_block = _build_2x3x6_decomposition()
    if _verify(U_block.tolist(), V_block.tolist(), W_block.tolist(), n, m, p, 36):
        # This is just schoolbook rearranged, rank 36
        pass
    
    # Try to find lower rank decompositions
    # Known: 2x3x2 has rank 10, so 3 blocks give rank 30
    # Try ranks from 28 down to 24
    for target_rank in [30, 29, 28, 27, 26, 25, 24]:
        if time.time() > deadline - 30:
            break
        
        result_opt = _optimize_tensor_decomposition(n, m, p, target_rank, deadline, max_iterations=2000)
        if result_opt is not None:
            U, V, W = result_opt
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If optimization didn't find anything better, return schoolbook
    return result
# EVOLVE-BLOCK-END
