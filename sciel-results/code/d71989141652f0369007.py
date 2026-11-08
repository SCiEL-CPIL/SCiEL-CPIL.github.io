"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(val, tol=1e-6):
    """Check if val is close to a half-integer."""
    closest = np.round(val * 2) / 2
    return abs(val - closest) < tol

def _to_half_integer(val):
    """Round to nearest half-integer."""
    return np.round(val * 2) / 2

def _strassen_block_decomp(n, m, p):
    """
    Build a Strassen-based block decomposition for n x m * m x p.
    For n=2, m=4, p=4: split into 2x2 blocks of size 2x2.
    Each 2x2*2x2->2x2 uses Strassen (rank 7).
    4 block multiplications, each rank 7, total rank = 4*7 = 28.
    """
    block_size = 2
    num_blocks_row = n // block_size  # 1
    num_blocks_col = m // block_size  # 2
    num_blocks_out = p // block_size  # 2
    
    # Strassen matrices for 2x2*2x2->2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-f), M5 = (a+b)h, M6 = (c-d)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # Each Strassen term: U_factor * V_factor * W_factor
    # For block (bi, bj) -> output block (bi, bk):
    # A_block = A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # B_block = B[bj*2:(bj+1)*2, bk*2:(bk+1)*2]
    # C_block = C[bi*2:(bi+1)*2, bk*2:(bk+1)*2]
    
    # U index: (i, j) where i in [0,n), j in [0,m)
    # V index: (j, k) where j in [0,m), k in [0,p)
    # W index: (k, i) where k in [0,p), i in [0,n)
    
    # Strassen terms for 2x2:
    # Term 0: (a+d)(e+h) -> U: a=1,d=1 for (i,j); V: e=1,h=1 for (j,k); W: contributes to C11
    # Let's define the 7 Strassen factors:
    # For A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each M_t contributes to specific C entries with coefficients
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 += -1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 += -1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    # For each block (bi, bj, bk), we need 7 rank-1 terms.
    # Total blocks: num_blocks_row * num_blocks_col * num_blocks_out = 1*2*2 = 4
    # Total rank: 4 * 7 = 28
    
    total_blocks = num_blocks_row * num_blocks_col * num_blocks_out
    total_rank = total_blocks * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_idx = 0
    for bi in range(num_blocks_row):
        for bj in range(num_blocks_col):
            for bk in range(num_blocks_out):
                # Block A indices: rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
                # Block B indices: rows bj*2 to bj*2+1, cols bk*2 to bk*2+1
                # Block C indices: rows bi*2 to bi*2+1, cols bk*2 to bk*2+1
                
                # Local indices within block:
                # A: (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
                # B: (0,0)=B00, (0,1)=B01, (1,0)=B10, (1,1)=B11
                # C: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
                
                # Strassen factor definitions:
                # (local_A_rows, local_A_cols, local_B_rows, local_B_cols, coeff_A, coeff_B, W_contribs)
                # W_contribs: list of (local_C_row, local_C_col, coeff)
                
                strassen_factors = [
                    # M1 = (A00+A11)(B00+B11)
                    {
                        'A_terms': [((0,0), 1), ((1,1), 1)],
                        'B_terms': [((0,0), 1), ((1,1), 1)],
                        'W_terms': [((0,0), 1), ((1,1), 1)]
                    },
                    # M2 = (A10+A11)B00
                    {
                        'A_terms': [((1,0), 1), ((1,1), 1)],
                        'B_terms': [((0,0), 1)],
                        'W_terms': [((1,0), 1), ((1,1), -1)]
                    },
                    # M3 = A00(B01-B11)
                    {
                        'A_terms': [((0,0), 1)],
                        'B_terms': [((0,1), 1), ((1,1), -1)],
                        'W_terms': [((0,1), 1), ((1,1), 1)]
                    },
                    # M4 = A11(B10-B00)
                    {
                        'A_terms': [((1,1), 1)],
                        'B_terms': [((1,0), 1), ((0,0), -1)],
                        'W_terms': [((0,0), 1), ((1,0), 1)]
                    },
                    # M5 = (A00+A01)B11
                    {
                        'A_terms': [((0,0), 1), ((0,1), 1)],
                        'B_terms': [((1,1), 1)],
                        'W_terms': [((0,0), -1), ((0,1), 1)]
                    },
                    # M6 = (A10-A11)(B00+B01)
                    {
                        'A_terms': [((1,0), 1), ((1,1), -1)],
                        'B_terms': [((0,0), 1), ((0,1), 1)],
                        'W_terms': [((1,1), 1)]
                    },
                    # M7 = (A01-A11)(B10+B11)
                    {
                        'A_terms': [((0,1), 1), ((1,1), -1)],
                        'B_terms': [((1,0), 1), ((1,1), 1)],
                        'W_terms': [((0,0), 1)]
                    },
                ]
                
                for factor in strassen_factors:
                    # Set U entries
                    for (local_i, local_j), coeff_A in factor['A_terms']:
                        global_i = bi * block_size + local_i
                        global_j = bj * block_size + local_j
                        U[global_i * m + global_j][rank_idx] = coeff_A
                    
                    # Set V entries
                    for (local_j, local_k), coeff_B in factor['B_terms']:
                        global_j = bj * block_size + local_j
                        global_k = bk * block_size + local_k
                        V[global_j * p + global_k][rank_idx] = coeff_B
                    
                    # Set W entries
                    for (local_k, local_i), coeff_W in factor['W_terms']:
                        global_k = bk * block_size + local_k
                        global_i = bi * block_size + local_i
                        W[global_k * n + global_i][rank_idx] = coeff_W
                    
                    rank_idx += 1
    
    return {"U": U, "V": V, "W": W}

def _random_search_rank(n, m, p, rank, max_attempts=200, deadline=None):
    """Random search for a valid decomposition of given rank."""
    half_ints = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    probs = [0.2, 0.2, 0.3, 0.15, 0.15]
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(attempt * 7919 + rank * 104729 + 42)
        
        U_r = np.random.choice(half_ints, size=(n * m, rank), p=probs)
        V_r = np.random.choice(half_ints, size=(m * p, rank), p=probs)
        
        W_np = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U_r[i * m + j, r] * V_r[j * p + k, r]
                
                try:
                    W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-6:
                        valid = False
                        break
                    
                    for r in range(rank):
                        val = W_col[r]
                        if not _is_half_integer(val, tol=1e-5):
                            valid = False
                            break
                        W_np[k * n + i, r] = _to_half_integer(val)
                    if not valid:
                        break
                except np.linalg.LinAlgError:
                    valid = False
                    break
                if not valid:
                    break
            
            if not valid:
                break
        
        if valid:
            if _check_decomposition(U_r.tolist(), V_r.tolist(), W_np.tolist(), n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_np.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Try Strassen block decomposition first
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        strassen_result = _strassen_block_decomp(n, m, p)
        strassen_rank = len(strassen_result["U"][0])
        if _check_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
            best_result = strassen_result
            best_rank = strassen_rank
    
    # Search for lower rank decompositions
    # Start from best_rank - 1 down to a reasonable lower bound
    # Lower bound: for n x m * m x p, rank >= max(n*m, m*p, p*n) is not tight
    # A better lower bound involves the tensor rank, but we'll search down to 16
    
    for rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Number of random attempts for this rank
        max_attempts = 300 if rank > 25 else 500
        
        result = _random_search_rank(n, m, p, rank, max_attempts=max_attempts, deadline=deadline)
        
        if result is not None:
            best_result = result
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
