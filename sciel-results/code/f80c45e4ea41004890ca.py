"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()
_DEADLINE = 130  # seconds, safety margin before 150s kill


def _verify_fast(U_arr, V_arr, W_arr, n, m, p):
    """Fast verification assuming numpy arrays already."""
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < 1e-7


def _compute_error(U, V, W, n, m, p):
    """Compute max absolute error of the decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))


def _build_strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4x4 matrix multiplication using recursive Strassen.
    Returns U (16x49), V (16x49), W (16x49) as numpy arrays.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Recursive Strassen for 4x4x4
    # Level 1: 7 terms for 2x2 block structure
    # Level 2: 7 terms for each 2x2x2 sub-problem
    # Total: 49 terms
    
    # The 7 Strassen terms for 2x2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # For 4x4, we have 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Axy is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bxy is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cxy is 2x2
    
    # High-level Strassen terms:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # Each Mx is a 2x2x2 product of 2x2 matrices
    # We decompose each using the 7-term Strassen
    
    # Define the 7 Strassen patterns for 2x2x2
    # Each pattern: (A_coeffs, B_coeffs, C_coeffs) as dicts {(i,j): coeff}
    strassen_2x2 = [
        # M0: A=a00+a11, B=b00+b11, C: C00+=1, C11-=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # M1: A=a10+a11, B=b00, C: C10+=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M2: A=a00, B=b01-b11, C: C01+=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M3: A=a11, B=b10-b00, C: C00+=1, C10+=1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M4: A=a00+a01, B=b11, C: C01+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        # M5: A=a10-a00, B=b00+b01, C: C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M6: A=a01-a11, B=b10+b11, C: C00+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # High-level term definitions: (a_blocks, b_blocks) where each is list of (block_idx, coeff)
    high_level_terms = [
        # M0 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M1 = (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)]),
        # M2 = A00(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)]),
        # M3 = A11(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)]),
        # M4 = (A00+A01)B11
        ([(0, 1), (1, 1)], [(3, 1)]),
        # M5 = (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M6 = (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    # C block formulas: which high-level terms contribute to which C blocks
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    c_formulas = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00
        [(2, 1), (4, 1)],                    # C01
        [(1, 1), (3, 1)],                    # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
    ]
    
    col = 0
    for hl_idx, (a_blocks, b_blocks) in enumerate(high_level_terms):
        # a_blocks and b_blocks are lists of (block_idx, coeff)
        # Block idx: 0=A00, 1=A01, 2=A10, 3=A11 for A
        #             0=B00, 1=B01, 2=B10, 3=B11 for B
        
        # Convert to 2x2 coefficient matrices
        a_entry = np.zeros((2, 2))
        for block_idx, coeff in a_blocks:
            br = block_idx // 2
            bc = block_idx % 2
            a_entry[br, bc] += coeff
        
        b_entry = np.zeros((2, 2))
        for block_idx, coeff in b_blocks:
            br = block_idx // 2
            bc = block_idx % 2
            b_entry[br, bc] += coeff
        
        # Apply Strassen to this 2x2x2 problem
        for sp_idx, (A_comb, B_comb, C_comb) in enumerate(strassen_2x2):
            # A_comb and B_comb are dicts {(i,j): coeff} for the 2x2 entries
            # We need to expand this into scalar terms
            
            # U_term: 4x4 matrix
            U_term = np.zeros((4, 4))
            for (br, bc), coeff in A_comb.items():
                val = coeff * a_entry[br, bc]
                if abs(val) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            U_term[r, c] += val
            
            # V_term: 4x4 matrix
            V_term = np.zeros((4, 4))
            for (br, bc), coeff in B_comb.items():
                val = coeff * b_entry[br, bc]
                if abs(val) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            V_term[r, c] += val
            
            # W_term: 4x4 matrix
            # The C contribution is determined by C_comb and the high-level C formula
            # For this specific scalar term, it contributes to C blocks according to C_comb
            # But we need to combine with the high-level C formula
            
            # Actually, the W matrix should encode how this term contributes to C
            # The C contribution from this term is: for each (cr, cc) in C_comb,
            # add C_comb[(cr,cc)] to C block (cr,cc)
            # But this is per high-level term. The actual C is a sum over all high-level terms.
            
            # For now, let's just store U_term, V_term, W_term as the raw contributions
            # and verify at the end.
            
            W_term = np.zeros((4, 4))
            for (br, bc), coeff in C_comb.items():
                if abs(coeff) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            W_term[r, c] += coeff
            
            # Store this term
            U[col*4:(col+1)*4, col] = U_term.flatten()
            V[col*4:(col+1)*4, col] = V_term.flatten()
            W[col*4:(col+1)*4, col] = W_term.flatten()
            
            col += 1
    
    return U, V, W


def _build_strassen_4x4x8():
    """
    Build a rank-98 decomposition for 4x4x8 matrix multiplication using two rank-49 4x4x4 decompositions.
    Returns U (16x98), V (32x98), W (32x98) as numpy arrays.
    """
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Split B into two 4x4 blocks: B1 (columns 0-3) and B2 (columns 4-7)
    # C1 = A * B1 (4x4), C2 = A * B2 (4x4)
    # Use rank-49 decomposition for each
    
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # First half: C1 = A * B1
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4  # B1 is rows 0-3 of B
    W[0:16, 0:49] = W4  # C1 is rows 0-3 of C
    
    # Second half: C2 = A * B2
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4  # B2 is rows 4-7 of B
    W[16:32, 49:98] = W4  # C2 is rows 4-7 of C
    
    return U, V, W, rank


def _quick_als(U, V, W, n, m, p, max_iter=50):
    """Quick ALS iterations."""
    rank = U.shape[1]
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        # Update U
        for i in range(n):
            for j in range(m):
                A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                A = U[np.arange(n)*m + j, :] * W[k*n + np.arange(n), :]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                A = U[i*m + np.arange(m), :] * V[np.arange(m)*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try to build a rank-98 decomposition using recursive Strassen
    try:
        U98, V98, W98, rank98 = _build_strassen_4x4x8()
        err98 = _compute_error(U98, V98, W98, n, m, p)
        if err98 < 1e-7:
            best_U, best_V, best_W = U98, V98, W98
            best_rank = rank98
    except:
        pass
    
    # Try to find a better decomposition with ALS
    from itertools import product
    
    for r in [90, 92, 94, 96, 98]:
        if time.time() > _DEADLINE - 20:
            break
        if r >= best_rank:
            break
        
        for seed in range(3):
            if time.time() > _DEADLINE - 15:
                break
            np.random.seed(seed)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            U, V, W = _quick_als(U, V, W, n, m, p, max_iter=30)
            err = _compute_error(U, V, W, n, m, p)
            
            if err < 1e-6 and _verify_fast(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
