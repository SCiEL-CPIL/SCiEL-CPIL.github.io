import time
import random

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit

    n, m, p = 5, 5, 5

    # Block partition: split each dimension into sizes that sum to 5
    # We'll use a 2+2+1 partition for all three dimensions
    # This gives blocks of sizes: (2,2), (2,1), (1,2), (1,1) for the 3x3 grid
    # For each block of size (a,b,c) where we multiply an a×b matrix by b×c matrix to get a×c:
    # - If a=b=c=2, use Strassen (rank 7 instead of 8)
    # - Otherwise use schoolbook (rank a*b*c)
    
    # Partition indices
    # Rows of A: split into [0,1], [2,3], [4] -> sizes 2,2,1
    # Cols of A / Rows of B: split into [0,1], [2,3], [4] -> sizes 2,2,1
    # Cols of B: split into [0,1], [2,3], [4] -> sizes 2,2,1
    
    row_parts = [(0, 2), (2, 4), (4, 5)]  # start, end (exclusive)
    col_parts = [(0, 2), (2, 4), (4, 5)]
    out_parts = [(0, 2), (2, 4), (4, 5)]
    
    U_factors = []
    V_factors = []
    W_factors = []
    
    for ri, (rs, re) in enumerate(row_parts):
        a_size = re - rs
        for ci, (cs, ce) in enumerate(col_parts):
            b_size = ce - cs
            for oi, (os_, oe) in enumerate(out_parts):
                c_size = oe - os_
                
                # Block: A[rs:re, cs:ce] @ B[cs:ce, os_:oe] = C[rs:re, os_:oe]
                # Size: a_size × b_size × c_size
                
                if a_size == 2 and b_size == 2 and c_size == 2:
                    # Use Strassen's algorithm for this 2x2x2 block
                    # Strassen decomposition for 2x2 matmul:
                    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
                    # M1 = (A11 + A22)(B11 + B22)
                    # M2 = (A21 + A22)B11
                    # M3 = A11(B12 - B22)
                    # M4 = A22(B21 - B11)
                    # M5 = (A11 + A12)B22
                    # M6 = (A21 - A11)(B11 + B12)
                    # M7 = (A12 - A22)(B21 + B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # For the tensor decomposition, each M_k contributes one rank-1 term
                    # M1: (A11+A22)(B11+B22) -> U has 1 at (A11) and (A22), V has 1 at (B11) and (B22)
                    # We need to map local indices to global indices
                    
                    # Local (i,j,k) for this block: i in {0,1} (row of A block), j in {0,1} (col of A block / row of B block), k in {0,1} (col of B block)
                    # Global: row = rs + i, col_a = cs + j, col_b = os_ + k
                    
                    # M1: (A11+A22)(B11+B22)
                    # A11: local (0,0) -> global (rs+0, cs+0)
                    # A22: local (1,1) -> global (rs+1, cs+1)
                    # B11: local (0,0) in B -> (cs+0, os_+0)
                    # B22: local (1,1) in B -> (cs+1, os_+1)
                    # C11: local (0,0) -> (rs+0, os_+0)
                    # C22: local (1,1) -> (rs+1, os_+1)
                    # W contribution: C11 and C22 both get +1 from M1
                    
                    u_idx1 = []
                    v_idx1 = []
                    w_idx1 = []
                    # M1
                    u_idx1.append(rs * m + cs)       # A11
                    u_idx1.append((rs+1) * m + (cs+1))  # A22
                    v_idx1.append((cs) * p + os_)   # B11
                    v_idx1.append((cs+1) * p + (os_+1))  # B22
                    w_idx1.append((os_) * n + rs)    # C11
                    w_idx1.append((os_+1) * n + (rs+1))  # C22
                    
                    # M2: (A21+A22)B11
                    u_idx2 = []
                    v_idx2 = []
                    w_idx2 = []
                    u_idx2.append((rs+1) * m + cs)       # A21
                    u_idx2.append((rs+1) * m + (cs+1))   # A22
                    v_idx2.append(cs * p + os_)          # B11
                    w_idx2.append((os_+1) * n + (rs+1))  # C21
                    
                    # M3: A11(B12-B22)
                    u_idx3 = [rs * m + cs]  # A11
                    v_idx3 = [cs * p + (os_+1), (cs+1) * p + (os_+1)]  # B12, -B22
                    w_idx3 = [(os_+1) * n + rs]  # C12 (with signs)
                    
                    # Actually, for exact tensor decomposition with integer/half-integer entries,
                    # we need to be more careful. Let me just use the standard Strassen rank-7.
                    
                    # Let me build the factors properly using the Strassen decomposition
                    # Each of the 7 M's gives one rank-1 term in the tensor
                    
                    factors_U = []
                    factors_V = []
                    factors_W = []
                    
                    # M1 = (A11 + A22)(B11 + B22)
                    factors_U.append({rs * m + cs: 1, (rs+1) * m + (cs+1): 1})
                    factors_V.append({cs * p + os_: 1, (cs+1) * p + (os_+1): 1})
                    factors_W.append({(os_) * n + rs: 1, (os_+1) * n + (rs+1): 1})
                    
                    # M2 = (A21 + A22)B11
                    factors_U.append({(rs+1) * m + cs: 1, (rs+1) * m + (cs+1): 1})
                    factors_V.append({cs * p + os_: 1})
                    factors_W.append({(os_+1) * n + (rs+1): 1})
                    
                    # M3 = A11(B12 - B22)
                    factors_U.append({rs * m + cs: 1})
                    factors_V.append({cs * p + (os_+1): 1, (cs+1) * p + (os_+1): -1})
                    factors_W.append({(os_+1) * n + rs: 1})
                    
                    # M4 = A22(B21 - B11)
                    factors_U.append({(rs+1) * m + (cs+1): 1})
                    factors_V.append({(cs+1) * p + os_: 1, cs * p + os_: -1})
                    factors_W.append({(os_) * n + (rs+1): 1})
                    
                    # M5 = (A11 + A12)B22
                    factors_U.append({rs * m + cs: 1, rs * m + (cs+1): 1})
                    factors_V.append({(cs+1) * p + (os_+1): 1})
                    factors_W.append({(os_) * n + rs: -1})
                    
                    # M6 = (A21 - A11)(B11 + B12)
                    factors_U.append({(rs+1) * m + cs: 1, rs * m + cs: -1})
                    factors_V.append({cs * p + os_: 1, cs * p + (os_+1): 1})
                    factors_W.append({(os_+1) * n + (rs+1): 1})
                    
                    # M7 = (A12 - A22)(B21 + B22)
                    factors_U.append({rs * m + (cs+1): 1, (rs+1) * m + (cs+1): -1})
                    factors_V.append({(cs+1) * p + os_: 1, (cs+1) * p + (os_+1): 1})
                    factors_W.append({(os_) * n + rs: 1})
                    
                    # Now check: C11 = M1 + M4 - M5 + M7
                    # C11 gets: M1(+1) + M4(+1) + M5(-1) + M7(+1) = 1+1-1+1 = 2? That's wrong.
                    # Let me re-check Strassen's formula.
                    # Standard Strassen:
                    # M1 = (A11 + A22)(B11 + B22)
                    # M2 = (A21 + A22)B11
                    # M3 = A11(B12 - B22)
                    # M4 = A22(B21 - B11)
                    # M5 = (A11 + A12)B22
                    # M6 = (A21 - A11)(B11 + B12)
                    # M7 = (A12 - A22)(B21 + B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # So for C11: M1(+1), M4(+1), M5(-1), M7(+1)
                    # For C12: M3(+1), M5(+1)
                    # For C21: M2(+1), M4(+1)
                    # For C22: M1(+1), M2(-1), M3(+1), M6(+1)
                    
                    # Let me redo the W factors:
                    factors_W = []
                    # M1
                    factors_W.append({(os_) * n + rs: 1, (os_+1) * n + (rs+1): 1})  # C11: +1, C22: +1
                    # M2
                    factors_W.append({(os_+1) * n + (rs+1): 1, (os_+1) * n + (rs+1): -1})  # C21: +1, C22: -1
                    # Wait, M2 contributes to C21 (+1) and C22 (-1)
                    factors_W = []
                    # M1: C11(+1), C22(+1)
                    factors_W.append({(os_) * n + rs: 1, (os_+1) * n + (rs+1): 1})
                    # M2: C21(+1), C22(-1)
                    factors_W.append({(os_+1) * n + (rs+1): 1, (os_+1) * n + (rs+1): -1})
                    # Hmm, M2 only contributes to C21 and C22, both at position (rs+1, os_+1)
                    # Actually C21 is at row rs+1, col os_ (local), so global (rs+1, os_)
                    # C22 is at row rs+1, col os_+1 (local), so global (rs+1, os_+1)
                    
                    # Let me be very careful:
                    # Local C indices: (i,k) where i in {0,1} is row, k in {0,1} is col
                    # C11: (0,0) -> global (rs, os_)
                    # C12: (0,1) -> global (rs, os_+1)
                    # C21: (1,0) -> global (rs+1, os_)
                    # C22: (1,1) -> global (rs+1, os_+1)
                    
                    factors_W = []
                    # M1: C11(+1), C22(+1)
                    factors_W.append({os_ * n + rs: 1, (os_+1) * n + (rs+1): 1})
                    # M2: C21(+1), C22(-1)
                    factors_W.append({os_ * n + (rs+1): 1, (os_+1) * n + (rs+1): -1})
                    # M3: C12(+1)
                    factors_W.append({(os_+1) * n + rs: 1})
                    # M4: C11(+1), C21(+1)
                    factors_W.append({os_ * n + rs: 1, os_ * n + (rs+1): 1})
                    # M5: C11(-1), C12(+1)
                    factors_W.append({os_ * n + rs: -1, (os_+1) * n + rs: 1})
                    # M6: C22(+1)
                    factors_W.append({(os_+1) * n + (rs+1): 1})
                    # M7: C11(+1)
                    factors_W.append({os_ * n + rs: 1})
                    
                    # Now U and V:
                    factors_U = []
                    factors_V = []
                    # M1: (A11 + A22)(B11 + B22)
                    factors_U.append({rs * m + cs: 1, (rs+1) * m + (cs+1): 1})
                    factors_V.append({cs * p + os_: 1, (cs+1) * p + (os_+1): 1})
                    # M2: (A21 + A22)B11
                    factors_U.append({(rs+1) * m + cs: 1, (rs+1) * m + (cs+1): 1})
                    factors_V.append({cs * p + os_: 1})
                    # M3: A11(B12 - B22)
                    factors_U.append({rs * m + cs: 1})
                    factors_V.append({cs * p + (os_+1): 1, (cs+1) * p + (os_+1): -1})
                    # M4: A22(B21 - B11)
                    factors_U.append({(rs+1) * m + (cs+1): 1})
                    factors_V.append({(cs+1) * p + os_: 1, cs * p + os_: -1})
                    # M5: (A11 + A12)B22
                    factors_U.append({rs * m + cs: 1, rs * m + (cs+1): 1})
                    factors_V.append({(cs+1) * p + (os_+1): 1})
                    # M6: (A21 - A11)(B11 + B12)
                    factors_U.append({(rs+1) * m + cs: 1, rs * m + cs: -1})
                    factors_V.append({cs * p + os_: 1, cs * p + (os_+1): 1})
                    # M7: (A12 - A22)(B21 + B22)
                    factors_U.append({rs * m + (cs+1): 1, (rs+1) * m + (cs+1): -1})
                    factors_V.append({(cs+1) * p + os_: 1, (cs+1) * p + (os_+1): 1})
                    
                    U_factors.append(factors_U)
                    V_factors.append(factors_V)
                    W_factors.append(factors_W)
                else:
                    # Schoolbook for this block
                    # For each (i, j, k) in the block, add a rank-1 term
                    u_list = []
                    v_list = []
                    w_list = []
                    for i in range(a_size):
                        for j in range(b_size):
                            for k in range(c_size):
                                u_list.append({(rs + i) * m + (cs + j): 1})
                                v_list.append({(cs + j) * p + (os_ + k): 1})
                                w_list.append({(os_ + k) * n + (rs + i): 1})
                    U_factors.append(u_list)
                    V_factors.append(v_list)
                    W_factors.append(w_list)
    
    # Now assemble the full factors
    # Count total rank
    total_rank = 0
    for block_U, block_V, block_W in zip(U_factors, V_factors, W_factors):
        total_rank += len(block_U)
    
    rank = total_rank
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block_U, block_V, block_W in zip(U_factors, V_factors, W_factors):
        for u_dict, v_dict, w_dict in zip(block_U, block_V, block_W):
            for idx, val in u_dict.items():
                U[idx][col] += val
            for idx, val in v_dict.items():
                V[idx][col] += val
            for idx, val in w_dict.items():
                W[idx][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
