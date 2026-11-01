import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5: (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into blocks using Strassen where possible.
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 block (j in {0,1}, k in {0,1}) -> Strassen rank 7
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2) -> schoolbook rank 4
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # Part 3: 2x1x3 block (j=2, k in {0,1,2}) -> schoolbook rank 6
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _try_extended_strassen_16():
    """
    Try to find a rank-16 decomposition by extending Strassen's 7 products
    to cover more of the 2x3x3 tensor. The idea: use 7 Strassen-like products
    for the 2x2x2 core, then add 9 more products that handle the remaining
    dimensions more efficiently than pure schoolbook.
    
    Strategy: Treat the 2x3x3 as overlapping 2x2x2 sub-problems and try to
    share computational work.
    """
    n, m, p = 2, 3, 3
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use 7 Strassen products for j in {0,1}, k in {0,1}
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Remaining: need to cover all (i,j,k) where NOT (j in {0,1} AND k in {0,1})
    # That's: (j in {0,1}, k=2) -> 4 terms, (j=2, k in {0,1,2}) -> 6 terms
    # Total 10 remaining terms, but we only have 9 slots.
    # Try to merge one pair using a sum/difference trick.
    
    # Part 2: j in {0,1}, k=2 (4 terms: i=0,1 x j=0,1)
    # Part 3: j=2, k in {0,1,2} (6 terms: i=0,1 x k=0,1,2)
    
    # Try: merge (i=0,j=0,k=2) and (i=1,j=0,k=2) into one product?
    # That would require U[0][r]=U[3][r] for some r, V[2][r] same, W[4][r]=W[5][r]
    # But then both C[2,0] and C[2,1] get the same contribution, which is wrong
    # unless we can separate them elsewhere.
    
    # Alternative: use a different block split.
    # Try: 2x3x2 (j in {0,1,2}, k in {0,1}) + 2x3x1 (j in {0,1,2}, k=2)
    # 2x3x2 can be split as 2x2x2 + 2x1x2 = 7 + 4 = 11
    # 2x3x1 = 6
    # Total = 17. Same.
    
    # Try yet another approach: use 3 overlapping 2x2x2 blocks
    # Block A: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen 7
    # Block B: i in {0,1}, j in {1,2}, k in {0,1} -> Strassen 7 (overlaps j=1)
    # Block C: i in {0,1}, j in {0,1,2}, k=2 -> schoolbook 6
    # But overlapping means double-counting for j=1, k in {0,1}
    
    # This approach gets complicated. Let's just try rank 17 for now and see
    # if we can verify it works.
    return None  # Skip this for now, too complex without more thought

def _attempt_rank_reduction_by_absorption():
    """
    Try to reduce rank by absorbing some schoolbook terms into modified
    Strassen terms. Start with rank 17 and try to merge pairs.
    """
    n, m, p = 2, 3, 3
    U17, V17, W17 = _build_decomposed_2x3x3()
    rank = 17
    U = [row[:] for row in U17]
    V = [row[:] for row in V17]
    W = [row[:] for row in W17]
    
    # Try to see if any two columns can be merged (i.e., one is a linear combo
    # of the other plus a permutation of existing columns)
    # For simplicity, check if any column is zero or duplicate
    for r in range(rank):
        u_col = [U[i][r] for i in range(n*m)]
        v_col = [V[i][r] for i in range(m*p)]
        w_col = [W[i][r] for i in range(p*n)]
        if all(x == 0 for x in u_col + v_col + w_col):
            # Zero column, can remove
            pass
    
    return None  # No obvious reduction found

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # First: try the known rank-17 decomposition
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
