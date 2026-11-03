"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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


def _strassen_2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]
    W[k*2+i, c] corresponds to C[i,k]
    """
    # Strassen's formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) for A[i,j], columns by M1..M7
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    U = [
        [1, 0, 1, 0, 1, -1, -1],  # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, -1, 1, 0, 0, -1],  # A11
    ]
    
    # V: rows indexed by (j,k) for B[j,k], columns by M1..M7
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, -1, 0, 0, 1, 0],   # B01
        [1, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, 0, 1, 1, 0, 1],    # B11
    ]
    
    # W: rows indexed by (k,i) for C[i,k], columns by M1..M7
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    W = [
        [1, 0, 0, 0, 0, 0, 1],    # C00 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0,i=1)
        [1, -1, 1, 0, 0, 1, 0],   # C11 (k=1,i=1)
    ]
    
    return np.array(U), np.array(V), np.array(W)


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _block_strassen_3x3_6():
    """
    Try to construct a decomposition for 3x3 * 3x6 by splitting into blocks.
    
    Strategy: Split A (3x3) into 2x2 + 1x1 blocks, and B (3x6) into 2x2 + 1x2 blocks.
    
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    C = [[C11, C12], [C21, C22]] where:
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x2)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x2)
    
    We can use Strassen for the 2x2*2x2 multiplications (C11's A11*B11 part, C12's A11*B12 part).
    But the cross terms complicate things. Let's try a simpler approach.
    
    Alternative: Use the fact that 3x3 * 3x6 can be viewed as three 3x3 * 3x2 multiplications.
    Or better: try to use known results.
    
    For now, let's try a structured approach:
    - Use Strassen for the 2x2*2x2 blocks
    - Handle the remaining terms with additional rank-1 terms
    """
    n, m, p = 3, 3, 6
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    rank_sb = 54
    
    # Try to build a lower-rank decomposition
    # Split into 2x2 + 1x1 for A, and 2x2+1x2 for B
    
    # For the 2x2 * 2x2 block (A11 * B11), use Strassen (rank 7)
    # A11*B11 contributes to C11 (2x2 block)
    U_s, V_s, W_s = _strassen_2x2()
    
    # We need to extend this to the full 3x3 * 3x6 problem
    # Let's try a different approach: use the structure of the tensor
    
    # The tensor T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5}
    # We want to decompose this as sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Try random search with verification, starting from moderate ranks
    best_result = sb
    best_rank = rank_sb
    
    deadline = time.time() + 130  # safety margin
    
    # Try structured search: use the Kronecker/product structure
    # For 3x3*3x6, we can try ranks like 27, 30, 33, ...
    
    # First, let's try to verify if a rank-27 decomposition exists by random search
    # This is optimistic but worth trying
    
    for target_rank in [27, 28, 29, 30, 33, 36, 40, 45]:
        if time.time() > deadline - 10:
            break
        
        max_attempts = 200 if target_rank >= 40 else 500
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Try with small integer entries
            U = np.random.choice([-2, -1, 0, 1, 2], size=(n * m, target_rank)).astype(float)
            V = np.random.choice([-2, -1, 0, 1, 2], size=(m * p, target_rank)).astype(float)
            W = np.random.choice([-2, -1, 0, 1, 2], size=(p * n, target_rank)).astype(float)
            
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            continue  # Try lower ranks
    
    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 54
    
    # Try the structured/block approach
    result = _block_strassen_3x3_6()
    if result["U"] is not None:
        U_arr = np.array(result["U"], dtype=float)
        V_arr = np.array(result["V"], dtype=float)
        W_arr = np.array(result["W"], dtype=float)
        if U_arr.shape[1] < best_rank:
            if _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
                best_result = result
                best_rank = U_arr.shape[1]
    
    # If we have time, try more random search
    if time.time() < deadline - 15 and best_rank > 30:
        for target_rank in range(best_rank - 1, max(25, best_rank - 15), -1):
            if time.time() > deadline - 5:
                break
            
            max_attempts = 100
            found = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 3:
                    break
                
                np.random.seed(attempt * 7777 + target_rank * 13)
                
                # Try with entries in {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(n * m, target_rank)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(m * p, target_rank)).astype(float)
                W = np.random.choice([-1, 0, 1], size=(p * n, target_rank)).astype(float)
                
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
            
            if found:
                continue
    
    return best_result

# EVOLVE-BLOCK-END
