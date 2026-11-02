import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """
    Return (U1, V1, W1) for 2x2x2 Strassen decomposition.
    Shapes: U1 (4, 7), V1 (4, 7), W1 (4, 7).
    Indices: U1[i*2+j, s] corresponds to A[i,j].
             V1[j*2+k, s] corresponds to B[j,k].
             W1[k*2+i, s] corresponds to C[i,k] (Wait, standard is C[i,k] = sum A[i,j]B[j,k]).
             The task uses W[k*n+i, r]. So W index is k*2+i for 2x2.
    """
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3
    
    # M1
    U1[0, 0] = 1; U1[3, 0] = 1  # A00, A11
    V1[0, 0] = 1; V1[3, 0] = 1  # B00, B11
    W1[0, 0] = 1  # C00 (k=0, i=0 -> idx 0)
    W1[3, 0] = 1  # C11 (k=1, i=1 -> idx 3)
    
    # M2
    U1[2, 1] = 1; U1[3, 1] = 1  # A10, A11
    V1[0, 1] = 1                # B00
    W1[2, 1] = 1                # C10 (k=0, i=1 -> idx 2)
    W1[3, 1] = -1               # C11 (k=1, i=1 -> idx 3)
    
    # M3
    U1[0, 2] = 1                # A00
    V1[1, 2] = 1; V1[3, 2] = -1 # B01, B11
    W1[1, 2] = 1                # C01 (k=1, i=0 -> idx 1)
    W1[3, 2] = 1                # C11 (k=1, i=1 -> idx 3)
    
    # M4
    U1[3, 3] = 1                # A11
    V1[2, 3] = 1; V1[0, 3] = -1 # B10, B00
    W1[0, 3] = -1               # C00 (k=0, i=0 -> idx 0)
    W1[2, 3] = 1                # C10 (k=0, i=1 -> idx 2)
    
    # M5
    U1[0, 4] = 1; U1[1, 4] = 1  # A00, A01
    V1[3, 4] = 1                # B11
    W1[0, 4] = 1                # C00 (k=0, i=0 -> idx 0)
    W1[1, 4] = 1                # C01 (k=1, i=0 -> idx 1)
    
    # M6
    U1[2, 5] = 1; U1[0, 5] = -1 # A10, A00
    V1[0, 5] = 1; V1[1, 5] = 1  # B00, B01
    W1[2, 5] = 1                # C10 (k=0, i=1 -> idx 2)
    
    # M7
    U1[1, 6] = 1; U1[3, 6] = -1 # A01, A11
    V1[2, 6] = 1; V1[3, 6] = 1  # B10, B11
    W1[1, 6] = 1                # C01 (k=1, i=0 -> idx 1)
    
    return U1, V1, W1

def _build_rank42():
    """
    Build rank-42 decomposition for 2x4 @ 4x6.
    Partition B (4x6) into three 4x2 blocks: B1, B2, B3.
    C = [A B1, A B2, A B3].
    Each A Bi is a 2x4 @ 4x2 multiplication, which decomposes into two 2x2 @ 2x2 multiplications.
    Total 3 * 2 * 7 = 42 rank.
    """
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # There are 3 blocks of B (each 4x2).
    # For each block, there are 2 sub-blocks (top 2x2, bottom 2x2 of B, corresponding to left/right 2x2 of A).
    # Actually, A is 2x4. A = [AL AR] (2x2 each).
    # B_block is 4x2. B_block = [BT; BB] (2x2 each).
    # A * B_block = AL * BT + AR * BB.
    # So for each of the 3 B-blocks, we have 2 Strassen decompositions.
    
    for b_idx in range(3):
        # b_idx corresponds to the 3rd dimension of B (columns 2*b_idx to 2*b_idx+1)
        base_col_offset = b_idx * 14
        
        for side in range(2):
            # side 0: AL * BT
            # side 1: AR * BB
            col_offset = base_col_offset + side * 7
            
            # A part: AL is cols 0-1, AR is cols 2-3
            a_col_start = side * 2
            
            # B part: BT is rows 0-1, BB is rows 2-3
            b_row_start = side * 2
            
            # For each of the 7 Strassen terms
            for s in range(7):
                # U contributions
                # U1[i*2+j, s] where i in 0..1, j in 0..1
                # Global U index: i*m + (a_col_start + j)
                for i in range(2):
                    for j in range(2):
                        coeff = U1[i * 2 + j, s]
                        if coeff != 0:
                            U[i * m + a_col_start + j, col_offset + s] += coeff
                
                # V contributions
                # V1[j*2+k, s] where j in 0..1, k in 0..1
                # Global V index: (b_row_start + j) * p + (b_idx * 2 + k)
                for j in range(2):
                    for k in range(2):
                        coeff = V1[j * 2 + k, s]
                        if coeff != 0:
                            V[(b_row_start + j) * p + (b_idx * 2 + k), col_offset + s] += coeff
                
                # W contributions
                # W1[k*2+i, s] where k in 0..1, i in 0..1
                # Global W index: (b_idx * 2 + k) * n + i
                for k in range(2):
                    for i in range(2):
                        coeff = W1[k * 2 + i, s]
                        if coeff != 0:
                            W[(b_idx * 2 + k) * n + i, col_offset + s] += coeff
                            
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_output(mat):
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank42()
    
    if _verify(U, V, W, n, m, p):
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}
    
    # Fallback (should not happen)
    raise ValueError("Decomposition verification failed")

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"Verified: {_verify(U, V, W, n, m, p)}")
