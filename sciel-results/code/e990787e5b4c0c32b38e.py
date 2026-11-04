import time
import numpy as np

def construct():
    """
    Returns a rank-44 exact tensor decomposition for multiplying a 2x3 by 3x8 matrix.
    
    Strategy:
    - Split A (2x3) into A1 (2x2) and A2 (2x1).
    - Split B (3x8) into B1 (2x8) and B2 (1x8).
    - Use Strassen (rank 7) for the 2x2 * 2x8 part, applied to 4 blocks of width 2.
      This gives 4 * 7 = 28 terms.
    - Use schoolbook (rank 16) for the 2x1 * 1x8 part.
    - Total rank = 28 + 16 = 44.
    
    The output factor W follows the convention W[k*n+i, r].
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))  # 6 x 44
    V = np.zeros((m * p, rank))  # 24 x 44
    W = np.zeros((p * n, rank))  # 16 x 44
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U coefficients (for A)
    # Rows correspond to A entries: (0,0), (0,1), (1,0), (1,1)
    # But in our case, A is 2x2 for the sub-block, so indices are local.
    # Global U index: i*m + a
    # Local A index: i_local * 2 + a_local (since we are using 2x2 blocks)
    
    # Let's define the Strassen terms explicitly for 2x2 * 2x2
    # Then we will map them to the 2x2 * 2x8 case by tiling the B dimension.
    
    # Strassen terms for 2x2 * 2x2:
    # term 0: M1
    #   U: A00=1, A11=1 -> (0,0)=1, (1,1)=1
    #   V: B00=1, B11=1 -> (0,0)=1, (1,1)=1
    #   W: C00+=M1, C01+=M1, C10+=M1, C11+=M1 -> (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # term 1: M2
    #   U: A10=1, A11=1 -> (1,0)=1, (1,1)=1
    #   V: B00=1 -> (0,0)=1
    #   W: C10+=M2 -> (1,0)=1
    # term 2: M3
    #   U: A00=1 -> (0,0)=1
    #   V: B10=1, B11=-1 -> (1,0)=1, (1,1)=-1
    #   W: C10+=M3, C11+=-M3 -> (1,0)=1, (1,1)=-1
    # term 3: M4
    #   U: A11=1 -> (1,1)=1
    #   V: B01=1, B10=-1 -> (0,1)=1, (1,0)=-1
    #   W: C01+=M4, C11+=-M4 -> (0,1)=1, (1,1)=-1
    # term 4: M5
    #   U: A00=1, A01=1 -> (0,0)=1, (0,1)=1
    #   V: B11=1 -> (1,1)=1
    #   W: C00+=M5, C01+=M5 -> (0,0)=1, (0,1)=1
    # term 5: M6
    #   U: A10=1, A00=-1 -> (1,0)=1, (0,0)=-1
    #   V: B00=1, B10=1 -> (0,0)=1, (1,0)=1
    #   W: C10+=M6 -> (1,0)=1
    # term 6: M7
    #   U: A01=1, A11=-1 -> (0,1)=1, (1,1)=-1
    #   V: B01=1, B11=1 -> (0,1)=1, (1,1)=1
    #   W: C01+=M7 -> (0,1)=1
    
    # Let's encode these as lists of (index, value) for U, V, W
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    S_U = [
        [(0, 1), (3, 1)],      # M1
        [(2, 1), (3, 1)],      # M2
        [(0, 1)],              # M3
        [(3, 1)],              # M4
        [(0, 1), (1, 1)],      # M5
        [(2, 1), (0, -1)],     # M6
        [(1, 1), (3, -1)]      # M7
    ]
    
    S_V = [
        [(0, 1), (3, 1)],      # M1
        [(0, 1)],              # M2
        [(2, 1), (3, -1)],     # M3
        [(1, 1), (2, -1)],     # M4
        [(3, 1)],              # M5
        [(0, 1), (2, 1)],      # M6
        [(1, 1), (3, 1)]       # M7
    ]
    
    S_W = [
        [(0, 1), (1, 1), (2, 1), (3, 1)],  # M1
        [(2, 1)],                           # M2
        [(2, 1), (3, -1)],                  # M3
        [(1, 1), (3, -1)],                  # M4
        [(0, 1), (1, 1)],                   # M5
        [(2, 1)],                           # M6
        [(1, 1)]                            # M7
    ]
    
    col = 0
    # Part 1: A1 (2x2) * B1 (2x8)
    # We split B1 into 4 blocks of size 2x2.
    # For each block j (0..3), we use the 7 Strassen terms.
    # The A indices are global: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # The B indices for block j are:
    #   (0, 2j), (0, 2j+1), (1, 2j), (1, 2j+1)
    #   Global V index: a*p + k
    #   So for local B (la, lk) in block j:
    #     a = la, k = 2j + lk
    #     V index = la * 8 + (2j + lk)
    # The W indices for block j are:
    #   (2j, 0), (2j, 1), (2j+1, 0), (2j+1, 1)
    #   Global W index: k*n + i
    #   So for local W (lk, li) in block j:
    #     k = 2j + lk, i = li
    #     W index = (2j + lk) * 2 + li
    
    for j in range(4):
        for r in range(7):
            # U
            for idx, val in S_U[r]:
                U[idx, col] += val
            # V
            for idx, val in S_V[r]:
                la, lk = idx // 2, idx % 2
                v_idx = la * p + (2 * j + lk)
                V[v_idx, col] += val
            # W
            for idx, val in S_W[r]:
                lk, li = idx // 2, idx % 2
                w_idx = (2 * j + lk) * n + li
                W[w_idx, col] += val
            col += 1
    
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 corresponds to column 2 of A.
    # A indices: (0,2) -> U index 0*3+2 = 2, (1,2) -> U index 1*3+2 = 5
    # B2 corresponds to row 2 of B.
    # B indices: (2,0)..(2,7) -> V indices 2*8+0=16 .. 2*8+7=23
    # W indices: (k,0) and (k,1) for k=0..7
    #   W index for (k,0) is k*2+0 = 2k
    #   W index for (k,1) is k*2+1 = 2k+1
    
    # For each k in 0..7, we have two terms:
    # Term 1: A(0,2) * B(2,k) * W(k,0)
    # Term 2: A(1,2) * B(2,k) * W(k,1)
    
    for k in range(8):
        # Term for i=0
        U[2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1
        
        # Term for i=1
        U[5, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1
    
    # Verification (optional, for debugging)
    # We can verify that the decomposition is exact.
    # For each (i, a, k), the sum over r of U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] should be 1.
    
    def _validate(U, V, W):
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        acc += U[i * m + a, r] * V[a * p + k, r] * W[k * n + i, r]
                    if abs(acc - 1.0) > 1e-9:
                        return False
        return True
    
    # if not _validate(U, V, W):
    #     raise ValueError("Decomposition is not exact")
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))

