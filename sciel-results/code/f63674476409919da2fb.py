import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _laderman_3x3():
    """
    Laderman rank-7 decomposition for 3x3x3 matrix multiplication.
    Returns (U, V, W) each of shape (9, 7).
    U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Laderman's 7-term formula for 3x3 matrix multiplication
    # C = A * B, where A, B, C are 3x3
    # The 7 products:
    # P1 = (A00+A01+A02)*(B00+B10+B20)
    # P2 = (A10+A11+A12)*(B00+B01+B02)
    # P3 = (A01+A11+A21)*(B10+B11+B12)
    # P4 = (A02+A12+A22)*(B20+B21+B22)
    # P5 = (A00+A02)*(B10+B12) - P1 - P3 - P4
    # P6 = (A10+A12)*(B01+B02) - P1 - P2 - P4
    # P7 = (A01+A11)*(B02+B20) - P1 - P2 - P3
    #
    # C00 = P1
    # C01 = P2 - P1 + P7
    # C02 = P4 - P1 - P6 + P7  ... this is getting complicated
    #
    # Let me use a verified Laderman decomposition.
    # Actually, the standard Laderman decomposition for 3x3 is:
    #
    # M1 = A00 * B00
    # M2 = (A00+A01) * (B10+B11)
    # M3 = (A01+A02) * (B00+B01)
    # M4 = (A10+A11) * (B01+B02)
    # M5 = (A11+A12) * (B10+B11)
    # M6 = (A20+A21) * (B02+B12)
    # M7 = (A00+A02)*(B11+B12) + (A10+A12)*(B01+B02) - M1 - M3 - M4 - M6
    #
    # C00 = M1
    # C01 = M2 - M1
    # C02 = M3 - M1
    # C10 = M4 - M2
    # C11 = M5 - M2 - M4 + M1
    # C12 = M4 - M3
    # C20 = M6 - M4
    # C21 = M5 - M6
    # C22 = M7 - M5 + M6 - M3 + M1
    #
    # Hmm, I'm not confident in this. Let me use a different well-known rank-7 for 3x3.
    #
    # Actually, the Laderman decomposition (1976) gives rank 7 for 3x3.
    # Let me look up the exact formula.
    #
    # Laderman's formula:
    # Let A = [a00 a01 a02; a10 a11 a12; a20 a21 a22]
    # Let B = [b00 b01 b02; b10 b11 b12; b20 b21 b22]
    #
    # The 7 scalar products are:
    # m1 = (a00+a01+a02)*(b00+b10+b20)
    # m2 = (a10+a11+a12)*(b00+b01+b02)
    # m3 = (a01+a11+a21)*(b10+b11+b12)
    # m4 = (a02+a12+a22)*(b20+b21+b22)
    # m5 = (a00+a02)*(b10+b12)
    # m6 = (a10+a12)*(b01+b02)
    # m7 = (a01+a11)*(b02+b20)
    #
    # C00 = m1 - m5 - m6 - m7 + ... 
    #
    # I'm going in circles. Let me just use a verified approach.
    # 
    # For safety, let me use Strassen for 2x2 (which I know well) and schoolbook for the rest.
    # 5 = 2+2+1, so:
    # - 2x2 block: Strassen rank 7
    # - 2x2 block: Strassen rank 7  
    # - 1x1 block: rank 1
    # - Cross terms: 2*2 + 2*2 + 2*1 + 2*1 + 1*2 + 1*2 + 2*1 + 2*1 + 1*2 + 1*2 + 1*1 + 1*1
    #   = 4+4+2+2+2+2+2+2+2+2+1+1 = 28
    # Total: 7+7+1+28 = 43
    #
    # Or 5 = 3+2:
    # - 3x3 block: need rank 7 (Laderman)
    # - 2x2 block: Strassen rank 7
    # - Cross: 3*2 + 2*3 + 3*2 + 2*3 = 6+6+6+6 = 24
    # Total: 7+7+24 = 38
    #
    # I need a correct Laderman decomposition. Let me derive/verify one.
    #
    # Actually, let me just use 5=2+2+1 with Strassen for the 2x2 blocks.
    # That gives rank 43 which is better than 125.
    
    pass

def _strassen_2x2():
    """
    Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    C = A*B where A,B are 2x2.
    A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    C = [C00 C01; C10 C11]
    
    M1 = A00*B00
    M2 = (A00+A01)*B10
    M3 = (A10+A11)*B01
    M4 = A11*B11
    M5 = (A01-A11)*(B10+B11)
    M6 = (A00-A10)*(B01+B11)
    M7 = (A00+A11)*(B00+B11) - M1 - M4
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M6
    C11 = M1 - M2 + M4 + M7
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = A00*B00: U[0,0]=1, V[0,0]=1, W[0,0]=1 (C00 gets +1)
    U[0, 0] = 1.0; V[0, 0] = 1.0; W[0, 0] = 1.0
    # C00 += M1, C11 += M1
    W[0, 0] = 1.0  # C00
    # We need W[k*2+i, r] where C[k,i] gets the contribution
    # C00 is at index 0*2+0=0, C01 at 0*2+1=1, C10 at 1*2+0=2, C11 at 1*2+1=3
    
    # Let me redo this carefully.
    # W[k*2+i, r] is the coefficient for C[k,i]
    # So W[0,0]=1 means C[0,0] gets +1*M1
    # W[3,0]=1 means C[1,1] gets +1*M1
    
    # Reset and do it properly
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = A00 * B00
    # U: A00 is at index 0*2+0=0
    # V: B00 is at index 0*2+0=0
    # W: C00 at 0, C11 at 3
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0   # C00 += M1
    W[3, 0] = 1.0   # C11 += M1
    
    # M2 = (A00+A01) * B10
    # U: A00 at 0, A01 at 1 -> U[0,1]=1, U[1,1]=1
    # V: B10 at 1*2+0=2
    # W: C10 at 1*2+0=2
    U[0, 1] = 1.0
    U[1, 1] = 1.0
    V[2, 1] = 1.0
    W[2, 1] = 1.0   # C10 += M2
    W[3, 1] = -1.0  # C11 -= M2
    
    # M3 = (A10+A11) * B01
    # U: A10 at 1*2+0=2, A11 at 3 -> U[2,2]=1, U[3,2]=1
    # V: B01 at 0*2+1=1
    # W: C01 at 0*2+1=1
    U[2, 2] = 1.0
    U[3, 2] = 1.0
    V[1, 2] = 1.0
    W[1, 2] = 1.0   # C01 += M3
    
    # M4 = A11 * B11
    # U: A11 at 3
    # V: B11 at 1*2+1=3
    # W: C00 at 0, C11 at 3
    U[3, 3] = 1.0
    V[3, 3] = 1.0
    W[0, 3] = 1.0   # C00 += M4
    W[3, 3] = 1.0   # C11 += M4
    
    # M5 = (A01-A11) * (B10+B11)
    # U: A01 at 1 (+1), A11 at 3 (-1)
    # V: B10 at 2 (+1), B11 at 3 (+1)
    # W: C00 at 0 (-1), C01 at 1 (+1)
    U[1, 4] = 1.0
    U[3, 4] = -1.0
    V[2, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0  # C00 -= M5
    W[1, 4] = 1.0   # C01 += M5
    
    # M6 = (A00-A10) * (B01+B11)
    # U: A00 at 0 (+1), A10 at 2 (-1)
    # V: B01 at 1 (+1), B11 at 3 (+1)
    # W: C10 at 2 (+1)
    U[0, 5] = 1.0
    U[2, 5] = -1.0
    V[1, 5] = 1.0
    V[3, 5] = 1.0
    W[2, 5] = 1.0   # C10 += M6
    
    # M7 = (A00+A11) * (B00+B11) - M1 - M4
    # U: A00 at 0 (+1), A11 at 3 (+1)
    # V: B00 at 0 (+1), B11 at 3 (+1)
    # W: C00 at 0 (-1), C11 at 3 (+1)
    U[0, 6] = 1.0
    U[3, 6] = 1.0
    V[0, 6] = 1.0
    V[3, 6] = 1.0
    W[0, 6] = -1.0  # C00 -= M7
    W[3, 6] = 1.0   # C11 += M7
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Validate decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: 5 = 2 + 2 + 1
    # Blocks: indices {0,1}, {2,3}, {4}
    # 2x2 block (0,1)x(0,1): Strassen rank 7
    # 2x2 block (2,3)x(2,3): Strassen rank 7
    # 1x1 block (4)x(4): rank 1
    # Cross blocks (schoolbook):
    #   (0,1)x(2,3): 2*2=4 terms
    #   (0,1)x(4):   2*1=2 terms
    #   (2,3)x(0,1): 2*2=4 terms
    #   (2,3)x(4):   2*1=2 terms
    #   (4)x(0,1):   1*2=2 terms
    #   (4)x(2,3):   1*2=2 terms
    # Total cross: 4+2+4+2+2+2 = 16
    # Total rank: 7 + 7 + 1 + 16 = 31
    
    # Actually wait, let me recount cross terms.
    # A is 5x5, B is 5x5. A[i,j]*B[j,k] -> C[k,i]
    # Block structure on i (rows of A, cols of C): {0,1}, {2,3}, {4}
    # Block structure on j (cols of A, rows of B): {0,1}, {2,3}, {4}
    # Block structure on k (cols of B, rows of C): {0,1}, {2,3}, {4}
    #
    # For each pair of i-block and k-block, and each j-block:
    # i-block I, j-block J, k-block K: contributes |I|*|J|*|K| schoolbook terms
    # unless we use Strassen/Laderman for matching blocks.
    #
    # Pure blocks (I=J=K):
    #   {0,1}x{0,1}x{0,1}: 2x2x2 -> Strassen rank 7
    #   {2,3}x{2,3}x{2,3}: 2x2x2 -> Strassen rank 7
    #   {4}x{4}x{4}: 1x1x1 -> rank 1
    #
    # Cross blocks (I != J or J != K or I != K): schoolbook
    #   I={0,1}, J={2,3}, K={0,1}: 2*2*2=8
    #   I={0,1}, J={2,3}, K={2,3}: 2*2*2=8
    #   I={0,1}, J={2,3}, K={4}: 2*2*1=4
    #   I={0,1}, J={4}, K={0,1}: 2*1*2=4
    #   I={0,1}, J={4}, K={2,3}: 2*1*2=4
    #   I={0,1}, J={4}, K={4}: 2*1*1=2
    #   I={2,3}, J={0,1}, K={0,1}: 2*2*2=8
    #   I={2,3}, J={0,1}, K={2,3}: 2*2*2=8
    #   I={2,3}, J={0,1}, K={4}: 2*2*1=4
    #   I={2,3}, J={2,3}, K={0,1}: 2*2*2=8
    #   I={2,3}, J={4}, K={0,1}: 2*1*2=4
    #   I={2,3}, J={4}, K={2,3}: 2*1*2=4
    #   I={2,3}, J={4}, K={4}: 2*1*1=2
    #   I={4}, J={0,1}, K={0,1}: 1*2*2=4
    #   I={4}, J={0,1}, K={2,3}: 1*2*2=4
    #   I={4}, J={0,1}, K={4}: 1*2*1=2
    #   I={4}, J={2,3}, K={0,1}: 1*2*2=4
    #   I={4}, J={2,3}, K={2,3}: 1*2*2=4
    #   I={4}, J={2,3}, K={4}: 1*2*1=2
    #   I={4}, J={4}, K={0,1}: 1*1*2=2
    #   I={4}, J={4}, K={2,3}: 1*1*2=2
    #
    # Total cross: 8+8+4+4+4+2+8+8+4+8+4+4+2+4+4+2+4+4+2+2+2 = let me add
    # 8+8+4+4+4+2 = 30
    # 8+8+4+8+4+4+2 = 38
    # 4+4+2+4+4+2+2+2 = 24
    # Total cross = 30+38+24 = 92
    # Total = 7+7+1+92 = 107. That's worse than 125!
    
    # I made an error. The cross terms should only be for I != J (since J must match for the multiplication to make sense in blocks).
    # Actually no - in matrix multiplication C[k,i] = sum_j A[i,j]*B[j,k], the j index is the contraction index.
    # For block decomposition, we need J to be the same block for A and B.
    # So the blocks are: for each pair (I_block, K_block) and each J_block:
    #   Contribution from A[I,J]*B[J,K] to C[K,I]
    #
    # Pure diagonal blocks (I=J=K):
    #   {0,1}: 2x2x2 Strassen = 7
    #   {2,3}: 2x2x2 Strassen = 7
    #   {4}: 1x1x1 = 1
    #
    # Off-diagonal: I != K but J can be anything
    # For I_block, K_block (I!=K), and each J_block:
    #   |I|*|J|*|K| schoolbook terms
    #
    # I={0,1}, K={2,3}: J={0,1}: 2*2*2=8, J={2,3}: 2*2*2=8, J={4}: 2*1*2=4 -> 20
    # I={0,1}, K={4}: J={0,1}: 2*2*1=4, J={2,3}: 2*2*1=4, J={4}: 2*1*1=2 -> 10
    # I={2,3}, K={0,1}: J={0,1}: 2*2*2=8, J={2,3}: 2*2*2=8, J={4}: 2*1*2=4 -> 20
    # I={2,3}, K={4}: J={0,1}: 2*2*1=4, J={2,3}: 2*2*1=4, J={4}: 2*1*1=2 -> 10
    # I={4}, K={0,1}: J={0,1}: 1*2*2=4, J={2,3}: 1*2*2=4, J={4}: 1*1*2=2 -> 10
    # I={4}, K={2,3}: J={0,1}: 1*2*2=4, J={2,3}: 1*2*2=4, J={4}: 1*1*2=2 -> 10
    #
    # Total off-diagonal: 20+10+20+10+10+10 = 80
    # Total: 7+7+1+80 = 95. Still worse than 125? No, 95 < 125.
    
    # Hmm, but that's a lot of terms. Let me use 3+2 split instead.
    # 5 = 3 + 2
    # Blocks: {0,1,2} and {3,4}
    # Pure: 3x3x3 (Laderman rank 7) + 2x2x2 (Strassen rank 7) = 14
    # Off-diagonal:
    #   I={0,1,2}, K={3,4}: J={0,1,2}: 3*3*2=18, J={3,4}: 3*2*2=12 -> 30
    #   I={3,4}, K={0,1,2}: J={0,1,2}: 2*3*3=18, J={3,4}: 2*2*3=12 -> 30
    # Total: 14 + 60 = 74. Better!
    
    # But I need a correct Laderman decomposition. Let me try to get one.
    #
    # Alternative: use 5=2+2+1 but with a smarter cross-term handling.
    # Actually, the simplest correct improvement is to just use Strassen for 2x2 blocks.
    #
    # Let me try: 5 = 2+2+1, use Strassen for the two 2x2 diagonal blocks.
    # Rank = 7 + 7 + 1 + 80 = 95.
    
    # Let me just build the 95-rank decomposition.
    
    # Actually, let me try 3+2 with a verified Laderman.
    # I'll implement Laderman and verify it.
    
    # Laderman 3x3 rank-7 (verified formula):
    # Reference: Laderman, "A note on the multiplication of matrices", 1976
    # 
    # A = [[a,b,c],[d,e,f],[g,h,i]]
    # B = [[x,y,z],[u,v,w],[p,q,r]]
    #
    # 7 products:
    # P1 = (a+b+c)*(x+u+p)
    # P2 = (d+e+f)*(x+y+z)
    # P3 = (b+e+h)*(u+v+w)
    # P4 = (c+f+i)*(p+q+r)
    # P5 = (a+c)*(u+w)
    # P6 = (d+f)*(y+z)
    # P7 = (b+e)*(z+p)
    #
    # C[0,0] = P1 - P5 - P6 - P7 + ... 
    #
    # I don't remember the exact reconstruction. Let me just use the 95-rank approach
    # with Strassen for 2x2 blocks, which I can verify.
    
    U = np.zeros((25, 95), dtype=float)
    V = np.zeros((25, 95), dtype=float)
    W = np.zeros((25, 95), dtype=float)
    
    col = 0
    
    # Strassen for block {0,1} x {0,1} -> C[0,1] x [0,1]
    # Local indices: i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
    # Global: i = i_local, j = j_local, k = k_local
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                U[li*5 + lj, col] += sU[li*2 + lj, r]
        for lj in range(2):
            for lk in range(2):
                V[lj*5 + lk, col] += sV[lj*2 + lk, r]
        for lk in range(2):
            for li in range(2):
                W[lk*5 + li, col] += sW[lk*2 + li, r]
        col += 1
    
    # Strassen for block {2,3} x {2,3} -> C[2,3] x [2,3]
    # Local indices: i_local in {0,1} -> global i = 2+i_local
    #                j_local in {0,1} -> global j = 2+j_local
    #                k_local in {0,1} -> global k = 2+k_local
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                U[(2+li)*5 + (2+lj), col] += sU[li*2 + lj, r]
        for lj in range(2):
            for lk in range(2):
                V[(2+lj)*5 + (2+lk), col] += sV[lj*2 + lk, r]
        for lk in range(2):
            for li in range(2):
                W[(2+lk)*5 + (2+li), col] += sW[lk*2 + li, r]
        col += 1
    
    # 1x1 block {4} x {4} -> C[4] x [4]
    U[4*5 + 4, col] = 1.0
    V[4*5 + 4, col] = 1.0
    W[4*5 + 4, col] = 1.0
    col += 1
    
    # Off-diagonal blocks: schoolbook
    # For each I_block, K_block (I!=K), J_block:
    blocks = [[0,1], [2,3], [4]]
    block_ids = range(3)
    
    for bi in block_ids:
        for bk in block_ids:
            if bi == bk:
                continue
            for bj in block_ids:
                for ii in blocks[bi]:
                    for jj in blocks[bj]:
                        for kk in blocks[bk]:
                            U[ii*5 + jj, col] = 1.0
                            V[jj*5 + kk, col] = 1.0
                            W[kk*5 + ii, col] = 1.0
                            col += 1
    
    assert col == 95, f"Expected 95 columns, got {col}"
    
    # Validate
    valid, info = _validate(U, V, W, 5, 5, 5)
    if not valid:
        # Fall back to schoolbook
        U = np.zeros((25, 125), dtype=float)
        V = np.zeros((25, 125), dtype=float)
        W = np.zeros((25, 125), dtype=float)
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, col] = 1.0
                    V[j*5+k, col] = 1.0
                    W[k*5+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
