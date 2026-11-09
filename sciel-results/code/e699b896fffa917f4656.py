"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_real(U, V, W, n, m, p):
    """Proper verification: check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _make_schoolbook(n, m, p, offset=0):
    """Create a schoolbook decomposition for n x m x p, appending to existing matrices."""
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, n * m * p

def _make_strassen_222(offset=0):
    """Create a rank-7 Strassen decomposition for 2x2x2."""
    # Strassen's 7 multiplications
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (W index is k*n+i = k*2+i)
    # W[0] is C00, W[1] is C01, W[2] is C10, W[3] is C11
    
    # P1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1 # Wait, C11 = P1... C00 = P1...
    # Let's map carefully.
    # C00 (W[0]) = P1 + P4 - P5 + P7
    # C01 (W[1]) = P3 + P5
    # C10 (W[2]) = P2 + P4
    # C11 (W[3]) = P1 - P2 + P3 + P6
    
    W[0][0] = 1
    W[3][0] = 1
    
    # P2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # P3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # P4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # P5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # P6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions.
    Each decomp is (U, V, W, rank).
    The decomps must correspond to a partition of the problem.
    This function is a placeholder for the actual logic which is specific to the split.
    """
    pass

def construct():
    n, m, p = DIMENSIONS # 3, 5, 6
    
    # We will construct a decomposition of rank 85.
    # Split M_{3,5,6} into M_{1,5,6} + M_{2,5,6}.
    # M_{1,5,6} is schoolbook rank 30.
    # M_{2,5,6} is split into M_{2,2,6} + M_{2,3,6}.
    # M_{2,2,6} is 3 * M_{2,2,2} (Strassen) = 21.
    # M_{2,3,6} is 2 * M_{2,3,3}.
    # M_{2,3,3} is M_{2,1,3} (rank 6) + M_{2,2,3} (rank 11).
    # M_{2,2,3} is M_{2,2,1} (rank 4) + M_{2,2,2} (Strassen, rank 7).
    
    # Total rank = 30 + 21 + 2*(6 + 4 + 7) = 30 + 21 + 2*17 = 30 + 21 + 34 = 85.
    
    # We need to build the U, V, W matrices.
    # This requires careful index mapping.
    
    # Let's use a helper to append blocks.
    # Since the structure is complex, we will build the full 85-rank matrices directly.
    
    R = 85
    U = [[0]*R for _ in range(15)] # 3*5
    V = [[0]*R for _ in range(30)] # 5*6
    W = [[0]*R for _ in range(18)] # 6*3
    
    col = 0
    
    # 1. M_{1,5,6} (A is row 0, B is all cols)
    # A indices: 0..4 (i=0)
    # B indices: 0..29 (j=0..4, k=0..5) -> j*p+k
    # W indices: k*n+i = k*3+0 = 0,3,6,9,12,15
    for j in range(5):
        for k in range(6):
            U[0*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 0][col] = 1
            col += 1
    # col is now 30
    
    # 2. M_{2,5,6}
    # A indices: 5..14 (i=1,2)
    # B indices: same
    # W indices: k*3+1, k*3+2
    
    # 2a. M_{2,2,6} (j=0,1)
    # Split into 3 blocks of M_{2,2,2} for k=0,1; 2,3; 4,5
    for block in range(3):
        k_start = block * 2
        # Strassen 2x2x2
        # A indices: i=1,2; j=0,1 -> 1*5+0=5, 1*5+1=6, 2*5+0=10, 2*5+1=11
        # B indices: j=0,1; k=k_start, k_start+1 -> 0*6+k_start, 0*6+k_start+1, 1*6+k_start, 1*6+k_start+1
        # W indices: k=k_start, k_start+1; i=1,2 -> k*3+1, k*3+2
        
        # Strassen U, V, W for 2x2x2
        sU, sV, sW, sR = _make_strassen_222()
        
        # Map indices
        a_idx = [5, 6, 10, 11]
        b_idx = [0*6+k_start, 0*6+k_start+1, 1*6+k_start, 1*6+k_start+1]
        w_idx = [k_start*3+1, k_start*3+2, (k_start+1)*3+1, (k_start+1)*3+2]
        
        for r in range(sR):
            for i in range(4):
                if sU[i][r] != 0:
                    U[a_idx[i]][col+r] = sU[i][r]
            for i in range(4):
                if sV[i][r] != 0:
                    V[b_idx[i]][col+r] = sV[i][r]
            for i in range(4):
                if sW[i][r] != 0:
                    W[w_idx[i]][col+r] = sW[i][r]
        col += sR
    # col is now 30 + 21 = 51
    
    # 2b. M_{2,3,6} (j=2,3,4)
    # Split into 2 blocks of M_{2,3,3} for k=0,1,2 and k=3,4,5
    for block in range(2):
        k_start = block * 3
        # M_{2,3,3} split into M_{2,1,3} (j=2) + M_{2,2,3} (j=3,4)
        
        # 2b-i. M_{2,1,3} (j=2)
        # Schoolbook rank 6
        # A: i=1,2; j=2 -> 1*5+2=7, 2*5+2=12
        # B: j=2; k=k_start..k_start+2 -> 2*6+k_start, 2*6+k_start+1, 2*6+k_start+2
        # W: k=k_start..k_start+2; i=1,2 -> k*3+1, k*3+2
        
        for k in range(3):
            for i in range(2):
                U[1*5+2][col] = 1 if i==0 else 0 # A[1,2]
                U[2*5+2][col] = 1 if i==1 else 0 # A[2,2]
                V[2*6+k_start+k][col] = 1
                W[(k_start+k)*3+1][col] = 1 if i==0 else 0
                W[(k_start+k)*3+2][col] = 1 if i==1 else 0
                col += 1
        # col += 6 (3 k's * 2 i's)
        
        # 2b-ii. M_{2,2,3} (j=3,4)
        # Split into M_{2,2,1} (k=k_start) + M_{2,2,2} (k=k_start+1, k_start+2)
        
        # M_{2,2,1}
        # Schoolbook rank 4
        # A: i=1,2; j=3,4 -> 8,9,13,14
        # B: j=3,4; k=k_start -> 3*6+k_start, 4*6+k_start
        # W: k=k_start; i=1,2 -> k_start*3+1, k_start*3+2
        
        for i in range(2):
            for j in range(2):
                U[1*5+3+j][col] = 1 if i==0 else 0
                U[2*5+3+j][col] = 1 if i==1 else 0
                V[(3+j)*6+k_start][col] = 1
                W[k_start*3+1][col] = 1 if i==0 else 0
                W[k_start*3+2][col] = 1 if i==1 else 0
                col += 1
        # col += 4
        
        # M_{2,2,2} (k=k_start+1, k_start+2)
        # Strassen rank 7
        sU, sV, sW, sR = _make_strassen_222()
        
        a_idx = [1*5+3, 1*5+4, 2*5+3, 2*5+4]
        b_idx = [3*6+k_start+1, 3*6+k_start+2, 4*6+k_start+1, 4*6+k_start+2]
        w_idx = [(k_start+1)*3+1, (k_start+1)*3+2, (k_start+2)*3+1, (k_start+2)*3+2]
        
        for r in range(sR):
            for i in range(4):
                if sU[i][r] != 0:
                    U[a_idx[i]][col+r] = sU[i][r]
            for i in range(4):
                if sV[i][r] != 0:
                    V[b_idx[i]][col+r] = sV[i][r]
            for i in range(4):
                if sW[i][r] != 0:
                    W[w_idx[i]][col+r] = sW[i][r]
        col += sR
    
    # Verify
    if _verify_real(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    U, V, W, r = _make_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
