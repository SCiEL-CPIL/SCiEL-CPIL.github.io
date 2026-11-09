"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, max_samples=50, tol=1e-6):
    """Verify that the decomposition correctly reconstructs matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Reshape W to (p, n, r) with W[k, i, r] corresponding to index k*n+i
    W_mat = W.reshape(p, n, r)
    
    for _ in range(max_samples):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute via decomposition: C[i,k] = sum_r (sum_{i',j} U[i',j,r]*A[i',j]) * (sum_{j',k'} V[j',k',r]*B[j',k']) * W[k,i,r]
        # Wait, that's not right. Let me think again.
        # C[i,k] = sum_r (sum_{j} U[i*m+j, r] * A[i,j]) ... no.
        # Actually the standard form: for each rank r, we have scalar M_r = <U_r, A> * <V_r, B>
        # where U_r is n x m, V_r is m x p. Then C += M_r * W_r where W_r is n x p (or p x n transposed).
        
        # U[i*m + j, r] -> U_r[i, j]
        # V[j*p + k, r] -> V_r[j, k]  
        # W[k*n + i, r] -> W_r[i, k] (since index k*n+i corresponds to output C[i,k])
        
        # M_r = sum_{i,j} U[i*m+j, r] * A[i,j]  -- wait no, U_r should contract with A
        # Actually: M_r = sum_{i,j} U_r[i,j] * A[i,j] is the trace of U_r^T * A = <U_r, A>
        # Similarly for V.
        
        # Let me just do it directly with matrix operations:
        # U_mat[r, i, j] = U[i*m+j, r]
        # V_mat[r, j, k] = V[j*p+k, r]
        # W_mat[r, i, k] = W[k*n+i, r]
        
        break  # We'll do vectorized below
    
    # Vectorized verification
    U_mat = U.reshape(r, n, m)  # U_mat[r, i, j]
    V_mat = V.reshape(r, m, p)  # V_mat[r, j, k]
    W_mat = W.reshape(r, p, n)  # W_mat[r, k, i] -> output C[i,k] gets W_mat[r, k, i]
    
    for _ in range(max_samples):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # M_r = sum_{i,j} U_mat[r,i,j] * A[i,j] = trace(U_r^T * A)
        M = np.einsum('rij,ij->r', U_mat, A) * np.einsum('rjk,jk->r', V_mat, B)
        
        # C[i,k] = sum_r M_r * W_mat[r, k, i]
        C_comp = np.einsum('r,rki->ik', M, W_mat)
        
        if np.max(np.abs(C_comp - C_true)) > tol:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 120
    
    # Try to build a rank-93 decomposition:
    # Split into 4x4x4 (columns 0-3) + 4x4x3 (columns 4-6)
    # For 4x4x4: use Strassen-based rank-49
    # For 4x4x3: use Strassen(2x2x2, rank 7) + schoolbook approach, target rank 44
    
    # ===== Part 1: 4x4x4 Strassen decomposition (rank 49) =====
    # Standard Strassen for 4x4: split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B with 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # But each 2x2x2 product itself has rank 7 (Strassen) or 8 (schoolbook).
    # For 4x4x4 with 7 top-level Strassen terms, each of which is a 2x2x2 multiply (rank 7):
    # Total rank = 7 * 7 = 49
    
    # Let me build this carefully.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (16, rank) for the 4x4x4 part
    # W has shape (p*n, rank) = (16, rank) for the 4x4x4 part
    
    # Actually, let me use a simpler approach: direct Strassen for 4x4x4.
    # The 4x4x4 tensor can be decomposed using the standard 7-product Strassen at the block level,
    # and each 2x2x2 sub-tensor uses rank-7 Strassen.
    
    # Let me define the 2x2x7 Strassen factors for a single 2x2x2 multiply.
    # For 2x2x2: A(2x2) * B(2x2) = C(2x2)
    # U_2x2 shape (4, 7), V_2x2 shape (4, 7), W_2x2 shape (4, 7)
    # A indexed as [i*2+j], B as [j*2+k], C as [k*2+i] (trace convention W[k*n+i])
    
    # Strassen 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    # A indices: a11=0, a12=1, a21=2, a22=3 (row-major: [i*2+j])
    # B indices: b11=0, b12=1, b21=2, b22=3 (row-major: [j*2+k])
    # C indices (trace): c11 -> k=0,i=0 -> idx 0; c12 -> k=1,i=0 -> idx 1; c21 -> k=0,i=1 -> idx 2; c22 -> k=1,i=1 -> idx 3
    # So W[k*2+i]: W[0]=c11 coeff, W[1]=c12 coeff, W[2]=c21 coeff, W[3]=c22 coeff
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22): U=[1,0,0,1], V=[1,1,0,0], W: c11+=1, c22+=1 -> W=[1,0,0,1]
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[1,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2 = (a21+a22)b11: U=[0,0,1,1], V=[1,0,0,0], W: c21+=1, c22-=1 -> W=[0,0,1,-1]
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # M3 = a11(b12-b22): U=[1,0,0,0], V=[0,1,-1,0], W: c12+=1, c22+=1 -> W=[0,1,0,1]
    U2[0,2]=1
    V2[1,2]=1; V2[2,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # M4 = a22(b21-b11): U=[0,0,0,1], V=[-1,0,1,0], W: c11+=1, c21+=1 -> W=[1,0,1,0]
    U2[3,3]=1
    V2[0,3]=-1; V2[2,3]=1
    W2[0,3]=1; W2[2,3]=1
    
    # M5 = (a11+a12)b22: U=[1,1,0,0], V=[0,0,0,1], W: c11-=1, c12+=1 -> W=[-1,1,0,0]
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # M6 = (a21-a11)(b11+b12): U=[-1,0,1,0], V=[1,1,0,0], W: c22+=1 -> W=[0,0,0,1]
    U2[0,5]=-1; U2[2,5]=1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7 = (a12-a22)(b21+b22): U=[0,1,0,-1], V=[0,0,1,1], W: c11+=1 -> W=[1,0,0,0]
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # Now build 4x4x4 using 7 top-level Strassen combinations, each using a 2x2x2 multiply (rank 7)
    # A 4x4 split into 2x2 blocks:
    # A = [[A11, A12],[A21, A22]] where each is 2x2
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # A index: i*4+j. A11: i=0,1,j=0,1 -> idx 0,1,4,5; A12: i=0,1,j=2,3 -> idx 2,3,6,7
    # A21: i=2,3,j=0,1 -> idx 8,9,12,13; A22: i=2,3,j=2,3 -> idx 10,11,14,15
    
    # B = [[B11, B12],[B21, B22]]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], etc.
    # B index: j*4+k. B11: j=0,1,k=0,1 -> idx 0,1,4,5; B12: j=0,1,k=2,3 -> idx 2,3,6,7
    # B21: j=2,3,k=0,1 -> idx 8,9,12,13; B22: j=2,3,k=2,3 -> idx 10,11,14,15
    
    # C = A*B, C[i,k]. C11 = A11*B11+A12*B21, C12=A11*B12+A12*B22, C21=A21*B11+A22*B21, C22=A21*B12+A22*B22
    # C trace index: k*4+i
    
    # Top-level Strassen:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    # M2 = (A21+A22)B11 -> contributes to C21 and C22
    # M3 = A11(B12-B22) -> contributes to C12 and C22
    # M4 = A22(B21-B11) -> contributes to C11 and C21
    # M5 = (A11+A12)B22 -> contributes to C11 and C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # Each M_t is a 2x2x2 multiply. Let's enumerate them.
    # For each top-level product M_t, we need to determine the effective 2x2 A_factor and B_factor.
    
    # Let me define helper: for a 2x2x2 multiply with A_factor (2x2) and B_factor (2x2),
    # the U, V, W contributions are scaled versions of U2, V2, W2.
    # A_factor[i,j] corresponds to A[row_start+i, col_start+j]
    # B_factor[j,k] corresponds to B[row_start_b+j, col_start_b+k]
    # Output C_out[i,k] contributes to C[row_start_c+i, col_start_c+k]
    
    # This is getting complex. Let me just build it programmatically.
    
    # Define the 7 top-level Strassen terms for 4x4:
    # Each term: (linear combo of A blocks) * (linear combo of B blocks)
    # where A blocks are 2x2 submatrices of A, B blocks are 2x2 submatrices of B.
    
    # A block indices: (row_block, col_block) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A block (rb, cb): rows 2*rb to 2*rb+1, cols 2*cb to 2*cb+1
    # A element A[i,j] where i=2*rb+ii, j=2*cb+ij -> global idx i*4+j
    
    # B block indices: (row_block, col_block) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B block (rb, cb): rows 2*rb to 2*rb+1, cols 2*cb to 2*cb+1
    # B element B[j,k] where j=2*rb+jj, k=2*cb+jk -> global idx j*4+k
    
    # C block (rb, cb): rows 2*rb to 2*rb+1, cols 2*cb to 2*cb+1
    # C element C[i,k] where i=2*rb+ii, k=2*cb+jk -> trace idx k*4+i
    
    # Top-level Strassen for 4x4 (using 2x2 blocks):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each M_t is a 2x2 * 2x2 = 2x2 multiply, decomposable with rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Let me define for each top-level term t:
    # a_coeffs: dict mapping A_block (rb,cb) to coefficient
    # b_coeffs: dict mapping B_block (rb,cb) to coefficient
    # c_contrib: dict mapping C_block (rb,cb) to coefficient
    
    top_terms = [
        # M1
        {'a': {(0,0):1, (1,1):1}, 'b': {(0,0):1, (1,1):1}, 'c': {(0,0):1, (1,1):1}},
        # M2
        {'a': {(1,0):1, (1,1):1}, 'b': {(0,0):1}, 'c': {(1,0):1, (1,1):-1}},
        # M3
        {'a': {(0,0):1}, 'b': {(0,1):1, (1,1):-1}, 'c': {(0,1):1, (1,1):1}},
        # M4
        {'a': {(1,1):1}, 'b': {(1,0):1, (0,0):-1}, 'c': {(0,0):1, (1,0):1}},
        # M5
        {'a': {(0,0):1, (0,1):1}, 'b': {(1,1):1}, 'c': {(0,0):-1, (0,1):1}},
        # M6
        {'a': {(1,0):1, (0,0):-1}, 'b': {(0,0):1, (0,1):1}, 'c': {(1,1):1}},
        # M7
        {'a': {(0,1):1, (1,1):-1}, 'b': {(1,0):1, (1,1):1}, 'c': {(0,0):1}},
    ]
    
    rank_44 = 49
    U44 = np.zeros((16, rank_44))
    V44 = np.zeros((16, rank_44))
    W44 = np.zeros((16, rank_44))  # 4x4 output: trace idx k*4+i
    
    col = 0
    for t in range(7):
        a_coeffs = top_terms[t]['a']
        b_coeffs = top_terms[t]['b']
        c_coeffs = top_terms[t]['c']
        
        for s in range(7):  # 7 Strassen factors within this 2x2x2
            # U contribution: for each A block (rb,cb) with coeff ca, and each local (ii,ij) with U2[ii*2+ij, s]:
            #   global A index = (2*rb+ii)*4 + (2*cb+ij)
            #   U44[global_A_idx, col] += ca * U2[ii*2+ij, s]
            for (rb, cb), ca in a_coeffs.items():
                for ii in range(2):
                    for ij in range(2):
                        u2val = U2[ii*2+ij, s]
                        if u2val != 0:
                            ga = (2*rb+ii)*4 + (2*cb+ij)
                            U44[ga, col] += ca * u2val
            
            # V contribution: for each B block (rb,cb) with coeff cb_coeff, and each local (jj,jk) with V2[jj*2+jk, s]:
            #   global B index = (2*rb+jj)*4 + (2*cb+jk)
            for (rb, cb), cb_coeff in b_coeffs.items():
                for jj in range(2):
                    for jk in range(2):
                        v2val = V2[jj*2+jk, s]
                        if v2val != 0:
                            gb = (2*rb+jj)*4 + (2*cb+jk)
                            V44[gb, col] += cb_coeff * v2val
            
            # W contribution: for each C block (rb,cb) with coeff cc, and each local (ii,jk) with W2[jk*2+ii, s]:
            #   Wait, W2 index is k*2+i for 2x2. Local C[ii, jk] -> W2[jk*2+ii, s]
            #   global C trace index = (2*cb+jk)*4 + (2*rb+ii)
            for (rb, cb), cc in c_coeffs.items():
                for ii in range(2):
                    for jk in range(2):
                        w2val = W2[jk*2+ii, s]
                        if w2val != 0:
                            gc = (2*cb+jk)*4 + (2*rb+ii)
                            W44[gc, col] += cc * w2val
            
            col += 1
    
    # Verify 4x4x4 part
    assert _verify_decomposition(U44, V44, W44, 4, 4, 4, max_samples=20), "4x4x4 verification failed"
    
    # ===== Part 2: 4x4x3 decomposition (target rank 44) =====
    # Split into two 2x4x3 subproblems? Or use a different approach.
    # Actually, let's split C into columns 0-1 (4x2) and column 2 (4x1)?
    # Or split A into rows 0-1 and 2-3?
    
    # Alternative: split the 4x4x3 into two 2x4x3 problems?
    # A(4x4) * B(4x3) = C(4x3)
    # Split A into A_top(2x4) and A_bot(2x4)
    # C_top = A_top * B (2x3), C_bot = A_bot * B (2x3)
    # Each 2x4x3: can we decompose efficiently?
    # 2x4x3 schoolbook = 2*4*3 = 24. Two of them = 48. Not better.
    
    # Let me try: split B into B_left(4x2) and B_right(4x1)
    # C_left = A * B_left (4x2), C_right = A * B_right (4x1)
    # 4x4x2 schoolbook = 32, 4x4x1 = 16. Total 48. Same.
    
    # Let's use Strassen for the 4x4x2 part (treat as 4x4x4 but ignore 2 columns).
    # Actually, for 4x4x2: split into 2x2 blocks. A is 4x4, B is 4x2 (two columns).
    # This is trickier. Let me just use a rank-44 construction.
    
    # Strategy for 4x4x3 rank 44:
    # Use the 4x4x4 Strassen decomposition (rank 49) but only keep contributions to columns 0-2.
    # That would give rank 49 but with some W entries zero. Still rank 49.
    
    # Better: dedicated 4x4x3.
    # Let me try: 4x4x3 = (2x4x3) + (2x4x3) where we split A rows.
    # For each 2x4x3: split B into B1(4x1) and B2(4x2)? 
    # 2x4x1 = 8, 2x4x2 = 16. Total per half = 24, two halves = 48. Still 48.
    
    # Let me try a different split for 4x4x3:
    # A(4x4), B(4x3). 
    # Split A into 2x2 blocks: A11,A12,A21,A22 (each 2x2)
    # Split B into 2x2 and 2x1: B is 4x3.
    # B = [[B11(2x2), B12(2x1)], [B21(2x2), B22(2x1)]]
    # C = A*B = 4x3.
    # C = [[C11(2x2), C12(2x1)], [C21(2x2), C22(2x1)]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This has 8 products of size 2x2x2 or 2x2x1.
    # 2x2x2: rank 7 each. 2x2x1: rank 4 each (schoolbook) or can we do better?
    # 2x2x1: A(2x2)*B(2x1) = C(2x1). c1 = a11*b1 + a12*b2, c2 = a21*b1 + a22*b2.
    # Rank: at least 4 (schoolbook). Can it be 3? Probably not for exact.
    # So 4 * 7 + 4 * 4 = 28 + 16 = 44. 
    
    # Let me build this.
    # 2x2x1 schoolbook:
    # c1 = a11*b1 + a12*b2
    # c2 = a21*b1 + a22*b2
    # A indices: a11=0, a12=1, a21=2, a22=3
    # B indices: b1=0, b2=1
    # C trace: c1 -> k=0,i=0 -> idx 0; c2 -> k=1,i=0 -> idx 1. Wait, C is 2x1, so n=2,p=1.
    # C[i,k] with i=0,1 and k=0. Trace idx = k*n+i = k*2+i.
    # c1 = C[0,0] -> idx 0*2+0 = 0
    # c2 = C[1,0] -> idx 0*2+1 = 1
    
    U21 = np.zeros((4, 4))  # A is 2x2 -> 4 elements
    V21 = np.zeros((2, 4))  # B is 2x1 -> 2 elements
    W21 = np.zeros((2, 4))  # C is 1x2 in trace: p*n = 1*2 = 2
    
    # r=0: a11*b1 -> c1
    U21[0,0]=1; V21[0,0]=1; W21[0,0]=1
    # r=1: a12*b2 -> c1
    U21[1,1]=1; V21[1,1]=1; W21[0,1]=1
    # r=2: a21*b1 -> c2
    U21[2,2]=1; V21[0,2]=1; W21[1,2]=1
    # r=3: a22*b2 -> c2
    U21[3,3]=1; V21[1,3]=1; W21[1,3]=1
    
    # Now assemble 4x4x3:
    # Products:
    # P1 = A11 * B11 (2x2 * 2x2 = 2x2), rank 7
    # P2 = A12 * B21 (2x2 * 2x2 = 2x2), rank 7
    # P3 = A11 * B12 (2x2 * 2x1 = 2x1), rank 4
    # P4 = A12 * B22 (2x2 * 2x1 = 2x1), rank 4
    # P5 = A21 * B11 (2x2 * 2x2 = 2x2), rank 7
    # P6 = A22 * B21 (2x2 * 2x2 = 2x2), rank 7
    # P7 = A21 * B12 (2x2 * 2x1 = 2x1), rank 4
    # P8 = A22 * B22 (2x2 * 2x1 = 2x1), rank 4
    # Total rank = 4*7 + 4*4 = 44
    
    # C11 = P1 + P2 (2x2)
    # C12 = P3 + P4 (2x1)
    # C21 = P5 + P6 (2x2)
    # C22 = P7 + P8 (2x1)
    
    rank_43 = 44
    U43 = np.zeros((16, rank_43))
    V43 = np.zeros((12, rank_43))  # m*p = 4*3 = 12
    W43 = np.zeros((12, rank_43))  # p*n = 3*4 = 12
    
    col = 0
    
    # Helper to add a 2x2x2 product (rank 7)
    def add_222(U43, V43, W43, a_rb, a_cb, b_rb, b_cb, c_rb, c_cb, col):
        """Add 2x2x2 product: A[2a_rb:2a_rb+2, 2a_cb:2a_cb+2] * B[2b_rb:2b_rb+2, 2b_cb:2b_cb+2] -> C[2c_rb:2c_rb+2, 2c_cb:2c_cb+2]"""
        for s in range(7):
            for ii in range(2):
                for ij in range(2):
                    uval = U2[ii*2+ij, s]
                    if uval != 0:
                        ga = (2*a_rb+ii)*4 + (2*a_cb+ij)
                        U43[ga, col+s] += uval
            for jj in range(2):
                for jk in range(2):
                    vval = V2[jj*2+jk, s]
                    if vval != 0:
                        gb = (2*b_rb+jj)*4 + (2*b_cb+jk)
                        V43[gb, col+s] += vval
            for ii in range(2):
                for jk in range(2):
                    wval = W2[jk*2+ii, s]
                    if wval != 0:
                        gc = (2*c_cb+jk)*4 + (2*c_rb+ii)  # trace idx = k*4+i
                        W43[gc, col+s] += wval
        return col + 7
    
    # Helper to add a 2x2x1 product (rank 4)
    def add_221(U43, V43, W43, a_rb, a_cb, b_rb, b_cb, c_rb, c_cb, col):
        """Add 2x2x1 product: A[2a_rb:2a_rb+2, 2a_cb:2a_cb+2] * B[2b_rb:2b_rb+2, 2b_cb] -> C[2c_rb:2c_rb+2, 2c_cb]"""
        for s in range(4):
            for ii in range(2):
                for ij in range(2):
                    uval = U21[ii*2+ij, s]
                    if uval != 0:
                        ga = (2*a_rb+ii)*4 + (2*a_cb+ij)
                        U43[ga, col+s] += uval
            for jj in range(2):
                vval = V21[jj, s]
                if vval != 0:
                    gb = (2*b_rb+jj)*4 + (2*b_cb)
                    V43[gb, col+s] += vval
            for ii in range(2):
                wval = W21[ii, s]  # C[i,0] -> trace idx = 0*4+i = i
                if wval != 0:
                    gc = (2*c_cb)*4 + (2*c_rb+ii)  # trace idx = k*4+i, k=2*c_cb, i=2*c_rb+ii
                    W43[gc, col+s] += wval
        return col + 4
    
    # P1 = A11 * B11 -> C11
    col = add_222(U43, V43, W43, 0, 0, 0, 0, 0, 0, col)
    # P2 = A12 * B21 -> C11
    col = add_222(U43, V43, W43, 0, 1, 1, 0, 0, 0, col)
    # P3 = A11 * B12 -> C12
    col = add_221(U43, V43, W43, 0, 0, 0, 1, 0, 1, col)
    # P4 = A12 * B22 -> C12
    col = add_221(U43, V43, W43, 0, 1, 1, 1, 0, 1, col)
    # P5 = A21 * B11 -> C21
    col = add_222(U43, V43, W43, 1, 0, 0, 0, 1, 0, col)
    # P6 = A22 * B21 -> C21
    col = add_222(U43, V43, W43, 1, 1, 1, 0, 1, 0, col)
    # P7 = A21 * B12 -> C22
    col = add_221(U43, V43, W43, 1, 0, 0, 1, 1, 1, col)
    # P8 = A22 * B22 -> C22
    col = add_221(U43, V43, W43, 1, 1, 1, 1, 1, 1, col)
    
    assert col == rank_43, f"Expected rank 44, got {col}"
    
    # Verify 4x4x3 part
    assert _verify_decomposition(U43, V43, W43, 4, 4, 3, max_samples=20), "4x4x3 verification failed"
    
    # ===== Combine 4x4x4 and 4x4x3 =====
    # For 4x4x4: U has shape (16, 49), V has shape (16, 49) [m*p for p=4 is 16], W has shape (16, 49)
    # For 4x4x3: U has shape (16, 44), V has shape (12, 44) [m*p for p=3 is 12], W has shape (12, 44)
    
    # The full problem: A(4x4) * B(4x7) = C(4x7)
    # U_full: (16, 49+44) = (16, 93)
    # V_full: (28, 93) where first 16 rows are for B(4x4) cols 0-3, next 12 rows for B(4x3) cols 4-6
    # W_full: (28, 93) where first 16 rows are for C(4x4) cols 0-3, next 12 rows for C(4x3) cols 4-6
    
    total_rank = 93
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((28, total_rank))  # m*p = 4*7 = 28
    W_full = np.zeros((28, total_rank))  # p*n = 7*4 = 28
    
    # Copy 4x4x4 part (columns 0-3 of B and C)
    U_full[:, :49] = U44
    V_full[:16, :49] = V44
    W_full[:16, :49] = W44
    
    # Copy 4x4x3 part (columns 4-6 of B and C)
    U_full[:, 49:] = U43
    V_full[16:28, 49:] = V43
    W_full[16:28, 49:] = W43
    
    # Full verification
    if time.time() < deadline - 30:
        assert _verify_decomposition(U_full, V_full, W_full, 4, 4, 7, max_samples=50), "Full 4x4x7 verification failed"
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
