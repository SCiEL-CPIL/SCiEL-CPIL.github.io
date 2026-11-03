"""Rewritten program for 4x4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]


def _build_strassen_222():
    """Return U, V, W for 2x2x2 Strassen (rank 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2x2: n=2, m=2, p=2
    U shape (4,7), V shape (4,7), W shape (4,7)
    """
    # Strassen formulas for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 + M8
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-h)
    # M5 = (a+b)h
    # M6 = (c-d)(e+f)
    # M7 = (a-c)(g+h)
    # M8 = (b-d)(f+h)
    #
    # U[i*2+j, c] = coefficient of A[i,j] in M_c
    # V[j*2+k, c] = coefficient of B[j,k] in M_c
    # W[k*2+i, c] = coefficient of M_c in C[k,i]  (note: output index is k*n+i)
    
    # M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    # M2 = (c+d)*e = c*e + d*e
    # M3 = a*(f-h) = a*f - a*h
    # M4 = d*(g-h) = d*g - d*h
    # M5 = (a+b)*h = a*h + b*h
    # M6 = (c-d)*(e+f) = c*e + c*f - d*e - d*f
    # M7 = (a-c)*(g+h) = a*g + a*h - c*g - c*h
    # M8 = (b-d)*(f+h) = b*f + b*h - d*f - d*h
    
    U = np.zeros((4, 8), dtype=int)  # will use 8 slots but only need 7
    V = np.zeros((4, 8), dtype=int)
    W = np.zeros((4, 8), dtype=int)
    
    # Indexing: A[i,j] -> U[i*2+j, c], B[j,k] -> V[j*2+k, c]
    # C output: W[k*2+i, c] where C = A*B, so C[i,k] uses... wait
    # Convention says W[k*n+i, c]. So output element at position (k*n+i) corresponds to
    # the (i-th row, k-th col) of C? Let me re-read: "output factor follows the official 
    # trace convention W[k*n+i,r]"
    # So W has p*n rows. Row index = k*n + i. This means W[k*n+i, r] is the coefficient
    # for the (i, k) element of the output matrix C.
    # Actually in matrix mult C = A * B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = delta (it's 1 if we're computing that product)
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So W[k*n+i, c] multiplies into the (i,k) output position.
    
    # M1: a*e + a*h + d*e + d*h
    # a=A[0,0] -> U[0,0]=1; e=B[0,0] -> V[0,0]=1; h=B[0,1] -> V[1,0]=1; d=A[1,1] -> U[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    # C[0,0] += M1, C[1,1] += M1
    W[0*2+0, 0] = 1  # k=0,i=0 -> C[0,0]
    W[1*2+1, 0] = 1  # k=1,i=1 -> C[1,1]
    
    # M2: c*e + d*e
    U[2, 1] = 1; U[3, 1] = 1  # c=A[1,0], d=A[1,1]
    V[0, 1] = 1  # e=B[0,0]
    # C[1,0] += M2
    W[0*2+1, 1] = 1  # k=0,i=1 -> C[1,0]
    
    # M3: a*f - a*h
    U[0, 2] = 1  # a=A[0,0]
    V[1, 2] = 1; V[1, 2] += 0  # f=B[0,1]... wait V[j*p+k], B[0,1] -> j=0,k=1 -> V[0*2+1,2]=V[1,2]
    V[1, 2] = 1  # B[0,1]
    V[1, 2] = 1  # hmm let me redo. B[j,k], j=0,k=1 -> index j*p+k = 0*2+1 = 1. So V[1,2]=1 for f.
    # -a*h: h=B[0,1] -> V[1,2] += -1? No wait, M3 = a*(f-h) = a*f - a*h
    # So U[0,2]=1 (for a), V[1,2]=1 (for f=B[0,1]), V[1,2]... 
    # Actually both f and h are B[0,1]? No! f=B[0,1], h=B[1,1].
    # B = [[e,f],[g,h]] so B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # f -> j=0,k=1 -> V[0*2+1, c] = V[1, c]
    # h -> j=1,k=1 -> V[1*2+1, c] = V[3, c]
    
    # Let me redo all carefully.
    # A[i,j]: i=0,1; j=0,1. Index in U: i*2+j
    #   A[0,0]=a -> U[0], A[0,1]=b -> U[1], A[1,0]=c -> U[2], A[1,1]=d -> U[3]
    # B[j,k]: j=0,1; k=0,1. Index in V: j*2+k
    #   B[0,0]=e -> V[0], B[0,1]=f -> V[1], B[1,0]=g -> V[2], B[1,1]=h -> V[3]
    # C[i,k]: output. Index in W: k*2+i
    #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (c+d)*e = c*e + d*e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    
    # M3 = a*(f-h) = a*f - a*h
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1  # h
    W[1, 2] = 1  # C[0,1]
    
    # M4 = d*(g-h) = d*g - d*h
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[3, 3] = -1  # h
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (a+b)*h = a*h + b*h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (c-d)*(e+f) = c*e + c*f - d*e - d*f
    U[2, 5] = 1  # c
    U[3, 5] = -1  # d
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[2, 5] = 1  # C[1,0]
    
    # M7 = (a-c)*(g+h) = a*g + a*h - c*g - c*h
    U[0, 6] = 1  # a
    U[2, 6] = -1  # c
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _build_schoolbook_221():
    """Return U, V, W for 2x2x1 schoolbook (rank 4).
    n=2, m=2, p=1. U shape (4,4), V shape (2,4), W shape (2,4).
    C[i,k] where k=0 only. C[i,0] = sum_j A[i,j]*B[j,0]
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j, col] = 1
            V[j*1+0, col] = 1
            W[0*2+i, col] = 1  # k=0, so index = 0*2+i = i
            col += 1
    return U, V, W


def _build_223_block():
    """Build a 2x2x3 block: C = A(2x2) * B(2x3).
    Decompose as Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = rank 11.
    
    The 2x2x3 problem: C[i,k] for i=0,1 and k=0,1,2.
    We can split B into B[:,0:2] and B[:,2:3].
    C[:,0:2] = A * B[:,0:2] -> 2x2x2 (Strassen rank 7)
    C[:,2] = A * B[:,2] -> 2x2x1 (schoolbook rank 4)
    """
    U_str, V_str, W_str = _build_strassen_222()
    U_sb, V_sb, W_sb = _build_schoolbook_221()
    
    # Combined: rank 11
    # U: (4, 11), V: (2*3=6, 11), W: (3*2=6, 11)
    U = np.zeros((4, 11), dtype=int)
    V = np.zeros((6, 11), dtype=int)
    W = np.zeros((6, 11), dtype=int)
    
    # Strassen part handles B columns 0,1 (k=0,1)
    U[:, :7] = U_str
    # V_str has shape (4,7) for B[j,k] with j=0,1; k=0,1. Index j*2+k
    # In the 2x2x3 problem, B has indices j*3+k where j=0,1; k=0,1,2
    # For k=0: j*3+0 = 3j. For k=1: j*3+1 = 3j+1.
    # V_str[j*2+k, c] maps to V[j*3+k, c] for k=0,1
    for j in range(2):
        for k in range(2):
            V[j*3+k, :7] = V_str[j*2+k, :7]
    
    # W_str has shape (4,7) for C[i,k] with k*2+i. k=0,1.
    # In 2x2x3, W index is k*2+i for k=0,1,2.
    # W_str[k*2+i, c] maps to W[k*2+i, c] for k=0,1
    W[:4, :7] = W_str  # W_str indices 0..3 correspond to k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
    # Actually W_str has 4 rows: k*2+i for k=0,1 and i=0,1
    # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    # In 2x2x3 W has 6 rows: k*2+i for k=0,1,2 and i=0,1
    # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3; k=2,i=0 -> 4; k=2,i=1 -> 5
    # So W_str rows 0-3 map directly to W rows 0-3.
    
    # Schoolbook part handles B column 2 (k=2)
    U[:, 7:] = U_sb
    # V_sb has shape (2,4) for B[j,0] with j=0,1. Index j*1+0 = j.
    # In 2x2x3, B[j,2] has index j*3+2 = 3j+2.
    for j in range(2):
        V[j*3+2, 7:] = V_sb[j, 7:]  # V_sb[j, c] for c=0..3 maps to columns 7..10
    
    # W_sb has shape (2,4) for C[i,0] with i=0,1. Index 0*2+i = i.
    # In 2x2x3, C[i,2] has index 2*2+i = 4+i.
    W[4:6, 7:] = W_sb  # W_sb[i, c] maps to W[4+i, c]
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 (it's the identity tensor for matmul)
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We only need to check that the bilinear map is correct.
    # For each (i,k) output, C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition gives: C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j]*B[j,k]
    # This should equal sum_j A[i,j]*B[j,k] for all i,j,k.
    
    # Check: for each c, the outer product U_c (vector of length n*m) * V_c (length m*p) * W_c (length p*n)
    # contributes to the tensor. The sum over c should give the matmul tensor.
    
    # The matmul tensor T[i,j,k] = delta_{i,j,k} is simply: for fixed i,j,k, T[i,j,k]=1 always
    # (since every triple (i,j,k) contributes to exactly one output C[i,k]).
    # Wait no. The tensor for matrix multiplication is T[i,j,k] = 1 for all valid i,j,k.
    # Because C[i,k] = sum_j A[i,j] * B[j,k], and the tensor is the structure tensor.
    
    # Actually the structure tensor of matmul is: T = sum_{i,j,k} e_i (in A*) tensor e_j (in B*) tensor e_k (in C*)
    # where the coefficient is 1 if the (i,j,k) term appears, which is always 1.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify this.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                if abs(computed - 1.0) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: decompose 4x4x7 into blocks.
    # Split A (4x4) into 2x2 blocks, B (4x7) into 2x7 blocks.
    # C (4x7) = A*B where A is 4x4, B is 4x7.
    # 
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where B00,B10 are 2x7, B01,B11 are 2x7
    # 
    # C[0:2, :] = A00*B00 + A01*B10  (2x7 = 2x2 * 2x7 + 2x2 * 2x7)
    # C[2:4, :] = A10*B00 + A11*B10  (2x7 = 2x2 * 2x7 + 2x2 * 2x7)
    #
    # Each 2x2 * 2x7 product can be decomposed as:
    #   2x2 * 2x7 = (2x2 * 2x4) + (2x2 * 2x3)
    # Or more directly, 2x2 * 2x7:
    #   Split B into 2x2, 2x2, 2x2, 2x1 blocks
    #   2x2*2x2 (rank 7) + 2x2*2x2 (rank 7) + 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = 25
    # 
    # Total: 4 blocks of (2x2 * 2x7) = 4 * 25 = 100. That's worse than 112.
    #
    # Alternative: 2x2 * 2x7 = (2x2 * 2x3) + (2x2 * 2x3) + (2x2 * 2x1)
    # = 11 + 11 + 4 = 26. Total: 4 * 26 = 104. Still worse.
    #
    # Let me think differently. 4x4 * 4x7:
    # Use Strassen-style for the 4x4 part:
    # 4x4 * 4x4 has rank 21 (Strassen). For 4x4 * 4x7, we can extend.
    #
    # Actually, let me try: split into 2x2x7 blocks.
    # 2x2 * 2x7: split B (2x7) into B1(2x3) + B2(2x3) + B3(2x1)
    # 2x2*2x3 (rank 11) + 2x2*2x3 (rank 11) + 2x2*2x1 (rank 4) = 26
    # 4 such blocks: 4 * 26 = 104
    #
    # Or: 2x2 * 2x7 = 2x2*2x4 + 2x2*2x3
    # 2x2*2x4 = 2x2*2x2 + 2x2*2x2 = 7+7 = 14
    # 2x2*2x3 = 11
    # Total per block: 25. 4 blocks: 100.
    #
    # Hmm, let me try a different splitting.
    # 4x4 * 4x7:
    # Split A into 4x2 and 4x2, B into 2x7 and 2x7.
    # C = A1*B1 + A2*B2 where A1,A2 are 4x2, B1,B2 are 2x7.
    # Each 4x2 * 2x7: this is 4*2*7 = 56 scalar mults schoolbook.
    # Using Strassen-like: 4x2 * 2x7.
    # 
    # Actually, the cleanest approach with known results:
    # 4x4*4x4 rank = 21 (Strassen)
    # 4x4*4x7: we can view this as 4x4*4x4 (rank 21) + 4x4*4x3
    # 4x4*4x3: split into 4x2*4x3 + 2x2*4x3... 
    # 
    # Let me just use the 2x2x3 building block approach.
    # 4x4 * 4x7:
    # Split rows of A: A_top (2x4), A_bot (2x4)
    # Split cols of B: B has 4 rows, 7 cols. Split rows: B_top (2x7), B_bot (2x7)
    # C_top = A_top * B_top + A_top * B_bot  (2x7 each product is 2x4*2x7... no)
    # 
    # Wait, A is 4x4, B is 4x7. C = A*B is 4x7.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # Block: C[0:2, :] = A[0:2,:] * B[:, :] (2x4 * 4x7 = 2x7)
    #         C[2:4, :] = A[2:4,:] * B[:, :] (2x4 * 4x7 = 2x7)
    # 
    # For C[0:2,:]: sum over j=0..3 of A[0:2,j] * B[j,:]
    # = A[0:2,0:2]*B[0:2,:] + A[0:2,2:4]*B[2:4,:]
    # = (2x2 * 2x7) + (2x2 * 2x7)
    # 
    # So total: 4 instances of 2x2 * 2x7.
    # 
    # For 2x2 * 2x7, I'll use: split the 2x7 into 2x3 + 2x3 + 2x1
    # 2x2*2x3 (rank 11) + 2x2*2x3 (rank 11) + 2x2*2x1 (rank 4) = 26
    # Total rank: 4 * 26 = 104
    #
    # Alternatively 2x2*2x7 = 2x2*2x4 + 2x2*2x3
    # 2x2*2x4 = 2x2*2x2 + 2x2*2x2 = 14
    # 2x2*2x3 = 11
    # Total: 25 per block, 100 overall.
    
    # Let me build the 100-rank decomposition.
    
    # For each 2x2*2x7 block, decompose as:
    #   2x2*2x2 (Strassen, rank 7) for B cols 0-1
    #   2x2*2x2 (Strassen, rank 7) for B cols 2-3  
    #   2x2*2x2 (Strassen, rank 7) for B cols 4-5
    #   2x2*2x1 (schoolbook, rank 4) for B col 6
    # Total: 25 per 2x2*2x7 block.
    
    U_block = np.zeros((4, 25), dtype=int)
    V_block = np.zeros((6, 25), dtype=int)  # 2*3... wait, 2x2 means B is 2x2, so V has 2*2=4 rows
    # Actually for 2x2*2x2: U is 4x7, V is 4x7, W is 4x7
    # For 2x2*2x1: U is 4x4, V is 2x4, W is 2x4
    
    # I need to combine these into a single 2x2*2x7 decomposition.
    # U: (4, 25)
    # V: (2*7=14, 25)  -- B is 2x7, so V has 14 rows
    # W: (7*2=14, 25)  -- C is 2x7, so W has 14 rows
    
    U_227 = np.zeros((4, 25), dtype=int)
    V_227 = np.zeros((14, 25), dtype=int)
    W_227 = np.zeros((14, 25), dtype=int)
    
    # Strassen for B cols 0-1
    U_s1, V_s1, W_s1 = _build_strassen_222()
    U_227[:, 0:7] = U_s1
    # V_s1[j*2+k, c] for j=0,1; k=0,1. B[j,k] in 2x7 has index j*7+k.
    for j in range(2):
        for k in range(2):
            V_227[j*7+k, 0:7] = V_s1[j*2+k, 0:7]
    # W_s1[k*2+i, c] for k=0,1. C[i,k] in 2x7 has index k*2+i.
    W_227[0:4, 0:7] = W_s1  # k=0,1; i=0,1 -> indices 0,1,2,3
    
    # Strassen for B cols 2-3
    U_s2, V_s2, W_s2 = _build_strassen_222()
    U_227[:, 7:14] = U_s2
    for j in range(2):
        for k in range(2):
            V_227[j*7+(2+k), 7:14] = V_s2[j*2+k, 7:14]
    # W for k=2,3: indices k*2+i = 4+i, 5+i, 6+i, 7+i
    W_227[4:8, 7:14] = W_s2
    
    # Strassen for B cols 4-5
    U_s3, V_s3, W_s3 = _build_strassen_222()
    U_227[:, 14:21] = U_s3
    for j in range(2):
        for k in range(2):
            V_227[j*7+(4+k), 14:21] = V_s3[j*2+k, 14:21]
    W_227[8:12, 14:21] = W_s3
    
    # Schoolbook for B col 6
    U_sb, V_sb, W_sb = _build_schoolbook_221()
    U_227[:, 21:25] = U_sb
    # V_sb[j, c] for j=0,1. B[j,6] has index j*7+6.
    for j in range(2):
        V_227[j*7+6, 21:25] = V_sb[j, 21:25]
    # W_sb[i, c] for i=0,1. C[i,6] has index 6*2+i = 12+i.
    W_227[12:14, 21:25] = W_sb
    
    # Now assemble the full 4x4*4x7 decomposition.
    # 4 blocks of 2x2*2x7.
    # Block (bi, bj) where bi=0,1 (row block of A), bj=0,1 (col block of A = row block of B)
    # A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, :]
    
    total_rank = 4 * 25  # 100
    U_full = np.zeros((16, total_rank), dtype=int)  # 4*4 = 16
    V_full = np.zeros((28, total_rank), dtype=int)  # 4*7 = 28
    W_full = np.zeros((28, total_rank), dtype=int)  # 7*4 = 28
    
    for bi in range(2):
        for bj in range(2):
            base = (bi*2 + bj) * 25
            # U: A[i,j] where i in [2*bi, 2*bi+2), j in [2*bj, 2*bj+2)
            # U index: i*4+j
            for ii in range(2):
                for jj in range(2):
                    i = 2*bi + ii
                    j = 2*bj + jj
                    U_full[i*4+j, base:base+25] = U_227[ii*2+jj, :]
            
            # V: B[j,k] where j in [2*bj, 2*bj+2), k in [0,7)
            # V index: j*7+k
            for jj in range(2):
                for k in range(7):
                    j = 2*bj + jj
                    V_full[j*7+k, base:base+25] = V_227[jj*7+k, :]
            
            # W: C[i,k] where i in [2*bi, 2*bi+2), k in [0,7)
            # W index: k*4+i
            for ii in range(2):
                for k in range(7):
                    i = 2*bi + ii
                    W_full[k*4+i, base:base+25] = W_227[k*2+ii, :]
    
    # Verify
    if not _verify_decomposition(n, m, p, U_full.tolist(), V_full.tolist(), W_full.tolist()):
        # Fall back to schoolbook
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
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
