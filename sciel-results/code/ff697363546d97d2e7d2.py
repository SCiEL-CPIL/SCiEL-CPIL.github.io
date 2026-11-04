"""
Program for matrix multiplication 2x4x8 tensor decomposition.
Strategy: Structured algebraic search for rank reduction using Strassen-like cancellations.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    
    Strategy:
    1. Acknowledge that the schoolbook rank is 64.
    2. Utilize the structure of 2x4x8 which can be seen as two 1x4x8 blocks or one 2x4x8 block.
    3. More effectively, view the 2x4 multiplication as operating on 2x2 blocks.
       Let A be 2x4, B be 4x8.
       Partition A into 1x2 blocks? No, standard Strassen applies to square matrices.
       
    However, we can use the known rank-56 decomposition for 2x4x8.
    The lower bound for 2x4x8 is max(2*4, 4*8, 8*2) = 32.
    Known results:
    - 2x2x2: Rank 7 (Strassen)
    - 2x4x8 can be decomposed using 2 copies of 2x2x2 and some linear combinations?
    
    Actually, a direct construction for rank 56 exists.
    Let's construct it by explicitly solving the linear constraints for a rank 56 decomposition
    using a greedy random search over the space of integer/half-integer coefficients,
    seeded with the structure of Strassen's algorithm applied to the 2x2x2 sub-problems
    embedded in the 2x4x8 tensor.
    
    For 2x4x8, we can view the indices i in {0,1}, j in {0..3}, k in {0..7}.
    We can split j into two parts of size 2, and k into four parts of size 2?
    Or j into one part of size 4?
    
    Let's try to construct a rank 56 solution directly.
    Rank 56 is 64 - 8. This suggests saving 8 multiplications.
    
    A known decomposition for 2x4x8 of rank 56 can be constructed by:
    Using Strassen on the 2x2x2 blocks?
    If we split A (2x4) into two 2x2 blocks A1, A2.
    Split B (4x8) into two 4x4 blocks B1, B2? No, dimensions don't match for direct Strassen.
    
    Let's use the "Coppersmith-Winograd" or similar asymptotic ideas? No, too complex.
    
    Let's look at the inspiration: "rank 56 using a Strassen-based block decomposition".
    This implies we should find the specific coefficients for a rank 56 decomposition.
    
    Since I cannot search the entire space in 150s, I will hardcode the known rank 56
    decomposition structure derived from Strassen's algorithm applied to the 2x2x2 case
    and extended to 2x4x8.
    
    Specifically, we can decompose the 2x4x8 multiplication into:
    - 2 independent 2x2x2 multiplications (Rank 7 each = 14)
    - Plus the "cross" terms handled by schoolbook?
    
    Actually, let's look at the tensor T_{i,j,k}.
    If we use Strassen on the i-dimension (size 2), we need to combine j and k.
    Strassen works on (A*B) where A, B are square.
    Here A is 2x4, B is 4x8.
    
    We can pad A to 4x4 and B to 4x8? No.
    
    Let's try a different structural approach:
    The rank of 2x4x8 is at most 56.
    We can construct it by taking the schoolbook rank 64 and finding 8 linear dependencies
    that allow us to eliminate 8 basis tensors.
    
    Instead of searching, I will provide a hardcoded rank 56 decomposition.
    This is a "different strategy" because it uses a pre-computed algebraic structure
    (Strassen-based) rather than the trivial schoolbook loop.
    
    The rank 56 decomposition for 2x4x8 can be constructed as follows:
    Split the 4 columns of A into two groups of 2: A = [A1, A2] where A1, A2 are 2x2.
    Split the 4 rows of B into two groups of 2: B = [B1; B2] where B1, B2 are 2x8.
    Wait, B is 4x8. So B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8.
    C = A1 * B_top + A2 * B_bottom.
    Both A1*B_top and A2*B_bottom are 2x8 multiplications.
    The rank of 2x2x8 multiplication is ?
    It is equivalent to 4 independent 2x2x2 multiplications? No.
    2x2x8: A is 2x2, B is 2x8.
    We can view B as 4 columns of 2x2.
    So it's 4 independent 2x2x2 multiplications.
    Rank of 2x2x2 is 7.
    So rank of 2x2x8 is 4 * 7 = 28.
    Total rank = 28 + 28 = 56.
    
    This is the key insight!
    2x4x8 = (2x2x8) + (2x2x8).
    Each 2x2x8 can be decomposed into 4 instances of 2x2x2.
    Each 2x2x2 uses Strassen (rank 7).
    Total rank = 4 * 7 + 4 * 7 = 56.
    
    Let's implement this explicitly.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 formulas for 2x2x2:
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
    
    # We need to map these to the 2x2x8 case.
    # A is 2x2 (indices i in {0,1}, j_local in {0,1})
    # B is 2x8 (indices j_local in {0,1}, k in {0..7})
    # C is 2x8 (indices i in {0,1}, k in {0..7})
    
    # We process 4 blocks of B, each 2x2.
    # For block t (0..3), B_t is B[:, 2t:2t+2].
    # This gives C_t = A * B_t.
    
    # Total rank is 4 blocks * 7 Strassen terms = 28 per 2x2x8 part.
    # We have two such parts (A1 and A2).
    # Total rank = 56.
    
    # Indices:
    # i: 0, 1 (rows of A)
    # j: 0, 1, 2, 3 (cols of A / rows of B)
    #   j=0,1 correspond to A1 (first 2 cols of A, first 2 rows of B)
    #   j=2,3 correspond to A2 (last 2 cols of A, last 2 rows of B)
    # k: 0..7 (cols of B / cols of C)
    #   k=0,1,2,3 correspond to B1 (first 4 cols of B) -> Wait, B is 4x8.
    #   B_top is rows 0,1 of B. B_bottom is rows 2,3 of B.
    #   A1 is cols 0,1 of A. A2 is cols 2,3 of A.
    
    # C = A1 @ B_top + A2 @ B_bottom
    # B_top is 2x8. B_bottom is 2x8.
    
    # Let's define the Strassen coefficients for a single 2x2x2 multiplication.
    # U_coeff[i, j, term]
    # V_coeff[j, k, term]
    # W_coeff[k, i, term]
    
    # Strassen terms:
    # 0: (A00+A11)(B00+B11)
    # 1: (A10+A11)B00
    # 2: A00(B10-B11)
    # 3: A11(B01-B00)
    # 4: (A00+A01)B11
    # 5: (A10-A00)(B00+B01)
    # 6: (A01-A11)(B10+B11)
    
    # Let's create the coefficient matrices for one 2x2x2 block.
    # U is 4x7 (i*2+j, term)
    # V is 4x7 (j*2+k, term)
    # W is 4x7 (k*2+i, term) -- Note: W[k*n+i, r] convention. Here n=2.
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    # U: A00(0,0)+1, A11(1,1)+1 -> U[0,0]=1, U[3,0]=1
    # V: B00(0,0)+1, B11(1,1)+1 -> V[0,0]=1, V[3,0]=1
    # W: C00(0,0)+1, C11(1,1)+1 -> W[0,0]=1, W[3,0]=1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # Term 1: (A10+A11)B00
    # U: A10(1,0)+1, A11(1,1)+1 -> U[2,1]=1, U[3,1]=1
    # V: B00(0,0)+1 -> V[0,1]=1
    # W: C10(0,1)+1 -> W[2,1]=1 (k=1, i=1 -> 1*2+1=3? No. W[k*n+i]. k=1, i=1 -> 3. Wait.
    # C10 is row 1, col 0. i=1, k=0. Index k*n+i = 0*2+1 = 1.
    # Let's re-verify W indexing: W[k*n+i, r].
    # C[i, k]. i is row, k is col.
    # Term 1 contributes to C10 (i=1, k=0). Index 0*2+1 = 1.
    W_strassen[1, 1] = 1
    
    # Term 2: A00(B10-B11)
    # U: A00(0,0)+1 -> U[0,2]=1
    # V: B10(1,0)+1, B11(1,1)-1 -> V[2,2]=1, V[3,2]=-1
    # W: C01(0,1)+1 -> i=0, k=1. Index 1*2+0 = 2.
    W_strassen[2, 2] = 1
    
    # Term 3: A11(B01-B00)
    # U: A11(1,1)+1 -> U[3,3]=1
    # V: B01(0,1)+1, B00(0,0)-1 -> V[1,3]=1, V[0,3]=-1
    # W: C11(1,1)+1 -> i=1, k=1. Index 1*2+1 = 3.
    W_strassen[3, 3] = 1
    
    # Term 4: (A00+A01)B11
    # U: A00(0,0)+1, A01(0,1)+1 -> U[0,4]=1, U[1,4]=1
    # V: B11(1,1)+1 -> V[3,4]=1
    # W: C00(0,0)-1, C10(1,0)-1? No.
    # C00 = M1 + M4 - M5 + M7. So M4 (Term 4) contributes +1 to C00?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # My Term 4 is M5: (A00+A01)B11.
    # M5 contributes to C01 (+1) and C00 (-1).
    # So W for Term 4:
    # C01 (i=0, k=1) -> Index 1*2+0 = 2. Coeff +1.
    # C00 (i=0, k=0) -> Index 0*2+0 = 0. Coeff -1.
    W_strassen[2, 4] = 1
    W_strassen[0, 4] = -1
    
    # Term 5: (A10-A00)(B00+B01) -> M6
    # U: A10(1,0)+1, A00(0,0)-1 -> U[2,5]=1, U[0,5]=-1
    # V: B00(0,0)+1, B01(0,1)+1 -> V[0,5]=1, V[1,5]=1
    # W: M6 contributes to C11 (+1).
    # C11 (i=1, k=1) -> Index 1*2+1 = 3.
    W_strassen[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> M7
    # U: A01(0,1)+1, A11(1,1)-1 -> U[1,6]=1, U[3,6]=-1
    # V: B10(1,0)+1, B11(1,1)+1 -> V[2,6]=1, V[3,6]=1
    # W: M7 contributes to C00 (+1).
    # C00 (i=0, k=0) -> Index 0.
    W_strassen[0, 6] = 1
    
    # Now we have U_strassen, V_strassen, W_strassen for a 2x2x2 multiplication.
    # We need to embed this into 2x4x8.
    
    # Case 1: A1 (cols 0,1 of A) * B_top (rows 0,1 of B)
    # A1 is 2x2. B_top is 2x8.
    # B_top consists of 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], etc.
    # Let's call these B_t for t=0..3.
    # For each t, we perform a 2x2x2 multiplication A1 * B_t.
    # This contributes to C[:, 2t:2t+2].
    
    # Case 2: A2 (cols 2,3 of A) * B_bottom (rows 2,3 of B)
    # A2 is 2x2. B_bottom is 2x8.
    # B_bottom consists of 4 blocks of 2x2: B_bottom[:, 0:2], etc.
    # Let's call these B'_t for t=0..3.
    # For each t, we perform a 2x2x2 multiplication A2 * B'_t.
    # This contributes to C[:, 2t:2t+2].
    
    # Total 8 blocks of 2x2x2.
    # Each uses 7 rank terms. Total 56.
    
    # Mapping indices:
    # U: 8x56 (n*m = 2*4 = 8)
    # V: 32x56 (m*p = 4*8 = 32)
    # W: 16x56 (p*n = 8*2 = 16)
    
    # For Block t (0..3) of Case 1 (A1 * B_top):
    # U indices: i in {0,1}, j in {0,1} (local). Global j = j_local.
    # V indices: j_local in {0,1} (local row of B_top). Global j = j_local.
    #            k_local in {0,1} (local col of B_t). Global k = 2*t + k_local.
    # W indices: k_local in {0,1}. Global k = 2*t + k_local.
    #            i in {0,1}.
    
    # For Block t (0..3) of Case 2 (A2 * B_bottom):
    # U indices: i in {0,1}, j in {2,3} (local). Global j = 2 + j_local.
    # V indices: j_local in {0,1} (local row of B_bottom). Global j = 2 + j_local.
    #            k_local in {0,1}. Global k = 2*t + k_local.
    # W indices: k_local in {0,1}. Global k = 2*t + k_local.
    #            i in {0,1}.
    
    # Let's fill the matrices.
    
    for t in range(4):
        # Case 1: A1 * B_top_t
        base_rank = t * 7
        for term in range(7):
            r = base_rank + term
            
            # U
            for i in range(2):
                for j_local in range(2):
                    u_val = U_strassen[i*2+j_local, term]
                    if u_val != 0:
                        global_j = j_local
                        U[i*4 + global_j, r] = u_val
                        
            # V
            for j_local in range(2):
                for k_local in range(2):
                    v_val = V_strassen[j_local*2+k_local, term]
                    if v_val != 0:
                        global_j = j_local
                        global_k = 2*t + k_local
                        V[global_j*8 + global_k, r] = v_val
                        
            # W
            for k_local in range(2):
                for i in range(2):
                    w_val = W_strassen[k_local*2+i, term]
                    if w_val != 0:
                        global_k = 2*t + k_local
                        W[global_k*2 + i, r] = w_val
                        
        # Case 2: A2 * B_bottom_t
        base_rank = 28 + t * 7
        for term in range(7):
            r = base_rank + term
            
            # U
            for i in range(2):
                for j_local in range(2):
                    u_val = U_strassen[i*2+j_local, term]
                    if u_val != 0:
                        global_j = 2 + j_local
                        U[i*4 + global_j, r] = u_val
                        
            # V
            for j_local in range(2):
                for k_local in range(2):
                    v_val = V_strassen[j_local*2+k_local, term]
                    if v_val != 0:
                        global_j = 2 + j_local
                        global_k = 2*t + k_local
                        V[global_j*8 + global_k, r] = v_val
                        
            # W
            for k_local in range(2):
                for i in range(2):
                    w_val = W_strassen[k_local*2+i, term]
                    if w_val != 0:
                        global_k = 2*t + k_local
                        W[global_k*2 + i, r] = w_val
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
