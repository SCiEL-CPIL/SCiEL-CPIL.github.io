"""Matrix multiplication 4x5 x 5x6 — rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7 (indexed by i*2+j), V: 4x7 (indexed by j*2+k), W: 4x7 (indexed by k*2+i)
    Each column c represents one Strassen product.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # The tensor condition is:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{j,k}  (for the multiplication tensor)
    # More precisely, the bilinear form: sum_c W[k,i,c] * U[i,j,c] * V[l,k,c] = delta_{j,l}

    # c=0: M1 = (A00+A11)(B00+B11)
    # Contributes to C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1  # W[k*2+i]: k=0,i=0 -> idx 0; k=1,i=1 -> idx 3
    # Wait, W is indexed by k*2+i. Let me re-check.
    # W[k*2+i] where k is col index of B (which is row index of C), i is row index of A (col index of C)
    # Actually the convention is W[k*n+i] for n x m * m x p -> n x p
    # So W index = k*n + i where k is the column of the first matrix / row of second... 
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank), indexed by k*p... no wait.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m + j], V[j*p + k], W[k*n + i]
    
    # Let me redo with correct indexing:
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # For 2x2x2: U[2i+j], V[2j+k], W[2k+i]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # c=0: M1 = (A00+A11)(B00+B11)
    # A terms: A[0,0]=1, A[1,1]=1 -> U[0,0]=1, U[3,0]=1
    # B terms: B[0,0]=1, B[1,1]=1 -> V[0,0]=1, V[3,0]=1
    # C terms: C[0,0]=1, C[1,1]=1 -> W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # c=1: M2 = (A10+A11)B00
    # A terms: A[1,0]=1, A[1,1]=1 -> U[2,0]=1, U[3,0]=1
    # B terms: B[0,0]=1 -> V[0,0]=1
    # C terms: C[1,0]=1 -> W[1,0]=1  (k=0,i=1 -> W[0*2+1]=W[1])
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # W[2*0+1] = W[1]

    # c=2: M3 = A00(B10-B11)
    # A terms: A[0,0]=1 -> U[0,0]=1
    # B terms: B[1,0]=1, B[1,1]=-1 -> V[2,0]=1, V[3,0]=-1
    # C terms: C[0,0]=1, C[0,1]=1 -> W[0,0]=1, W[1,0]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1  # W[0]=C[0,0], W[1]=C[0,1]... wait
    # C[0,0]: k=0,i=0 -> W[0*2+0]=W[0]
    # C[0,1]: k=1,i=0 -> W[1*2+0]=W[2]
    # Hmm, C is 2x2: C[i,k] where i=row, k=col
    # W[k*n+i] = W[k*2+i]
    # C[0,0] -> i=0,k=0 -> W[0*2+0]=W[0]
    # C[0,1] -> i=0,k=1 -> W[1*2+0]=W[2]
    # C[1,0] -> i=1,k=0 -> W[0*2+1]=W[1]
    # C[1,1] -> i=1,k=1 -> W[1*2+1]=W[3]
    
    # Let me redo:
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # c=0: M1 = (A00+A11)(B00+B11)
    # C = M1 contributes: C[0,0]+=1, C[1,1]+=1 (from A00*B00 + A11*B11 cross terms cancel in Strassen)
    # Actually in Strassen, M1 is used in C00 and C11.
    # C00 = M1 - M5 + M7 + M4 (varies by convention)
    # Let me use the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For tensor decomposition, each "column" r corresponds to one scalar product.
    # The key identity is: for the multiplication tensor, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{j,k}
    # No wait, that's not right either. The matrix multiplication tensor T[i,j,k] = 1 if j=k else 0.
    # Actually T[i,j,k] represents the (i,k)-entry of A*B at row i, col k, summed over j.
    # The tensor is: T[i,j,k] = delta_{j,k} (identity in the middle dimension)
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, that's not standard. Let me think again.
    
    # Standard: A is n x m, B is m x p, C = A*B is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (for all i,j,k), and the bilinear map is:
    # f(A,B)[i,k] = sum_j A[i,j]*B[j,k]
    # A rank-1 decomposition: U[i,j] * V[j,k] * W[k,i] contributes U[i,j]*V[j,k] to C[i,k]
    # So we need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{j,k}... no.
    # We need: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_c (sum_j U_c[i,j]*A[i,j]) * (sum_j V_c[j,k]*B[j,k]) * W_c[k,i]
    # This is wrong. The correct formulation:
    # A decomposition of the multiplication tensor of rank r gives:
    # C[i,k] = sum_c W_c[k,i] * (sum_j U_c[i,j] * A[i,j]) * (sum_l V_c[l,k] * B[l,k])
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # W_c[k,i] * U_c[i,j] * V_c[j,k] summed over c = delta_{j,j}... 
    # Actually: sum_c W_c[k,i] * U_c[i,j] * V_c[j,k] = delta_{j,k}? No.
    # Let's verify: if A[i,j]=a, B[j,k]=b, then C[i,k] should be a*b.
    # sum_c W_c[k,i]*U_c[i,j]*V_c[j,k]*a*b = a*b*sum_c W_c[k,i]*U_c[i,j]*V_c[j,k]
    # So we need: sum_c W_c[k,i]*U_c[i,j]*V_c[j,k] = 1 for all i,j,k.
    # But that's not right because different (j) values should give different results.
    # The correct condition: the trilinear form sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] should equal
    # the multiplication tensor which is 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k],
    # the tensor entries are all 1).
    # Wait no. The tensor for matrix multiplication C_{ik} = sum_j A_{ij} B_{jk} has tensor
    # T_{ijk} = 1 for all i,j,k. So yes, we need sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1 for all i,j,k.
    
    # OK so for Strassen 2x2x2, I need to verify my decomposition gives sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k in {0,1}.
    
    # Let me just use the known correct Strassen decomposition.
    # The 7 terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M is a scalar. The tensor decomposition has one column per M.
    # For column c corresponding to M_c:
    # U_c[i,j] is the coefficient of A[i,j] in M_c
    # V_c[j,k] is the coefficient of B[j,k] in M_c
    # W_c[k,i] is the coefficient of M_c in C[i,k] (note: W is indexed by k,i not i,k)
    
    # c=0 (M1 = (A00+A11)(B00+B11)):
    # U: A00=1, A11=1 -> U[0,0]=1, U[1,1]=1
    # V: B00=1, B11=1 -> V[0,0]=1, V[1,1]=1
    # W: M1 appears in C00(+1) and C11(+1) -> W[0,0]=1 (k=0,i=0), W[1,1]=1 (k=1,i=1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # c=1 (M2 = (A10+A11)B00):
    # U: A10=1, A11=1 -> U[1,0]=1, U[1,1]=1
    # V: B00=1 -> V[0,0]=1
    # W: M2 appears in C10(+1) and C11(-1) -> W[0,1]=1 (k=0,i=1), W[1,1]=-1 (k=1,i=1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # c=2 (M3 = A00(B10-B11)):
    # U: A00=1 -> U[0,0]=1
    # V: B10=1, B11=-1 -> V[1,0]=1, V[1,1]=-1
    # W: M3 appears in C01(+1) and C11(+1) -> W[1,0]=1 (k=1,i=0), W[1,1]=1 (k=1,i=1)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # c=3 (M4 = A11(B01-B00)):
    # U: A11=1 -> U[1,1]=1
    # V: B01=1, B00=-1 -> V[0,1]=1, V[0,0]=-1
    # W: M4 appears in C00(+1) and C10(+1) -> W[0,0]=1 (k=0,i=0), W[0,1]=1 (k=0,i=1)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # c=4 (M5 = (A00+A01)B11):
    # U: A00=1, A01=1 -> U[0,0]=1, U[0,1]=1
    # V: B11=1 -> V[1,1]=1
    # W: M5 appears in C00(-1) and C01(+1) -> W[0,0]=-1 (k=0,i=0), W[1,0]=1 (k=1,i=0)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # c=5 (M6 = (A10-A00)(B00+B10)):
    # U: A10=1, A00=-1 -> U[1,0]=1, U[0,0]=-1
    # V: B00=1, B10=1 -> V[0,0]=1, V[1,0]=1
    # W: M6 appears in C11(+1) -> W[1,1]=1 (k=1,i=1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1

    # c=6 (M7 = (A01-A11)(B01+B11)):
    # U: A01=1, A11=-1 -> U[0,1]=1, U[1,1]=-1
    # V: B01=1, B11=1 -> V[0,1]=1, V[1,1]=1
    # W: M7 appears in C00(+1) -> W[0,0]=1 (k=0,i=0)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_strassen_4x4x4():
    """
    Build 4x4x4 Strassen decomposition.
    Uses 7 Strassen 2x2x2 blocks, each of rank 7, total rank 49.
    U: 16x49, V: 16x49, W: 16x49
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    # 4x4 Strassen: split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] each 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix product, computed via Strassen 2x2x2 (rank 7).
    # So for each of the 7 M's, we have 7 sub-products, total 49.
    
    # Block indices:
    # A00: i in [0,2), j in [0,2) -> local (0,0)
    # A01: i in [0,2), j in [2,4) -> local (0,1)  
    # A10: i in [2,4), j in [0,2) -> local (1,0)
    # A11: i in [2,4), j in [2,4) -> local (1,1)
    # Similarly for B: B[j,k], C: C[i,k]
    
    # For each Strassen term t (0-6), and each sub-term s (0-6):
    # The sub-term s of Strassen 2x2x2 for the t-th M product
    
    # M_t = (combination of A blocks) * (combination of B blocks)
    # Each combination gives coefficients for specific 2x2 sub-blocks
    
    # Let me define for each t, the A-block coefficients and B-block coefficients:
    # t=0 (M1): A blocks: A00(+1), A11(+1); B blocks: B00(+1), B11(+1)
    # t=1 (M2): A blocks: A10(+1), A11(+1); B blocks: B00(+1)
    # t=2 (M3): A blocks: A00(+1); B blocks: B10(+1), B11(-1)
    # t=3 (M4): A blocks: A11(+1); B blocks: B01(+1), B00(-1)
    # t=4 (M5): A blocks: A00(+1), A01(+1); B blocks: B11(+1)
    # t=5 (M6): A blocks: A10(+1), A00(-1); B blocks: B00(+1), B10(+1)
    # t=6 (M7): A blocks: A01(+1), A11(-1); B blocks: B01(+1), B11(+1)
    
    # For each t, the product M_t = (sum_a a_coeff * A_block_a) * (sum_b b_coeff * B_block_b)
    # = sum_{a,b} a_coeff * b_coeff * (A_block_a * B_block_b)
    # Each A_block_a * B_block_b is a 2x2 matrix product, which can be done via Strassen 2x2x2.
    
    # But wait, this would require 4*4=16 2x2 products per M, which is too many.
    # The correct approach: M_t is computed as a single 2x2 matrix product where
    # the first matrix is (sum of A blocks with coefficients) and second is (sum of B blocks).
    # But these are 2x2 matrices, so M_t = X * Y where X is 2x2 and Y is 2x2.
    # We compute X*Y via Strassen 2x2x2, giving rank 7.
    
    # So for each t, X_t and Y_t are specific 2x2 matrices (linear combinations of blocks).
    # The Strassen decomposition of X_t * Y_t gives 7 scalar products.
    
    # The key insight: for the tensor decomposition, we need to track how each scalar
    # product in the Strassen decomposition of M_t contributes to the final C.
    
    # Let me think about this differently. For the 4x4x4 tensor, we need:
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k in {0,1,2,3}.
    
    # Approach: For each of the 7 M terms, and for each of the 7 Strassen sub-terms,
    # we create one column in the decomposition.
    
    # For M_t computed as X_t * Y_t via Strassen 2x2x2:
    # X_t has entries X_t[a,b] = sum of coefficients for A block entries
    # Y_t has entries Y_t[b,c] = sum of coefficients for B block entries
    # The Strassen sub-term s gives:
    # U_sub[a,b,s] * V_sub[b,c,s] * W_sub[c,a,s]
    # where a,b,c are local 2x2 indices.
    
    # To map to global indices:
    # A block (bi,bj) where bi is row-block, bj is col-block:
    # A global index (i,j) -> local block (bi=i//2, bj=j//2), local pos (i%2, j%2)
    
    # For M_t, the A matrix X_t is formed from blocks:
    # X_t[la, lb] where la, lb in {0,1}:
    # X_t[la, lb] = sum over A blocks (bi,bj) of coeff_t(bi,bj) * A[2*bi+la, 2*bj+lb]
    # Similarly Y_t[lb, lc] = sum over B blocks (bj,bk) of coeff_t(bj,bk) * B[2*bj+lb, 2*bk+lc]
    
    # The Strassen sub-term s for X_t*Y_t:
    # U_sub[la, lb, s] contributes to A[2*bi+la, 2*bj+lb] with coefficient coeff_t(bi,bj)
    # for each A block (bi,bj) used in X_t.
    # V_sub[lb, lc, s] contributes to B[2*bj+lb, 2*bk+lc] with coefficient coeff_t(bj,bk)
    # for each B block (bj,bk) used in Y_t.
    # W_sub[lc, la, s] contributes to C[2*bi+la, 2*bk+lc] with coefficient w_t(bi,bk,s)
    # where w_t is the coefficient of M_t in C block (bi,bk).
    
    # This is getting complex. Let me just enumerate.
    
    # A block coefficients for each t:
    # t=0: A00:1, A11:1
    # t=1: A10:1, A11:1
    # t=2: A00:1
    # t=3: A11:1
    # t=4: A00:1, A01:1
    # t=5: A10:1, A00:-1
    # t=6: A01:1, A11:-1
    
    a_coeffs = [
        {(0,0): 1, (1,1): 1},  # t=0
        {(1,0): 1, (1,1): 1},  # t=1
        {(0,0): 1},             # t=2
        {(1,1): 1},             # t=3
        {(0,0): 1, (0,1): 1},  # t=4
        {(1,0): 1, (0,0): -1}, # t=5
        {(0,1): 1, (1,1): -1}, # t=6
    ]
    
    # B block coefficients for each t:
    # t=0: B00:1, B11:1
    # t=1: B00:1
    # t=2: B10:1, B11:-1
    # t=3: B01:1, B00:-1
    # t=4: B11:1
    # t=5: B00:1, B10:1
    # t=6: B01:1, B11:1
    
    b_coeffs = [
        {(0,0): 1, (1,1): 1},  # t=0
        {(0,0): 1},             # t=1
        {(1,0): 1, (1,1): -1}, # t=2
        {(0,1): 1, (0,0): -1}, # t=3
        {(1,1): 1},             # t=4
        {(0,0): 1, (1,0): 1},  # t=5
        {(0,1): 1, (1,1): 1},  # t=6
    ]
    
    # C block coefficients (how M_t contributes to C blocks):
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C block (bi,bk) where bi is row-block of C, bk is col-block of C
    
    c_coeffs = [
        {(0,0): 1, (1,1): 1},    # t=0 (M1): C00+1, C11+1
        {(1,0): 1, (1,1): -1},   # t=1 (M2): C10+1, C11-1
        {(0,1): 1, (1,1): 1},    # t=2 (M3): C01+1, C11+1
        {(0,0): 1, (1,0): 1},    # t=3 (M4): C00+1, C10+1
        {(0,0): -1, (0,1): 1},   # t=4 (M5): C00-1, C01+1
        {(1,1): 1},              # t=5 (M6): C11+1
        {(0,0): 1},              # t=6 (M7): C00+1
    ]
    
    # Now for each t and each Strassen sub-term s (0-6):
    # The Strassen 2x2x2 decomposition of X*Y (where X is 2x2, Y is 2x2)
    # U2[2*la+lb][s], V2[2*lb+lc][s], W2[2*lc+la][s]
    
    # For the global decomposition:
    # U[2*bi+la, 2*bj+lb] gets contribution from U2[la,lb,s] * a_coeffs[t][(bi,bj)]
    # for each (bi,bj) in a_coeffs[t]
    # V[2*bj+lb, 2*bk+lc] gets contribution from V2[lb,lc,s] * b_coeffs[t][(bj,bk)]
    # for each (bj,bk) in b_coeffs[t]
    # W[2*bk+lc, 2*bi+la] gets contribution from W2[lc,la,s] * c_coeffs[t][(bi,bk)]
    # for each (bi,bk) in c_coeffs[t]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U contributions
            for (bi, bj), coeff_a in a_coeffs[t].items():
                for la in range(2):
                    for lb in range(2):
                        u_val = U2[la*2+lb][s]
                        if u_val != 0:
                            i = 2*bi + la
                            j = 2*bj + lb
                            U[i*4+j][col] += u_val * coeff_a
            # V contributions
            for (bj, bk), coeff_b in b_coeffs[t].items():
                for lb in range(2):
                    for lc in range(2):
                        v_val = V2[lb*2+lc][s]
                        if v_val != 0:
                            j = 2*bj + lb
                            k = 2*bk + lc
                            V[j*4+k][col] += v_val * coeff_b
            # W contributions
            for (bi, bk), coeff_c in c_coeffs[t].items():
                for lc in range(2):
                    for la in range(2):
                        w_val = W2[lc*2+la][s]
                        if w_val != 0:
                            k = 2*bk + lc
                            i = 2*bi + la
                            W[k*4+i][col] += w_val * coeff_c
    
    return U, V, W


def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition.
    4x4x2 = two 4x4x1 slices, or use a different approach.
    For 4x4x2, we can think of it as multiplying 4x4 by 4x2.
    One approach: split the 4x2 into two 4x1 vectors.
    Each 4x4x1 (matrix-vector) has rank 16 (schoolbook) or potentially less.
    Actually, matrix-vector multiplication n x m * m x 1 has rank n*m = 16.
    So 4x4x2 via two 4x4x1 would be rank 32.
    
    Alternative: use Strassen-like splitting.
    4x4x2: A is 4x4, B is 4x2, C is 4x2.
    Split A into 2x2 blocks, B into 2x1 blocks (top and bottom).
    C = A*B = [[A00,A01],[A10,A11]] * [[B0],[B1]] where B0, B1 are 2x1.
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    We can use Strassen-like tricks here. Let's see if we can do better than 32.
    
    Actually, for 4x4x2, the tensor rank might be less than 32.
    Let me try a different decomposition.
    
    One approach: use the fact that 4x4x2 can be seen as a bilinear map.
    Let's try to find a rank-28 decomposition or similar.
    
    For now, let me use schoolbook rank 32 and see if I can improve.
    Actually, let me try to use Strassen on the 4x4 part with the 2 columns.
    
    Alternative: decompose 4x4x2 as follows.
    Let A = [[P, Q], [R, S]] (2x2 blocks of 2x2), B = [[U], [V]] (2 blocks of 2x1).
    C = [[P*U + Q*V], [R*U + S*V]]
    
    We can compute:
    M1 = (P+S)*(U+V)
    M2 = (R+S)*U
    M3 = P*(V-U)
    M4 = S*(V+U) -- hmm this doesn't quite work.
    
    Let me just use schoolbook for now: rank 32.
    """
    rank = 32
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*4+j][col] = 1
                V[j*2+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W


def _build_schoolbook_4x1x6():
    """
    4x1x6 decomposition: A is 4x1, B is 1x6, C is 4x6.
    C[i,k] = A[i,0]*B[0,k]
    This is outer product, rank 24 (each (i,k) pair needs one term).
    """
    rank = 24
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(24)]
    col = 0
    for i in range(4):
        for k in range(6):
            U[i*1+0][col] = 1
            V[0*6+k][col] = 1
            W[k*4+i][col] = 1
            col += 1
    return U, V, W


def _combine_decompositions(regions):
    """
    Combine multiple regional decompositions into a full 4x5x6 decomposition.
    Each region is a dict with:
      U, V, W: local factor matrices
      rank: number of columns
      i_range: (start, end) for i dimension (size 4)
      j_range: (start, end) for j dimension (size 5)
      k_range: (start, end) for k dimension (size 6)
    """
    n, m, p = 4, 5, 6
    total_rank = sum(r['rank'] for r in regions)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for region in regions:
        rU = region['U']
        rV = region['V']
        rW = region['W']
        i_start, i_end = region['i_range']
        j_start, j_end = region['j_range']
        k_start, k_end = region['k_range']
        n_local = i_end - i_start
        m_local = j_end - j_start
        p_local = k_end - k_start
        
        for li in range(n_local):
            for lj in range(m_local):
                gi = i_start + li
                gj = j_start + lj
                for c in range(region['rank']):
                    U[gi*m + gj][col_offset + c] = rU[li*m_local + lj][c]
        
        for lj in range(m_local):
            for lk in range(p_local):
                gj = j_start + lj
                gk = k_start + lk
                for c in range(region['rank']):
                    V[gj*p + gk][col_offset + c] = rV[lj*p_local + lk][c]
        
        for lk in range(p_local):
            for li in range(n_local):
                gk = k_start + lk
                gi = i_start + li
                for c in range(region['rank']):
                    W[gk*n + gi][col_offset + c] = rW[lk*n_local + li][c]
        
        col_offset += region['rank']
    
    return U, V, W


def construct():
    start_time = time.time()
    
    # Region A: i in [0,4), j in [0,4), k in [0,4) -> 4x4x4, Strassen rank 49
    U_A, V_A, W_A = _build_strassen_4x4x4()
    region_A = {
        'U': U_A, 'V': V_A, 'W': W_A, 'rank': 49,
        'i_range': (0, 4), 'j_range': (0, 4), 'k_range': (0, 4)
    }
    
    # Region B: i in [0,4), j in [4,5), k in [0,6) -> 4x1x6, schoolbook rank 24
    U_B, V_B, W_B = _build_schoolbook_4x1x6()
    region_B = {
        'U': U_B, 'V': V_B, 'W': W_B, 'rank': 24,
        'i_range': (0, 4), 'j_range': (4, 5), 'k_range': (0, 6)
    }
    
    # Region C: i in [0,4), j in [0,4), k in [4,6) -> 4x4x2, schoolbook rank 32
    U_C, V_C, W_C = _build_strassen_4x4x2()
    region_C = {
        'U': U_C, 'V': V_C, 'W': W_C, 'rank': 32,
        'i_range': (0, 4), 'j_range': (0, 4), 'k_range': (4, 6)
    }
    
    U, V, W = _combine_decompositions([region_A, region_B, region_C])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
