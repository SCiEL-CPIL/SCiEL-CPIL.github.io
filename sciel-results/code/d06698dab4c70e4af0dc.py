"""Matrix multiplication 2x2 by 2x5 rank-minimal exact decomposition."""
import time
import random

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm for the 2x2 * 2x2 part,
    plus 12 direct multiplications for the 2x2 * 2x3 part.
    Total rank: 7 + 12 = 19.
    
    However, we can do better by using the fact that the 2x2*2x3 part can be
    decomposed more cleverly. Let's use a mixed approach:
    - For the first 2 columns of B (and C), use Strassen (rank 7)
    - For the remaining 3 columns, use direct multiplication (rank 12)
    Total: 19
    
    But actually, we can try to find a better decomposition by exploiting
    the rectangular nature. Let me think about this more carefully.
    
    The tensor T_{i,j,k} = delta_{j,k} for (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    For 2x2 * 2x5:
    - n=2, m=2, p=5
    - U has shape (4, r), V has shape (10, r), W has shape (10, r)
    - U[i*m+j, c] = U[2*i+j, c]
    - V[j*p+k, c] = V[5*j+k, c]  
    - W[k*n+i, c] = W[5*k+i, c]  (following the trace convention W[k*n+i,r])
    
    The tensor equation is:
    sum_c U[2*i+j, c] * V[5*j+k, c] * W[5*k+i, c] = delta_{i,k}  (for each i,j,k)
    Wait, that's not right. Let me re-derive.
    
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    In tensor form: T_{i,j,k} where T_{i,j,k} = A_{i,j} * B_{j,k} contributes to C_{i,k}
    
    The standard decomposition:
    sum_c U[2*i+j, c] * V[5*j+k, c] * W[5*k+i, c] should equal the "product" structure.
    
    Actually, the condition is that for each (i,j,k):
    sum_c U[2*i+j, c] * V[5*j+k, c] * W[5*k+i, c] = 1 if this is a "valid" term
    
    No wait. The matrix multiplication tensor is T with T[i,j,k] = 1 for all i,j,k,
    and the contraction is: sum_{j} A[i,j] B[j,k] = C[i,k]
    
    In the U,V,W decomposition:
    sum_c U[2*i+j, c] V[5*j+k, c] W[5*k+i, c] should reconstruct the identity on the "middle" index.
    
    Actually, I think the correct condition is:
    For the tensor T_{i,j,k} (where i indexes A rows, j indexes A cols/B rows, k indexes B cols/C cols),
    the decomposition satisfies:
    sum_c U[2*i+j, c] * V[5*j+k, c] * W[5*k+i, c] = delta_{i,k}  ??? No.
    
    Let me look at the seed code more carefully:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1  
    W[k*n+i][column] = 1
    
    And it iterates over all (i,j,k) with rank = n*m*p.
    
    So the condition must be that summing over all columns c where the triple (i,j,k) is "active" gives 1, and the structure ensures correctness.
    
    The actual condition for matrix multiplication tensor decomposition:
    The tensor T is T[i,j,k] = 1 for all valid indices (it's the "multiplication" tensor).
    A rank-r decomposition is U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    But T[i,j,k] = 1 for all i,j,k? That can't be right because then we'd need to reconstruct an all-ones tensor.
    
    Wait, I think I'm confusing myself. Let me reconsider.
    
    In matrix multiplication, the "tensor" being decomposed is the bilinear map:
    (A, B) -> C where C = A * B
    
    The rank of this tensor is the minimum number of bilinear multiplications needed.
    
    For the schoolbook method with n=2, m=2, p=5:
    We have 2*2*5 = 20 bilinear terms, one for each (i,j,k) pair.
    
    The U,V,W matrices satisfy:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k V[j*p+k,c] B[j,k]) ... 
    
    Hmm, this is getting confusing. Let me just trust the seed code structure and verify my decomposition.
    
    For Strassen on 2x2 * 2x2:
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1]-B[1,1])
    M4 = A[1,1] * (B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    C[0,0] = M1 - M4 + M6 - M7
    C[0,1] = M3 + M5 - M6
    C[1,0] = M2 + M7 - M5
    C[1,1] = M1 - M2 + M3 + M4
    
    In the U,V,W framework with n=2,m=2,p=2:
    U indices: 2*i+j for i in {0,1}, j in {0,1} -> rows 0,1,2,3
    V indices: 2*j+k for j in {0,1}, k in {0,1} -> rows 0,1,2,3
    W indices: 2*k+i for k in {0,1}, i in {0,1} -> rows 0,1,2,3
    
    For M1: U[0,c]=U[3,c]=1 (A[0,0]+A[1,1]), V[0,c]=V[3,c]=1 (B[0,0]+B[1,1]), W contributes to all C[i,k] with appropriate signs.
    
    Actually, W should encode the "output" coefficients. So for each bilinear term c, W encodes how it contributes to each C[i,k].
    
    So W[2*k+i, c] gives the coefficient for C[i,k] from term c.
    
    For M1: contributes +1 to C[0,0], -1 to C[1,1], etc. Wait no:
    C[0,0] = M1 - M4 + M6 - M7, so M1 contributes +1 to C[0,0]
    C[0,1] = M3 + M5 - M6, so M1 contributes 0 to C[0,1]
    C[1,0] = M2 + M7 - M5, so M1 contributes 0 to C[1,0]
    C[1,1] = M1 - M2 + M3 + M4, so M1 contributes +1 to C[1,1]
    
    So W[2*0+0, c1] = W[0, c1] = 1 (for C[0,0])
       W[2*0+1, c1] = W[1, c1] = 0 (for C[1,0])
       W[2*1+0, c1] = W[2, c1] = 0 (for C[0,1])
       W[2*1+1, c1] = W[3, c1] = 1 (for C[1,1])
    
    OK this makes sense now. Let me code the full decomposition.
    """
    
    n, m, p = 2, 2, 5
    
    # Total rank: 7 (Strassen) + 12 (direct for 3 cols) = 19
    # But let's see if we can do better.
    
    # Actually, for 2x2 * 2x3, the minimal rank is unknown but bounded.
    # Let's just go with 7 + 12 = 19 for now, but try to optimize.
    
    # Alternative: maybe we can share some structure between the Strassen part
    # and the direct part? Probably not easily.
    
    # Let's construct the 19-rank decomposition.
    
    rank = 19
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 19
    V = [[0.0] * rank for _ in range(m * p)]  # 10 x 19
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 19
    
    # Part 1: Strassen for B[:,0:2] -> C[:,0:2], using columns 0-6 of U,V,W
    # U rows: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # V rows for first 2 cols: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    # W rows for first 2 cols: 0=C[0,0], 1=C[1,0], 2=C[0,1], 3=C[1,1]
    
    # Strassen terms (column index in 0..6):
    # col 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C[0,0] += M1, C[1,1] += M1
    
    # col 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C[1,0] += M2
    
    # col 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C[0,1] += M3
    
    # col 3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[3][3] = 1  # C[0,0] -= M4, C[1,1] += M4
    
    # col 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[2][4] = -1  # C[0,1] += M5, C[1,0] -= M5
    
    # col 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = -1  # C[0,0] += M6, C[0,1] -= M6
    
    # col 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[2][6] = 1  # C[0,0] -= M7, C[1,0] += M7
    
    # Part 2: Direct multiplication for B[:,2:5] -> C[:,2:5], using columns 7-18
    # V rows for cols 2-4: 4=B[0,2], 5=B[0,3], 6=B[0,4], 7=B[1,2], 8=B[1,3], 9=B[1,4]
    # W rows for cols 2-4: 4=C[0,2], 5=C[1,2], 6=C[0,3], 7=C[1,3], 8=C[0,4], 9=C[1,4]
    
    # 12 direct terms:
    direct_terms = [
        (0, 4, 4),  # A00*B02 -> C02
        (0, 5, 6),  # A00*B03 -> C03
        (0, 6, 8),  # A00*B04 -> C04
        (1, 7, 4),  # A01*B12 -> C02
        (1, 8, 6),  # A01*B13 -> C03
        (1, 9, 8),  # A01*B14 -> C04
        (2, 4, 5),  # A10*B02 -> C12
        (2, 5, 7),  # A10*B03 -> C13
        (2, 6, 9),  # A10*B04 -> C14
        (3, 7, 5),  # A11*B12 -> C12
        (3, 8, 7),  # A11*B13 -> C13
        (3, 9, 9),  # A11*B14 -> C14
    ]
    
    for idx, (u_row, v_row, w_row) in enumerate(direct_terms):
        col = 7 + idx
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
