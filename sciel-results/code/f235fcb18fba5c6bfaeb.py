import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2 x 2x2 multiplication with rank 7.
    U: (4, 7) indexed by (i, k)
    V: (4, 7) indexed by (k, j)
    W: (4, 7) indexed by (j, i)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen 7 terms
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Indices: U[i*2+k, r], V[k*2+j, r], W[j*2+i, r]
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds rank-49 decomposition for 4x4 x 4x4.
    U: (16, 49) indexed by (i, k) where i,k in 0..3
    V: (16, 49) indexed by (k, j) where k,j in 0..3
    W: (16, 49) indexed by (j, i) where j,i in 0..3
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # The 7 Strassen combinations at the 4x4 block level
    # Each combination r involves linear combinations of 2x2 blocks.
    # We define the coefficients for A blocks, B blocks, and C blocks.
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3 in block space)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # For each r, we need to map the 2x2x2 Strassen indices to 4x4x4 indices.
    # The 2x2x2 Strassen computes Z = X Y where X, Y, Z are 2x2 matrices.
    # X entries are linear combos of A blocks.
    # Y entries are linear combos of B blocks.
    # Z entries are linear combos of C blocks.
    
    # Let's define the linear combinations for X, Y, Z for each of the 7 big terms.
    # X is 2x2, Y is 2x2, Z is 2x2.
    # X[bi, bk], Y[bk, bj], Z[bj, bi]
    
    big_terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # X1 = [[A00, 0], [0, A11]]
    # Y1 = [[B00, 0], [0, B11]]
    # Z1 = [[C00, 0], [0, C11]]
    # Note: C00 += M1[0,0], C11 += M1[1,1]. M1[0,1] and M1[1,0] are 0?
    # Wait, M1 is a 2x2 matrix product.
    # M1 = (A00+A11)(B00+B11).
    # M1[0,0] = A00*B00.
    # M1[0,1] = A00*B11.
    # M1[1,0] = A11*B00.
    # M1[1,1] = A11*B11.
    # In Strassen, M1 is used as:
    # C00 += M1[0,0] + ...
    # C11 += M1[1,1] + ...
    # But the off-diagonals of M1 are not directly added to C01 or C10 in the standard Strassen formula?
    # Standard Strassen:
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,0] + M5[0,0]
    # C10 = M2[0,0] + M4[0,0]
    # C11 = M1[1,1] - M2[1,1] + M3[1,1] + M6[1,1]
    
    # Wait, the standard Strassen formulas for 2x2 matrices A, B, C:
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Here M_r are scalars in the 2x2 case.
    # In the 4x4 case, M_r are 2x2 matrices.
    # So C00 (which is a 2x2 block) = M1[0,0] - M5[0,0] + M4[0,0] + M7[0,0]?
    # No. The formula applies to the blocks.
    # C_{00} (block) = M_{1,00} - M_{5,00} + M_{4,00} + M_{7,00}?
    # Actually, the Strassen algorithm works on the entries.
    # If A, B are 2x2 blocks of 2x2 matrices, then the "entries" of the "big" A are the blocks.
    # So the formula $C_{00} = M_1 - M_5 + M_4 + M_7$ holds for the blocks $C_{00}, M_1, \dots$
    # Where $M_1$ is the result of the multiplication of the "big" entries.
    # So $M_1$ is a 2x2 matrix (product of two 2x2 matrices).
    # And $C_{00}$ is a 2x2 matrix.
    # So $C_{00} = M_1 - M_5 + M_4 + M_7$ is a matrix equation.
    # $C_{00}[i,j] = M_1[i,j] - M_5[i,j] + M_4[i,j] + M_7[i,j]$.
    
    # So we need to decompose each $M_r$ (which is a 2x2x2 multiplication) into rank 7.
    # Then sum them up with signs in W.
    
    # Let's define the "Big" Strassen terms.
    # For r in 0..6:
    #   Define X_r (2x2 matrix of A-blocks)
    #   Define Y_r (2x2 matrix of B-blocks)
    #   M_r = X_r Y_r
    
    # X_r entries:
    # r=0 (M1): X = [[A00, 0], [0, A11]]
    # r=1 (M2): X = [[0, 0], [A10+A11, 0]]? 
    #    M2 = (A10+A11)B00.
    #    X = [[0,0],[A10+A11, 0]]? No, X must be 2x2.
    #    M2 is a scalar in the "big" sense? No, M2 is a 2x2 matrix.
    #    M2 = (A10+A11) * B00.
    #    This is a product of a 2x2 matrix (A10+A11) and a 2x2 matrix (B00).
    #    So X = [[0, 0], [A10+A11, 0]]? No.
    #    The multiplication is (A10+A11) * B00.
    #    This is not a 2x2 matrix multiplication of the form [[a,b],[c,d]] * [[e,f],[g,h]].
    #    It is a single 2x2 matrix times another 2x2 matrix.
    #    So we can treat it as a 2x2x2 multiplication where the "A" matrix is [[0,0],[A10+A11, 0]]?
    #    No, that would compute:
    #    [0*0 + 0*B00, 0*0 + 0*B00]
    #    [(A10+A11)*0 + 0*B00, (A10+A11)*0 + 0*B00] -> Wrong.
    
    #    We need to embed the product $P Q$ into a 2x2 matrix multiplication.
    #    We can set $X = [[P, 0], [0, 0]]$ and $Y = [[Q, 0], [0, 0]]$?
    #    Then $X Y = [[PQ, 0], [0, 0]]$.
    #    Then we take the [0,0] block of the result.
    #    This works.
    
    #    So for each big term r:
    #    1. Identify the two 2x2 matrices $P_r, Q_r$ such that $M_r = P_r Q_r$.
    #    2. Embed them into 2x2 matrices $X_r, Y_r$ such that $X_r Y_r$ has $P_r Q_r$ in the [0,0] position (or anywhere, as long as we track it).
    #    3. Use the rank-7 Strassen decomposition for the 2x2x2 multiplication $X_r Y_r = Z_r$.
    #    4. The entries of $Z_r$ are linear combinations of the 7 rank-1 terms.
    #    5. The final C blocks are linear combinations of $Z_r$ entries.
    
    # Let's refine the embedding.
    # For M1: P=A00+A11, Q=B00+B11.
    # X1 = [[P, 0], [0, 0]], Y1 = [[Q, 0], [0, 0]].
    # Z1 = X1 Y1 = [[PQ, 0], [0, 0]].
    # We need PQ. It is Z1[0,0].
    
    # For M2: P=A10+A11, Q=B00.
    # X2 = [[P, 0], [0, 0]], Y2 = [[Q, 0], [0, 0]].
    # Z2 = [[PQ, 0], [0, 0]].
    
    # This embedding wastes ranks? No, the rank of the 2x2x2 multiplication is 7 regardless of the sparsity, unless we exploit sparsity.
    # But we are using the fixed rank-7 decomposition for 2x2x2.
    # So we just map the non-zero entries of X and Y to the corresponding indices in U and V.
    
    # Let's list P_r and Q_r for r=0..6:
    # 0: P=A00+A11, Q=B00+B11
    # 1: P=A10+A11, Q=B00
    # 2: P=A00, Q=B01-B11
    # 3: P=A11, Q=B10-B00
    # 4: P=A00+A01, Q=B11
    # 5: P=A10-A00, Q=B00+B01
    # 6: P=A01-A11, Q=B10+B11
    
    # For each r, we decompose $Z_r = X_r Y_r$ using Strassen 2x2x2.
    # $X_r$ is 2x2, $Y_r$ is 2x2.
    # $X_r = [[P_r, 0], [0, 0]]$, $Y_r = [[Q_r, 0], [0, 0]]$.
    # So $X_r[0,0]=P_r$, others 0.
    # $Y_r[0,0]=Q_r$, others 0.
    
    # In the 2x2x2 Strassen decomposition, the indices are:
    # U[i,k,r'], V[k,j,r'], W[j,i,r']
    # Here i,k,j are in {0,1}.
    
    # For a specific big term r, we have 7 sub-terms r' in {0..6}.
    # Total rank 49.
    
    # We need to fill U, V, W for the 4x4x4 case.
    
    # Let's map the 2x2 block indices to 4x4 indices.
    # Block (bi, bk) of A: rows 2*bi..2*bi+1, cols 2*bk..2*bk+1.
    # Entry (i, k) in 4x4: i = 2*bi + i2, k = 2*bk + k2.
    
    # For each big term r:
    #   P_r is a 2x2 matrix.
    #   Q_r is a 2x2 matrix.
    #   We need to express P_r and Q_r as linear combinations of A and B blocks.
    
    # Let's define the coefficients for P_r and Q_r.
    # P_r is 2x2. Let P_r[bi, bk] be a 2x2 matrix of coefficients for A blocks?
    # No, P_r is a sum of A blocks.
    # e.g. P_0 = A00 + A11.
    # A00 is a 2x2 matrix.
    # So P_0 is a 2x2 matrix where each entry is a linear combo of A entries?
    # No, P_0 is a 2x2 matrix of *blocks*?
    # No, A00 is a 2x2 matrix.
    # So P_0 is a 2x2 matrix.
    # P_0[i,j] = A00[i,j] + A11[i,j].
    
    # So for each entry (i2, k2) in P_r (where i2, k2 in 0,1),
    # P_r[i2, k2] = sum_{bi,bk} coeff[bi,bk] * A_{bi,bk}[i2, k2].
    
    # We need to add these contributions to U.
    # U has shape (16, 49).
    # Index (i, k) in U corresponds to A[i,k].
    # i = 2*bi + i2, k = 2*bk + k2.
    
    # For each big term r and sub-term r' (0..6):
    #   We use the 2x2x2 Strassen U2, V2, W2.
    #   U2[i2, k2, r'] is the coefficient for A2[i2, k2] in the r'-th term.
    #   But here A2 is X_r.
    #   X_r[0,0] = P_r. X_r[1,0]=0, etc.
    #   So X_r[i2, k2] is non-zero only if i2=0, k2=0.
    #   And X_r[0,0] = P_r.
    #   So the "A" part of the 2x2x2 multiplication is P_r at position (0,0).
    #   So for the 2x2x2 Strassen, the input matrix A2 has only A2[0,0] non-zero.
    #   A2[0,0] = P_r.
    #   So U2[0,0,r'] is the coefficient.
    #   This coefficient multiplies the entry P_r[i2, k2]?
    #   No.
    #   The 2x2x2 multiplication is $Z = X Y$.
    #   $Z[i,j] = \sum_k X[i,k] Y[k,j]$.
    #   In Strassen, $Z[i,j] = \sum_{r'} U2[i,k,r'] V2[k,j,r'] W2[j,i,r']$? No.
    #   The decomposition is:
    #   $\sum_{r'} U2[i,k,r'] V2[k,j,r'] W2[j,i,r']$?
    #   The tensor is $T[i,k,j] = \sum_{r'} U[i,k,r'] V[k,j,r'] W[j,i,r']$.
    #   Here $T[i,k,j]$ is the coefficient of $A[i,k] B[k,j]$ in $C[i,j]$.
    #   
    #   For the 2x2x2 case, $i,k,j \in \{0,1\}$.
    #   $X[i,k]$ is the entry of the "A" matrix.
    #   $Y[k,j]$ is the entry of the "B" matrix.
    #   
    #   In our embedding, $X[i,k] = 0$ unless $i=0, k=0$.
    #   And $X[0,0] = P_r$.
    #   So $X[0,0]$ is a 2x2 matrix (of scalars).
    #   Wait, $X$ is a 2x2 matrix of *scalars* in the 2x2x2 problem.
    #   But here $X[0,0]$ is a 2x2 matrix of *scalars* (the entries of P_r).
    #   This means we are effectively doing a $2 \times 2 \times 2$ multiplication where the "scalars" are $2 \times 2$ matrices?
    #   No, that's not right.
    #   
    #   Let's re-evaluate.
    #   We want to compute $M_r = P_r Q_r$.
    #   $P_r$ and $Q_r$ are $2 \times 2$ matrices.
    #   We can view this as a $2 \times 2 \times 2$ multiplication.
    #   The "A" matrix is $P_r$, "B" matrix is $Q_r$, "C" matrix is $M_r$.
    #   The indices for this 2x2x2 multiplication are $i_2, k_2, j_2 \in \{0,1\}$.
    #   $P_r[i_2, k_2]$ is a scalar.
    #   $Q_r[k_2, j_2]$ is a scalar.
    #   $M_r[i_2, j_2] = \sum_{k_2} P_r[i_2, k_2] Q_r[k_2, j_2]$.
    #   
    #   So we apply the rank-7 Strassen decomposition to this 2x2x2 multiplication.
    #   $M_r[i_2, j_2] = \sum_{r'} U2[i_2, k_2, r'] V2[k_2, j_2, r'] W2[j_2, i_2, r']$?
    #   No, the decomposition is for the *tensor* of the multiplication.
    #   $T[i_2, k_2, j_2] = \delta_{i_2, j_2}$? No.
    #   $C[i_2, j_2] = \sum_{k_2} A[i_2, k_2] B[k_2, j_2]$.
    #   The tensor is $T[i_2, k_2, j_2] = 1$ if we are looking at the coefficient?
    #   No, the tensor is defined by $T[i_2, k_2, j_2]$ such that $C[i_2, j_2] = \sum_{k_2} T[i_2, k_2, j_2] A[i_2, k_2] B[k_2, j_2]$.
    #   Here $T[i_2, k_2, j_2] = 1$ for all $i_2, k_2, j_2$?
    #   No, $C[i_2, j_2]$ depends on $k_2$.
    #   The standard matrix multiplication tensor $M_{n,m,p}$ has $T[i,k,j] = 1$ if $i$ is row of A, $k$ is col of A/row of B, $j$ is col of B.
    #   So for 2x2x2, $T[i,k,j] = 1$ for all $i,k,j \in \{0,1\}$.
    #   And the Strassen decomposition gives $T = \sum_{r'} U_r \otimes V_r \otimes W_r$.
    #   
    #   So for each big term r, we use the 7 Strassen terms.
    #   For each sub-term r', we have:
    #   $U2[i_2, k_2, r']$ is a scalar.
    #   $V2[k_2, j_2, r']$ is a scalar.
    #   $W2[j_2, i_2, r']$ is a scalar.
    #   
    #   The contribution to the global U, V, W is:
    #   For each $i_2, k_2, j_2$:
    #   The coefficient of $A[i_2, k_2]$ in the $r$-th big term is $P_r[i_2, k_2]$.
    #   But $P_r[i_2, k_2]$ is a linear combination of $A$'s global entries.
    #   
    #   This is getting complicated to track manually.
    #   
    #   Alternative: Just construct the full $4 \times 4 \times 4$ tensor and use the known rank-49 decomposition?
    #   No, I need to generate it.
    
    #   Let's use the property that $M_r = P_r Q_r$.
    #   $P_r$ is a $2 \times 2$ matrix.
    #   $P_r[i_2, k_2] = \sum_{bi, bk} c_{r, i_2, k_2, bi, bk} A[2bi+i_2, 2bk+k_2]$.
    #   
    #   For the $r$-th big term, the "A" matrix in the 2x2x2 multiplication is $P_r$.
    #   So the "A" entries are $P_r[i_2, k_2]$.
    #   The Strassen decomposition for 2x2x2 says:
    #   $\sum_{k_2} P_r[i_2, k_2] Q_r[k_2, j_2] = \sum_{r'} U2[i_2, k_2, r'] V2[k_2, j_2, r'] W2[j_2, i_2, r']$?
    #   No, the identity is:
    #   $C[i_2, j_2] = \sum_{r'} U2[i_2, k_2, r'] V2[k_2, j_2, r'] W2[j_2, i_2, r']$ is wrong notation.
    #   
    #   The identity is:
    #   $\sum_{k_2} A[i_2, k_2] B[k_2, j_2] = \sum_{r'} ( \sum_{k_2} U2[i_2, k_2, r'] A[i_2, k_2] ) ( \sum_{k_2} V2[k_2, j_2, r'] B[k_2, j_2] ) W2[j_2, i_2, r']$?
    #   No.
    #   
    #   The rank-r decomposition is:
    #   $T = \sum_{r'} u^{(r')} \otimes v^{(r')} \otimes w^{(r')}$.
    #   $T[i,k,j] = \sum_{r'} U[i,k,r'] V[k,j,r'] W[j,i,r']$.
    #   
    #   For the 2x2x2 case, $T[i,k,j] = 1$.
    #   So $1 = \sum_{r'} U2[i,k,r'] V2[k,j,r'] W2[j,i,r']$.
    #   
    #   In our big term r, we have $M_r[i_2, j_2] = \sum_{k_2} P_r[i_2, k_2] Q_r[k_2, j_2]$.
    #   We can substitute the Strassen identity for the product of $P_r$ and $Q_r$.
    #   $M_r[i_2, j_2] = \sum_{r'} U2[i_2, k_2, r'] V2[k_2, j_2, r'] W2[j_2, i_2, r'] P_r[i_2, k_2] Q_r[k_2, j_2]$?
    #   No, the indices $k_2$ in $U2, V2$ are summed over in the Strassen identity?
    #   No, $U2[i,k,r']$ is a specific entry.
    #   
    #   The correct substitution is:
    #   $M_r[i_2, j_2] = \sum_{r'} W2[j_2, i_2, r'] ( \sum_{k_2} U2[i_2, k_2, r'] P_r[i_2, k_2] ) ( \sum_{k_2} V2[k_2, j_2, r'] Q_r[k_2, j_2] )$.
    #   
    #   So the "new" A-entry for the $r'$-th sub-term is $A'_{r', i_2, k_2} = U2[i_2, k_2, r'] P_r[i_2, k_2]$.
    #   The "new" B-entry is $B'_{r', k_2, j_2} = V2[k_2, j_2, r'] Q_r[k_2, j_2]$.
    #   The "new" C-entry is $C'_{r', j_2, i_2} = W2[j_2, i_2, r'] M_r[i_2, j_2]$.
    
    #   Now, $P_r[i_2, k_2]$ is a linear combination of global $A$ entries.
    #   $P_r[i_2, k_2] = \sum_{bi, bk} c_{r, i_2, k_2, bi, bk} A[2bi+i_2, 2bk+k_2]$.
    #   
    #   So $A'_{r', i_2, k_2} = \sum_{bi, bk} (U2[i_2, k_2, r'] c_{r, i_2, k_2, bi, bk}) A[2bi+i_2, 2bk+k_2]$.
    #   
    #   This allows us to fill the global U matrix.
    #   $U[2bi+i_2, 2bk+k_2, 7r+r'] = U2[i_2, k_2, r'] c_{r, i_2, k_2, bi, bk}$.
    
    #   Similarly for V and W.
    
    #   We need the coefficients $c_{r, i_2, k_2, bi, bk}$ for $P_r$ and $Q_r$.
    
    #   Let's define P_r and Q_r coefficients.
    #   P_r is 2x2.
    #   r=0: P = A00+A11.
    #   P[0,0] = A00[0,0] + A11[0,0]
    #   P[0,1] = A00[0,1] + A11[0,1]
    #   P[1,0] = A00[1,0] + A11[1,0]
    #   P[1,1] = A00[1,1] + A11[1,1]
    
    #   So for r=0:
    #   c[0, i2, k2, 0, 0] = 1
    #   c[0, i2, k2, 1, 1] = 1
    #   others 0.
    
    #   This is manageable.
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    
    # Build 4x4x4 Strassen
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Build 4x4x1 Schoolbook
    U441 = np.zeros((16, 16), dtype=int)
    V441 = np.zeros((4, 16), dtype=int) # V is (m*p, rank) = (4*1, 16)
    W441 = np.zeros((4, 16), dtype=int) # W is (p*n, rank) = (1*4, 16)
    
    c = 0
    for i in range(4):
        for k in range(4):
            U441[i*4+k, c] = 1
            V441[k*1+0, c] = 1
            W441[0*4+i, c] = 1
            c += 1
            
    # Assemble 4x4x9
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # Rank = 49 + 49 + 16 = 114
    
    U = np.zeros((16, 114), dtype=int)
    V = np.zeros((36, 114), dtype=int)
    W = np.zeros((36, 114), dtype=int)
    
    # Part 1: B1 (cols 0-3)
    U[:, 0:49] = U444
    V[0:16, 0:49] = V444 # B1 is 4x4, so V indices 0..15
    W[0:16, 0:49] = W444 # C1 is 4x4, so W indices 0..15
    
    # Part 2: B2 (cols 4-7)
    U[:, 49:98] = U444
    V[16:32, 49:98] = V444 # B2 is 4x4, V indices 16..31 (k*9 + j, j=4..7)
    # Wait, V index is k*9 + j.
    # For B2, j is 4..7.
    # V444 has indices k*4 + j (j=0..3).
    # We need to map V444[k*4+j] to V[k*9 + (j+4)].
    
    # Let's redo the mapping carefully.
    
    # V444 shape (16, 49). Index (k*4+j).
    # V_global shape (36, 114). Index (k*9+j).
    
    # For Part 1 (j=0..3):
    # V_global[k*9+j, r] = V444[k*4+j, r]
    
    # For Part 2 (j=4..7):
    # V_global[k*9+j, r+49] = V444[k*4+(j-4), r]
    
    # For Part 3 (j=8):
    # V_global[k*9+8, r+98] = V441[k*1+0, r]
    
    # Similarly for W.
    # W444 shape (16, 49). Index (j*4+i).
    # W_global shape (36, 114). Index (j*4+i).
    # Wait, W index is (j, i) where j is col of C, i is row of C.
    # C is 4x9.
    # W index is (j*4 + i) where j=0..8, i=0..3.
    
    # For Part 1 (j=0..3):
    # W_global[j*4+i, r] = W444[j*4+i, r]
    
    # For Part 2 (j=4..7):
    # W_global[j*4+i, r+49] = W444[(j-4)*4+i, r]
    
    # For Part 3 (j=8):
    # W_global[j*4+i, r+98] = W441[0*4+i, r]
    
    # Let's implement the mapping.
    
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V444[k*4+j, 0:49]
            
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V444[k*4+j, 0:49]
            
    for k in range(4):
        V[k*9+8, 98:114] = V441[k, 0:16]
        
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W444[j*4+i, 0:49]
            
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W444[j*4+i, 0:49]
            
    for i in range(4):
        W[8*4+i, 98:114] = W441[i, 0:16]
        
    # U is just concatenated
    U[:, 0:49] = U444
    U[:, 49:98] = U444
    U[:, 98:114] = U441
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Need to implement _build_4x4x4_strassen properly
