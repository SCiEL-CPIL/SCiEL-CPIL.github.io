"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # We aim to find a rank-minimal decomposition.
    # Current seed is rank 36 (n*m*p).
    # We can improve this by using the fact that for any fixed column k,
    # the multiplication is A (2x2) * b_k (2x1), which has rank 4 schoolbook,
    # but perhaps we can share factors or use Strassen for the whole thing?
    # Actually, the tensor is T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a direct sum of p independent 2x2 matrix-vector multiplications?
    # No, A is shared.
    # C_{i,k} = sum_{j=0}^{1} A_{i,j} B_{j,k}.
    # This can be viewed as: for each k, C_{:,k} = A @ B_{:,k}.
    # The rank of the tensor is at most sum over k of rank(2x2x1 multiplication) which is 4*9=36.
    # Can we do better?
    # The tensor is T \in R^{2x2} \otimes R^{2x9} \otimes R^{2x9}.
    # Let's try to search for a lower rank using random linear combinations or known algorithms.
    # For 2x2 * 2x2, min rank is 7 (Strassen).
    # For 2x2 * 2x9, we can apply Strassen to the first 2 columns (rank 7)
    # and schoolbook to the remaining 7 columns (rank 4 each -> 28). Total 35.
    # Is 35 the minimum?
    # Let's check if we can mix columns.
    # The matrix multiplication tensor for m x n * n x p has rank R(m,n,p).
    # R(2,2,2) = 7.
    # R(2,2,3) = ?
    # Generally, R(m,n,p) <= m*n*p.
    # Using Strassen recursively isn't straightforward for rectangular.
    # However, we can use the "addition chain" style or specific decompositions.
    
    # Let's construct the rank 35 decomposition explicitly.
    # Columns 0,1 use Strassen (rank 7).
    # Columns 2..8 use schoolbook (rank 4 each).
    
    rank = 7 + 4 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indexing:
    # U: (n*m, rank) -> U[i*m+j, r] corresponds to A_{i,j}
    # V: (m*p, rank) -> V[j*p+k, r] corresponds to B_{j,k}
    # W: (p*n, rank) -> W[k*n+i, r] corresponds to C_{i,k}
    
    # Strassen for columns 0, 1
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]] (columns 0, 1)
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    
    # Strassen 7 multiplications:
    # m1 = (a+d)(e+g)
    # m2 = (c+d)(e+g)
    # m3 = a(f-g)
    # m4 = d(h-f)
    # m5 = (a+b)(g+h)
    # m6 = (c+a)(e+f)
    # m7 = (b+d)(h-f)
    
    # C00 = m1 - m4 + m6
    # C01 = m1 + m4
    # C10 = m2 - m7
    # C11 = m1 - m2 + m5
    
    # Note: This standard Strassen works for 2x2 * 2x2.
    # Here, the second matrix is 2x9.
    # We are treating the first 2 columns as a 2x2 block.
    
    col_idx = 0
    
    def add_term(u_coeff, v_coeff, w_coeff, term_idx):
        # u_coeff: list of (i,j) with coefficients
        # v_coeff: list of (j,k) with coefficients
        # w_coeff: list of (i,k) with coefficients (to be added to C)
        for (i, j), u_val in u_coeff.items():
            if u_val != 0:
                U[i * m + j][term_idx] += u_val
        for (j, k), v_val in v_coeff.items():
            if v_val != 0:
                V[j * p + k][term_idx] += v_val
        for (i, k), w_val in w_coeff.items():
            if w_val != 0:
                W[k * n + i][term_idx] += w_val
        return term_idx + 1
    
    # We need to be careful with signs and summations in W.
    # The tensor identity is: sum_r U_{i,j,r} V_{j,k,r} W_{i',k',r} = delta_{i,i'} delta_{k,k'} (coefficient)
    # Wait, the definition is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # We decompose T = sum_r u_r \otimes v_r \otimes w_r
    # where T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} ? No.
    # The tensor for matrix multiplication M_{n,m,p} has entries M_{i,j,k} such that
    # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T is indexed by (i, j, k) and the multiplication corresponds to
    # contracting j.
    # The rank decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}?
    # No, usually: T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in the output.
    # But the output is C_{i,k}.
    # So T_{i,j,k} = 1 if the output C_{i,k} contains A_{i,j} B_{j,k}.
    # Actually, T is a tensor in R^{n*m} x R^{m*p} x R^{p*n}.
    # T_{(i,j), (j,k), (i,k)} ? No, indices must align.
    # The standard tensor for MM is T \in V^* \otimes W^* \otimes Z where
    # T(v, w) = v w (the product).
    # In components: T_{i,j, k, l, m, n} ?
    # Let's stick to the interface provided in the seed.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # The constraint is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct the tensor?
    # No, the tensor value T_{i,j,k} (where i row of A, j col of A, k col of B)
    # is 1 if it contributes to C_{i,k}.
    # Actually, the "multiplication tensor" is defined such that
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # Since each C_{i,k} is a sum of such terms, and they are linearly independent,
    # the tensor T has entries 1 at positions (i,j,k) corresponding to the term A_{i,j} B_{j,k}
    # in the output C_{i,k}.
    # But wait, the output C_{i,k} is a linear form in A and B.
    # The tensor T_{i,j,k} is such that T(A,B) = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = C?
    # No, T(A,B) is a scalar.
    # The tensor decomposition is for the *bilinear map* (A,B) -> C.
    # This map can be written as C = sum_r (U_r \cdot A) (V_r \cdot B) W_r^T ?
    # Where U_r \cdot A is a scalar (trace(U_r^T A)), V_r \cdot B is a scalar.
    # Then C_{i,k} = sum_r (sum_{j,l} U_{r,j,l} A_{j,l}) (sum_{l,m} V_{r,l,m} B_{l,m}) W_{r,i,k}.
    # Comparing with C_{i,k} = sum_j A_{i,j} B_{j,k}:
    # We need sum_r U_{r,i,j} V_{r,j,k} W_{r,i,k} to pick out the coefficient of A_{i,j} B_{j,k}.
    # Since A and B are independent variables, we must have:
    # For each fixed i,j,k, the coefficient of A_{i,j} B_{j,k} in the r-th term
    # is U_{r,i,j} V_{r,j,k} W_{r,i,k} ?
    # No, the sum over r is outside.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want C_{i,k} = sum_r (U_r^T A) (V_r^T B) (W_r)_{i,k}.
    # Expanding:
    # C_{i,k} = sum_r (sum_{j',l'} U_{r,j',l'} A_{j',l'}) (sum_{l'',m'} V_{r,l'',m'} B_{l'',m'}) W_{r,i,k}
    # We want this to equal sum_j A_{i,j} B_{j,k}.
    # This requires that for each r, the term (U_r^T A) (V_r^T B) is a product of a linear form in A and a linear form in B.
    # Let \phi_r(A) = sum_{j,l} U_{r,j,l} A_{j,l} and \psi_r(B) = sum_{l,m} V_{r,l,m} B_{l,m}.
    # Then C_{i,k} = sum_r \phi_r(A) \psi_r(B) W_{r,i,k}.
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # The bilinear form \phi_r(A) \psi_r(B) must be of the form A_{i',j'} B_{j'',k'} (possibly summed).
    # But \phi_r and \psi_r are arbitrary linear forms.
    # The rank of the tensor is the minimum r such that we can express the MM tensor as a sum of r rank-1 tensors.
    # A rank-1 tensor in this context is U \otimes V \otimes W where
    # T_{i,j,k} = U_{i,j} V_{j,k} W_{i,k} ?
    # No, the indices of T are (i,j) for A, (j,k) for B, and (i,k) for C.
    # So T_{(i,j), (j,k), (i,k)} = U_{i,j} V_{j,k} W_{i,k} is NOT a valid rank-1 tensor because the indices j must match?
    # Actually, the tensor T lives in (R^{n*m})^* \otimes (R^{m*p})^* \otimes (R^{p*n}).
    # The basis for the first factor is e_{i,j}. Second is e_{j,k}. Third is e_{i,k}.
    # The MM tensor T is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
    # Wait, the third index is (i,k).
    # So T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
    # We want to decompose T = sum_r u_r \otimes v_r \otimes w_r.
    # Where u_r \in R^{n*m}, v_r \in R^{m*p}, w_r \in R^{p*n}.
    # Let u_r be a matrix U_r (n x m), v_r be a matrix V_r (m x p), w_r be a matrix W_r (p x n).
    # Then (u_r \otimes v_r \otimes w_r)_{(i,j), (j,k), (i,k)} = (U_r)_{i,j} (V_r)_{j,k} (W_r)_{i,k}.
    # So we need:
    # 1 = sum_r (U_r)_{i,j} (V_r)_{j,k} (W_r)_{i,k} for all i,j,k?
    # No, the sum is over r, and the indices i,j,k are fixed.
    # But wait, the tensor T has non-zero entries only when the indices match the pattern (i,j), (j,k), (i,k).
    # And the value is 1.
    # So the condition is:
    # sum_r (U_r)_{i,j} (V_r)_{j,k} (W_r)_{i,k} = 1 for all i,j,k.
    # This is the condition we must satisfy.
    
    # Let's check the seed:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # Sum over r (which is col) is 1 * 1 * 1 = 1. Correct.
    
    # Now, can we find a decomposition with rank < 36?
    # For n=m=2, p=9.
    # Condition: sum_r (U_r)_{i,j} (V_r)_{j,k} (W_r)_{i,k} = 1.
    
    # Let's try the Strassen decomposition for the first 2 columns.
    # For k=0,1, we want to represent the terms involving B_{j,0} and B_{j,1}.
    # The terms are:
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
    # C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}
    # C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1}
    
    # We can use 7 multiplications to compute these 4 outputs.
    # The 7 products are:
    # P1 = (A_{0,0} + A_{1,1})(B_{0,0} + B_{1,0})
    # P2 = (A_{1,0} + A_{1,1})(B_{0,0} + B_{1,0})
    # P3 = A_{0,0}(B_{0,1} - B_{1,0})
    # P4 = A_{1,1}(B_{1,1} - B_{0,1})
    # P5 = (A_{0,0} + A_{0,1})(B_{1,0} + B_{1,1})
    # P6 = (A_{1,0} + A_{0,0})(B_{0,0} + B_{0,1})
    # P7 = (A_{0,1} + A_{1,1})(B_{1,1} - B_{0,1})
    
    # These P_r are linear combinations of A and B.
    # Each P_r corresponds to a rank-1 term in the tensor decomposition?
    # No, P_r is a product of two linear forms.
    # In the tensor decomposition, each term r is u_r \otimes v_r \otimes w_r.
    # u_r is a vector in R^{2x2} (flattened), v_r in R^{2x9} (flattened), w_r in R^{9x2} (flattened).
    # The product (U_r)_{i,j} (V_r)_{j,k} (W_r)_{i,k} must sum to 1.
    
    # For the Strassen part, we can define U_r, V_r, W_r such that
    # (U_r)_{i,j} (V_r)_{j,k} is the coefficient of A_{i,j} B_{j,k} in P_r.
    # And W_r is the matrix such that C = sum_r P_r W_r.
    # Wait, C_{i,k} = sum_r (P_r) (W_r)_{i,k}.
    # And P_r = sum_{i',j',k',l'} (U_r)_{i',j'} (V_r)_{j',k'} A_{i',j'} B_{k',l'} ?
    # This is getting confusing.
    # Let's just use the explicit coefficients.
    
    # For each r, U_r is a 2x2 matrix, V_r is a 2x9 matrix, W_r is a 9x2 matrix.
    # We need sum_r U_r \circ V_r \circ W_r = T, where \circ denotes the appropriate contraction.
    # Specifically, for all i,j,k:
    # sum_r (U_r)_{i,j} (V_r)_{j,k} (W_r)_{i,k} = 1.
    
    # Let's construct the 7 terms for k=0,1.
    # For k >= 2, we can use schoolbook: 4 terms per column.
    # Total rank = 7 + 7*4 = 35.
    
    # Let's fill U, V, W.
    col = 0
    
    # Helper to add a term
    def add_strassen_term(U_mat, V_mat, W_mat):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[i*2+j][col] = U_mat[i][j]
        for j in range(2):
            for k in range(9):
                V[j*9+k][col] = V_mat[j][k]
        for k in range(9):
            for i in range(2):
                W[k*2+i][col] = W_mat[k][i]
        col += 1
        
    # Strassen Terms
    # P1 = (A00+A11)(B00+B10)
    # U1 = [[1,0],[0,1]], V1 = [[1,0...],[1,0...]] (only col 0,1 of B matter? No, V is 2x9)
    # V1 should have non-zeros only in columns 0,1 to avoid interference?
    # If V1 has non-zeros in other columns, they will contribute to C_{i,k} for k>=2.
    # But for k>=2, we are adding separate terms.
    # If V1 has non-zeros in k>=2, then for those k, the term (U1)_{i,j} (V1)_{j,k} (W1)_{i,k} will be added.
    # We want this to be 0 for k>=2.
    # So we can set V1 to be zero for k>=2.
    # Similarly for other Strassen terms.
    
    V1 = np.zeros((2,9))
    V1[0,0] = 1; V1[0,1] = 0
    V1[1,0] = 1; V1[1,1] = 0
    U1 = np.array([[1,0],[0,1]])
    # C00 += P1, C01 += 0, C10 += 0, C11 += 0 ?
    # C = P1 * W1 + ...
    # C00 = P1 - P4 + P6
    # C01 = P1 + P4
    # C10 = P2 - P7
    # C11 = P1 - P2 + P5
    # So W1 should be:
    # W1[0,0] = 1, W1[0,1] = 1, W1[1,0] = 0, W1[1,1] = 1
    # Wait, W1 is 9x2.
    # W1[k, i] is the coefficient for C_{i,k}.
    # For k=0: C00 += 1*P1, C10 += 0*P1.
    # For k=1: C01 += 1*P1, C11 += 1*P1.
    W1 = np.zeros((9,2))
    W1[0,0] = 1
    W1[1,0] = 1
    W1[1,1] = 1
    add_strassen_term(U1, V1, W1)
    
    # P2 = (A10+A11)(B00+B10)
    U2 = np.array([[0,0],[1,1]])
    V2 = np.zeros((2,9))
    V2[0,0] = 1; V2[1,0] = 1
    W2 = np.zeros((9,2))
    # C10 -= P2, C11 += P2
    W2[0,1] = -1
    W2[1,1] = 1
    add_strassen_term(U2, V2, W2)
    
    # P3 = A00(B01-B10)
    U3 = np.array([[1,0],[0,0]])
    V3 = np.zeros((2,9))
    V3[0,1] = 1; V3[1,0] = -1
    W3 = np.zeros((9,2))
    # C01 += P3
    W3[1,0] = 1
    add_strassen_term(U3, V3, W3)
    
    # P4 = A11(B11-B01)
    U4 = np.array([[0,0],[0,1]])
    V4 = np.zeros((2,9))
    V4[0,1] = -1; V4[1,1] = 1
    W4 = np.zeros((9,2))
    # C00 -= P4, C01 += P4
    W4[0,0] = -1
    W4[1,0] = 1
    add_strassen_term(U4, V4, W4)
    
    # P5 = (A00+A01)(B10+B11)
    U5 = np.array([[1,1],[0,0]])
    V5 = np.zeros((2,9))
    V5[1,0] = 1; V5[1,1] = 1
    W5 = np.zeros((9,2))
    # C11 += P5
    W5[1,1] = 1
    add_strassen_term(U5, V5, W5)
    
    # P6 = (A10+A00)(B00+B01)
    U6 = np.array([[1,0],[1,0]])
    V6 = np.zeros((2,9))
    V6[0,0] = 1; V6[0,1] = 1
    W6 = np.zeros((9,2))
    # C00 += P6
    W6[0,0] = 1
    add_strassen_term(U6, V6, W6)
    
    # P7 = (A01+A11)(B11-B01)
    U7 = np.array([[0,1],[0,1]])
    V7 = np.zeros((2,9))
    V7[0,1] = -1; V7[1,1] = 1
    W7 = np.zeros((9,2))
    # C10 += P7
    W7[0,1] = -1  # Wait, C10 = P2 - P7. So -1.
    add_strassen_term(U7, V7, W7)
    
    # Now for k=2..8, schoolbook.
    for k in range(2, 9):
        # C0k = A00 B0k + A01 B1k
        # Term 1: A00 B0k -> C0k
        U_t = np.zeros((2,2)); U_t[0,0] = 1
        V_t = np.zeros((2,9)); V_t[0,k] = 1
        W_t = np.zeros((9,2)); W_t[k,0] = 1
        add_strassen_term(U_t, V_t, W_t)
        
        # Term 2: A01 B1k -> C0k
        U_t[0,0] = 0; U_t[0,1] = 1
        V_t[:,k] = 0; V_t[1,k] = 1
        W_t[k,0] = 1
        add_strassen_term(U_t, V_t, W_t)
        
        # Term 3: A10 B0k -> C1k
        U_t[:] = 0; U_t[1,0] = 1
        V_t[:,k] = 0; V_t[0,k] = 1
        W_t[k,0] = 0; W_t[k,1] = 1
        add_strassen_term(U_t, V_t, W_t)
        
        # Term 4: A11 B1k -> C1k
        U_t[:] = 0; U_t[1,1] = 1
        V_t[:,k] = 0; V_t[1,k] = 1
        W_t[k,1] = 1
        add_strassen_term(U_t, V_t, W_t)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
