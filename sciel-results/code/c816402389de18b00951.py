import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We use the decomposition of 2x3x3 matrix multiplication (rank 9)
    # tensor-producted with the multiplication of R[x]/(x^3) (rank 5).
    # Total rank = 9 * 5 = 45.
    
    rank = 45
    
    # U: (n*p, rank) -> (18, 45)
    # V: (m*p, rank) -> (27, 45)
    # W: (p*n, rank) -> (18, 45)
    # Note: W dimension in contract is (p*n, rank). p=9, n=2. 18.
    # V dimension is (m*p, rank). m=3, p=9. 27.
    # U dimension is (n*m, rank)? No, contract says (n*m, rank).
    # Wait, contract says:
    # U: (n*m, rank) -> (6, 45)
    # V: (m*p, rank) -> (27, 45)
    # W: (p*n, rank) -> (18, 45)
    
    # Let's re-verify the indices for the tensor decomposition.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # U is for A? No.
    # Standard: C = sum_l U_l V_l W_l^T?
    # Contract:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # This implies:
    # C[i, k] (flattened to index i*n+k? or k*n+i?)
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i.
    # U index is i*m + j? (A is n x m)
    # V index is j*p + k? (B is m x p)
    
    # Let's map the 2x3x3 decomposition to this.
    # 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # Terms: (i, j, k) for i in 0..1, j in 0..2, k in 0..2.
    # U_233[i, j] = 1
    # V_233[j, k] = 1
    # W_233[k, i] = 1 (if C is 2x3, index k*n+i? No, C is 2x3, so k is col, i is row. k*2+i)
    
    # Now, B is 3x9. We view it as 3x3 over R[x]/(x^3).
    # B[j, l] where l in 0..2 is the algebra element.
    # B[j, l] = sum_m B[j, 3l+m] e_m.
    # A[i, j] is scalar, so A[i, j] e_0.
    # C[i, l] = sum_j A[i, j] B[j, l] (algebra product).
    # C[i, l] = sum_j A[i, j] (e_0 * B[j, l]).
    # e_0 * e_m = e_m.
    # So C[i, l] = sum_j A[i, j] B[j, l] (scalar product of coefficients).
    # This means the "algebra multiplication" part is just the identity on the basis?
    # No. The rank reduction comes from the fact that we are decomposing the *multiplication* of the algebra elements.
    # But here, one operand is always e_0 (scalar).
    # The multiplication of e_0 with any e_m is e_m.
    # This operation has rank 3 (identity).
    # So the rank should be 9 * 3 = 27?
    
    # Let's check.
    # If we use the algebra decomposition, we are effectively using:
    # T_total = sum_{l=1}^{9} u_l v_l w_l (for 2x3x3)
    # Tensor product with T_alg = sum_{k=1}^{5} a_k b_k c_k.
    # The term is u_l a_k, v_l b_k, w_l c_k.
    # Here v_l is the A-index. A is scalar.
    # In the algebra product, the "first operand" is A.
    # So b_k must correspond to the A-index in the algebra.
    # But A is scalar, so it only has the e_0 component.
    # If b_k has non-zero components in e_1, e_2, those terms will be 0 because A has no e_1, e_2 components.
    # So we can restrict to terms where b_k is e_0?
    # In the rank-5 decomposition of R[x]/(x^3):
    # 1. (0, 0, 0) -> b=0
    # 2. (1, 0, 1) -> b=0
    # 3. (2, 0, 2) -> b=0
    # 4. (1, 1, 0) -> b=1
    # 5. (2, 1, 1) -> b=1
    # Only terms 1, 2, 3 have b=0.
    # So only 3 terms survive?
    # Rank 9 * 3 = 27.
    
    # Let's verify if rank 27 is possible.
    # If we use only terms with b=0, we are using the sub-algebra spanned by e_0?
    # No, we are using the full output a_k and input c_k.
    # Terms:
    # 1. a=0, b=0, c=0
    # 2. a=1, b=0, c=1
    # 3. a=2, b=0, c=2
    # This corresponds to:
    # C_0 = A_0 B_0
    # C_1 = A_0 B_1
    # C_2 = A_0 B_2
    # This is just scalar multiplication by A_0.
    # It does not capture the full matrix multiplication.
    
    # Wait. The tensor product logic:
    # T_233 = sum_{i,j,k} e_{i,k} \otimes e_{i,j} \otimes e_{j,k}
    # T_alg = sum_{a,b,c} e_a \otimes e_b \otimes e_c
    # T_total = sum_{i,j,k,a,b,c} (e_{i,k} \otimes e_a) \otimes (e_{i,j} \otimes e_b) \otimes (e_{j,k} \otimes e_c)
    # The A-index is (i,j) in 233 and b in alg.
    # So A_{i,j} is multiplied by A_{alg, b}.
    # But A_{alg, b} is 0 for b != 0 because A is scalar.
    # So we only sum over b=0.
    # So we only use terms in T_alg where b=0.
    # In the rank-5 decomposition, only 3 terms have b=0.
    # So the rank is 9 * 3 = 27.
    
    # Is this correct?
    # Let's check the resulting C.
    # C_{i, l, a} = sum_{j, k, c} A_{i,j} B_{j, k, c} * (coeff from terms).
    # With b=0 terms:
    # Term 1: a=0, c=0. C_{i, l, 0} += A_{i,j} B_{j, k, 0}
    # Term 2: a=1, c=1. C_{i, l, 1} += A_{i,j} B_{j, k, 1}
    # Term 3: a=2, c=2. C_{i, l, 2} += A_{i,j} B_{j, k, 2}
    # This gives C_{i, l, a} = A_{i,j} B_{j, k, a}.
    # This is exactly the schoolbook multiplication!
    # And the rank is 27.
    
    # So we can achieve rank 27.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is the column index in the 3x3 block
                # The 3 terms for b=0
                # Term 1: a=0, c=0
                # U index: (i, k*3 + 0)
                # V index: (j, k*3 + 0)
                # W index: (k*3 + 0, i)
                
                # Term 2: a=1, c=1
                # U index: (i, k*3 + 1)
                # V index: (j, k*3 + 1)
                # W index: (k*3 + 1, i)
                
                # Term 3: a=2, c=2
                # U index: (i, k*3 + 2)
                # V index: (j, k*3 + 2)
                # W index: (k*3 + 2, i)
                
                for a in range(3):
                    U[i * m + j][col] = 1
                    V[j * p + (k * 3 + a)][col] = 1
                    W[(k * 3 + a) * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
