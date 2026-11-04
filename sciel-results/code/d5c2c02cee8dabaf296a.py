"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, 10s safety margin
    
    # We will construct a rank-36 decomposition.
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A*B = A*B1 + A*B2 + A*B3.
    # Rank(A*B1) = 16 (Laderman)
    # Rank(A*B2) = 16 (Laderman)
    # Rank(A*B3) = 4 (Rank 4 for 4x4x1)
    # Total Rank = 36.

    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman's 16 terms for 4x4x4 matrix multiplication.
    # Indices: A[i,j], B[j,k], C[i,k]
    # We need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For the sub-problem, local indices i, j, k are 0..3.
    
    # Laderman's coefficients (16 terms)
    # Source: Standard Laderman algorithm for 4x4 matrix multiplication.
    # Format: (u_indices, v_indices, w_indices, coeff)
    # Actually, it's easier to define the 16 bilinear forms directly.
    
    # I will use a known set of 16 terms.
    # If I don't have them hardcoded, I will use a fallback.
    # For now, I will implement the Rank 4 part and use Schoolbook for the 4x4x4 parts if Laderman is not available?
    # No, I will hardcode Laderman.
    
    # Laderman's 16 terms for 4x4x4:
    # Z1 = (A00+A11)(B00+B11)
    # Z2 = (A01+A11)(B10+B11)
    # Z3 = (A00+A01)(B01+B10)
    # Z4 = (A10+A11)(B01+B10)
    # Z5 = (A00+A22)(B00+B22)
    # Z6 = (A02+A22)(B20+B22)
    # Z7 = (A00+A02)(B02+B20)
    # Z8 = (A20+A22)(B02+B20)
    # Z9 = (A00+A33)(B00+B33)
    # Z10 = (A03+A33)(B30+B33)
    # Z11 = (A00+A03)(B03+B30)
    # Z12 = (A30+A33)(B03+B30)
    # Z13 = (A10+A21)(B12+B21)
    # Z14 = (A11+A20)(B12+B21)
    # Z15 = (A12+A23)(B13+B22)
    # Z16 = (A13+A22)(B13+B22)
    # ... This is getting messy and I might be misremembering the exact 16.
    
    # Safer approach: Use the current program's search but with a better initialization or just return Schoolbook?
    # No, the task requires improvement.
    
    # Let's use a simpler decomposition:
    # Split 4x4x9 into 9 blocks of 4x4x1.
    # Rank of 4x4x1 is 4.
    # Total Rank = 9 * 4 = 36.
    # This is much easier to implement correctly!
    
    # For each column k of B (0..8):
    # C[:, k] = A * B[:, k]
    # This is a 4x4x1 multiplication.
    # Rank 4 decomposition:
    # For term t in 0..3:
    # U[i, t] = delta(i, t) ? No.
    # C_i = sum_j A_ij B_jk.
    # Term t:
    # U[i, t] = 1 if i == t else 0?
    # V[j, t] = 1 if j == t else 0?
    # W[k, t] = 1 if k == k_target else 0?
    
    # Let's verify.
    # Sum_t U[i,t] V[j,t] W[k,t]
    # = Sum_t delta(i,t) delta(j,t) delta(k, k_target)
    # = delta(i,j) delta(k, k_target)
    # This gives C[i, k_target] = A[i,j] B[j,k_target] ?
    # No, the sum is over j in the multiplication definition.
    # C[i, k] = sum_j A[i,j] B[j,k].
    # My tensor decomposition:
    # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c].
    # If I use the above terms:
    # T[i, j, k] = delta(i,j) delta(k, k_target).
    # This means C[i, k] = A[i,i] B[i,k_target] ?
    # No, the indices i and j are separate inputs.
    # The term delta(i,j) implies i=j.
    # But in C[i,k] = sum_j A[i,j] B[j,k], i and j are not necessarily equal.
    
    # Correct Rank 4 decomposition for 4x4x1 (C = A b):
    # C_i = sum_j A_ij b_j.
    # We can use 4 terms.
    # Term j (0..3):
    # U[i, j] = 1 if i == j else 0? No.
    # We want to compute the dot product of row i of A and vector b.
    # This is not separable in i and j simply.
    # Wait, C = A b.
    # This is a linear map.
    # The tensor rank of matrix-vector multiplication 4x4x1 is 4.
    # Decomposition:
    # C_i = sum_j A_ij b_j.
    # We can write this as:
    # C_i = sum_{j=0}^3 (A_ij) (b_j) (1).
    # So for term j:
    # U[i, j] = A_ij ? No, U must be constant.
    # The decomposition is:
    # C_i = sum_c U_ic V_jc W_kc.
    # Here k is fixed to 0 (for the 4x4x1 case).
    # So W_kc is 1 for k=0, 0 otherwise.
    # We need sum_c U_ic V_jc = A_ij b_j ? No.
    # The bilinear form is B(A, b) = A b.
    # We need to decompose the tensor T such that T[i, j, 0] corresponds to the coefficient of A_ij b_j in C_i.
    # Actually, C_i = sum_j T[i, j, 0] A_ij b_j.
    # We want T[i, j, 0] = delta_ij ? No.
    # We want C_i = sum_j A_ij b_j.
    # So the coefficient of A_ij b_j in C_i is 1 if i is the row index?
    # Yes, for a fixed i, the term A_ij b_j appears with coeff 1.
    # So T[i, j, 0] = 1 for all i, j?
    # No, that would sum all A_ij b_j into every C_i.
    # We need T[i, j, 0] to be 1 only if the output index is i?
    # The tensor T has indices i (output), j (input A/B), k (input B/Output).
    # For 4x4x1, k=0.
    # C_i = sum_j A_ij b_j.
    # So the contribution of A_ij b_j to C_i is 1.
    # The contribution of A_ij b_j to C_{i'} (i' != i) is 0.
    # So T[i, j, 0] = 1 if the output index is i?
    # Yes.
    # So T[i, j, 0] = 1 for all i, j?
    # No, T is a 3D array.
    # T[i, j, k].
    # If I set T[i, j, 0] = 1 for all i, j, then:
    # C_i = sum_j sum_k T[i, j, k] A_ij b_k ?
    # No, the formula is C = sum_c U_c V_c W_c.
    # C_ik = sum_j A_ij B_jk.
    # For 4x4x1, B is 4x1.
    # C_i = sum_j A_ij B_j0.
    # We need a decomposition of the tensor T such that:
    # T[i, j, 0] = 1 for all i, j?
    # If T[i, j, 0] = 1, then the bilinear form is sum_ij A_ij B_j0.
    # This is a scalar. But C is a vector.
    # The tensor must produce the vector C.
    # So T must have an index for the output C.
    # The indices are i (out), j (in), k (in).
    # For 4x4x1, k is the column of B (0).
    # i is the row of C (0..3).
    # j is the inner index (0..3).
    # We want C_i = sum_j A_ij B_j0.
    # So the term A_ij B_j0 contributes to C_i with coeff 1.
    # It contributes to C_{i'} (i' != i) with coeff 0.
    # So T[i, j, 0] = 1 if i is the output index?
    # Yes, T[i, j, 0] = 1 for all i, j.
    # Wait, if T[i, j, 0] = 1 for all i, j, then:
    # C_i = sum_j sum_k T[i, j, k] A_ij B_jk ?
    # No, the decomposition is:
    # C_ik = sum_j A_ij B_jk.
    # This is a single tensor equation.
    # The tensor T is defined by T[i, j, k] = delta_{jk}? No.
    # For 4x4x1:
    # T[i, j, 0] = 1 for all i, j?
    # If I use rank 4:
    # Term t (0..3):
    # U[i, t] = delta(i, t).
    # V[j, t] = 1.
    # W[k, t] = delta(k, 0).
    # Sum_t U[i,t] V[j,t] W[k,t] = sum_t delta(i,t) * 1 * delta(k,0) = delta(i,t) delta(k,0).
    # This sums to delta(k,0) if i is in 0..3?
    # No, sum_t delta(i,t) is 1 for any i in 0..3.
    # So T[i, j, k] = delta(k,0).
    # This means C_ik = sum_j A_ij B_jk delta(k,0).
    # If k=0, C_i0 = sum_j A_ij B_j0.
    # If k!=0, C_ik = 0.
    # This is exactly correct!
    
    # So the Rank 4 decomposition for 4x4x1 is:
    # For t in 0..3:
    # U[i, t] = 1 if i == t else 0.
    # V[j, t] = 1 for all j.
    # W[k, t] = 1 if k == 0 else 0.
    
    # Now apply this to the 9 columns of B.
    # For column k of B (0..8):
    # We use 4 terms.
    # Total 36 terms.
    
    for k in range(9):
        for t in range(4):
            c = k * 4 + t
            
            # U[i, c]
            for i in range(4):
                if i == t:
                    U[i * 4 + t, c] = 1  # u = i*m + j? No.
                    # U index is i*m + j.
                    # In the 4x4x1 decomposition, j is the inner index.
                    # U corresponds to A. A has indices i, j.
                    # So U index is i*4 + j.
                    # My term t fixes i=t?
                    # No, U[i, t] = delta(i, t).
                    # So U[i*4 + j, c] = 1 if i == t else 0.
                    # This depends on j?
                    # In the derivation: U[i, t] = delta(i, t).
                    # This means U is 4x4.
                    # U[i, t] is the entry.
                    # So for a fixed term t, U has 1 at row i=t, col t?
                    # No, U is n x r.
                    # U[i, c].
                    # My derivation: U[i, t] = delta(i, t).
                    # This means U[i, c] = 1 if i == t else 0.
                    # This is independent of j?
                    # But U index is i*m + j.
                    # So U[i*4 + j, c] = 1 if i == t else 0.
                    
                    # Wait, if U[i*4 + j, c] = 1 for all j when i==t,
                    # and V[j*9 + k, c] = 1 for all j?
                    # And W[k*4 + i, c] = 1 if k == k_target?
                    
                    # Let's re-verify the indices for the full problem.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # Term t for column k:
                    # U[i*4 + j, c] = 1 if i == t else 0.
                    # V[j*9 + k, c] = 1.
                    # W[k*4 + i, c] = 1 if k == k else 0? (k is the column index).
                    
                    # Sum_c U[i*4+j, c] V[j*9+k, c] W[k*4+i, c]
                    # = sum_t (1 if i==t else 0) * 1 * (1 if k==k else 0)
                    # = 1 if i==t else 0.
                    # This sums to 1 for any i?
                    # No, the sum is over c (which is t).
                    # For a fixed i, j, k:
                    # Sum_t delta(i, t) * 1 * delta(k, k) = 1.
                    # So T[i, j, k] = 1.
                    # This gives C[i, k] = sum_j A[i, j] B[j, k] * 1.
                    # This is correct!
                    
                    # So:
                    # U[i*4 + j, c] = 1 if i == t else 0.
                    # V[j*9 + k, c] = 1.
                    # W[k*4 + i, c] = 1.
                    
                    # Wait, W index is k*n + i = 9k + i.
                    # So W[9k + i, c] = 1.
                    
                    # Let's check if this works for all i, j, k.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # My tensor:
                    # T[i, j, k] = sum_c U[i*4+j, c] V[j*9+k, c] W[9k+i, c].
                    # For term t (c = k*4 + t):
                    # U[i*4+j, c] = 1 if i == t else 0.
                    # V[j*9+k, c] = 1.
                    # W[9k+i, c] = 1.
                    # Sum_t (1 if i==t else 0) = 1.
                    # So T[i, j, k] = 1.
                    # This implies C[i, k] = sum_j A[i, j] B[j, k].
                    # Correct.
                    
                    # So for each k (0..8) and t (0..3):
                    # c = k*4 + t.
                    # For all i, j:
                    # U[i*4 + j, c] = 1 if i == t else 0.
                    # V[j*9 + k, c] = 1.
                    # W[9k + i, c] = 1.
                    
                    for i in range(4):
                        if i == t:
                            for j in range(4):
                                U[i * 4 + j, c] = 1
                        # else 0
                    for j in range(4):
                        V[j * 9 + k, c] = 1
                    for i in range(4):
                        W[9 * k + i, c] = 1

    # Verify
    if check_validity(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def check_validity(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

# EVOLVE-BLOCK-END
