import time

DIMENSIONS = [3, 3, 5]

def _add_dyad(U, V, W, i, j, k, r, n, m, p):
    """Add a rank-1 term A[i,j] * B[j,k] to C[i,k] at index r."""
    # U index: i*m + j
    U[i * m + j][r] += 1
    # V index: j*p + k
    V[j * p + k][r] += 1
    # W index: k*n + i
    W[k * n + i][r] += 1

def _build_strassen_222(offset, U, V, W, n, m, p, i_off, j_off, k_off):
    """
    Build Strassen 2x2x2 decomposition.
    i_off, j_off, k_off are the offsets for the 2x2x2 block within the larger 3x3x3 or 3x3x2.
    Returns the next rank index.
    """
    r = offset
    # M1 = (A11+A22)(B11+B22)
    # A11 is (i_off, j_off), A22 is (i_off+1, j_off+1)
    # B11 is (j_off, k_off), B22 is (j_off+1, k_off+1)
    # This requires expanding (A11+A22) and (B11+B22).
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # But Strassen combines these into one product M1.
    # In the tensor decomposition, we need to represent M1 as a single rank-1 term?
    # No, M1 is a scalar product of two sums.
    # M1 = (A[i,j] + A[i+1,j+1]) * (B[j,k] + B[j+1,k+1])
    # This is NOT a single dyad A[i',j']B[j',k'].
    # It is a product of sums.
    # To represent this in the tensor form \sum U V W, we need to expand it?
    # No, the tensor form allows linear combinations in U, V, W.
    # But each rank-1 term is a single product of entries.
    # Strassen's algorithm works by precomputing M1...M7.
    # M1 is not a single entry product. It is a bilinear form.
    # To fit into the standard tensor decomposition \sum_r U_{\alpha r} V_{\beta r} W_{\gamma r},
    # we must express M1 as a sum of dyads?
    # If we expand M1 into 4 dyads, we lose the rank reduction.
    # The rank reduction comes from the fact that M1 is computed once and used multiple times.
    # In the tensor model, the "rank" is the number of scalar multiplications.
    # The standard tensor decomposition \sum U V W corresponds to the number of scalar multiplications if U, V, W are 0/1 matrices?
    # No. If U, V, W have entries like 1, -1, 2, etc., it still corresponds to scalar multiplications.
    # However, a single term $U_{\alpha r} V_{\beta r} W_{\gamma r}$ computes $A_{\alpha} B_{\beta} C_{\gamma}$?
    # No, it contributes to $C_{\gamma}$.
    # The term is $A_{\alpha} B_{\beta}$.
    # If we want to compute $M1 = (A_{11}+A_{22})(B_{11}+B_{22})$, we cannot do it in one dyad $A_{\alpha} B_{\beta}$.
    # We need to expand it:
    # $M1 = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}$.
    # This uses 4 multiplications.
    # Then Strassen uses M1 in 4 places.
    # Total multiplications: 7 (for M1..M7) ?
    # No, M1 is ONE multiplication of two sums.
    # In the tensor model, we are counting the rank of the tensor.
    # The rank of the tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is $u \otimes v \otimes w$.
    # This corresponds to computing $u^T A v$? No.
    # It corresponds to a term $A_{i,j} B_{j,k}$.
    # If we use Strassen, we are effectively using a different basis or a different decomposition.
    # But the tensor rank is an invariant.
    # Strassen's algorithm shows that the rank of $2 \times 2 \times 2$ is 7.
    # This means there exist vectors $u_r, v_r, w_r$ such that
    # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
    # This is an exact identity.
    # We can hardcode the 7 dyads for $2 \times 2 \times 2$.
    
    # The 7 dyads for Strassen 2x2x2:
    # Indices 0,1 for 2x2.
    # M1: (A00+A11)(B00+B11) -> This is not a single dyad.
    # Wait, the tensor decomposition of Strassen's algorithm is NOT just expanding the products.
    # It is a specific set of 7 rank-1 tensors.
    # Let's find them.
    # The tensor T has entries T[i,j,k] = 1 if j is the middle index?
    # No, T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the indices match?
    # Actually, the tensor is defined by T[i,j,k] = 1 if we multiply A[i,j] and B[j,k] to get C[i,k].
    # So T[i,j,k] is 1 for all valid i,j,k?
    # No, T is a 3-tensor.
    # T[i,j,k] = 1 if the operation is A[i,j] * B[j,k] -> C[i,k].
    # So T[i,j,k] is 1 for all i,j,k?
    # No, the indices are fixed.
    # For 2x2x2:
    # T[0,0,0] = 1 (A00*B00 -> C00)
    # T[0,1,0] = 1 (A01*B10 -> C00)
    # T[1,0,0] = 1 (A10*B00 -> C10)
    # T[1,1,0] = 1 (A11*B10 -> C10)
    # T[0,0,1] = 1 (A00*B01 -> C01)
    # ...
    # The rank-1 tensors $u \otimes v \otimes w$ have entries $u_i v_j w_k$.
    # We need $\sum_r u_{r,i} v_{r,j} w_{r,k} = T[i,j,k]$.
    
    # The Strassen decomposition for 2x2x2 is known.
    # I will use a known set of 7 vectors.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    # We will build the 43 rank solution.
    # Rank 26 for 3x3x3, Rank 17 for 3x3x2.
    # Total 43.
    
    U = [[0] * 43 for _ in range(n * m)]
    V = [[0] * 43 for _ in range(m * p)]
    W = [[0] * 43 for _ in range(p * n)]
    
    # We need the explicit Strassen dyads.
    # If I can't recall them, I can use the 26-rank decomposition which is just schoolbook for the 1x1, 1x2, 2x1, 2x2 blocks?
    # No, 26 used Strassen for 2x2x2.
    # If I use schoolbook for all, it's 27 + 18 = 45.
    # I MUST implement Strassen for 2x2x2 to get the improvement.
    
    # Let's hardcode the Strassen 2x2x2 decomposition.
    # Reference: Strassen's 7 products.
    # The 7 rank-1 tensors are:
    # 1. u=(1,0,1,0)? No, u is length 4 (2x2).
    # u1 = [1, 0, 1, 0] (A00, A01, A10, A11)? No, A is 2x2.
    # Let's use 0-indexed.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # The 7 terms are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # To convert these to rank-1 tensors u ⊗ v ⊗ w:
    # M1 is not a single dyad.
    # This implies that the tensor rank 7 is NOT achieved by simply expanding these products into dyads.
    # The tensor rank is the minimum number of dyads.
    # Strassen's algorithm proves that 7 dyads exist.
    # I need to find them.
    
    # Actually, I can just use the schoolbook 45 and hope for a small improvement?
    # No, 45 is the baseline.
    
    # Let's try to find a 44 solution.
    # 3x3x3 schoolbook is 27.
    # 3x3x2 schoolbook is 18.
    # Total 45.
    
    # Can we reduce 3x3x2 to 17?
    # 3x3x2 = 2x2x2 (7) + 2x1x2 (4) + 1x2x2 (4) + 1x1x2 (2) = 17.
    # This uses Strassen for the 2x2x2 part.
    # So if I implement Strassen for 2x2x2, I get 17 for the second part.
    # For the first part 3x3x3:
    # 2x2x2 (7) + 2x2x1 (4) + 2x1x2 (4) + 1x2x2 (4) + 2x1x1 (2) + 1x2x1 (2) + 1x1x2 (2) + 1x1x1 (1) = 26.
    # Total 26 + 17 = 43.
    
    # I will implement this.
    
    pass

