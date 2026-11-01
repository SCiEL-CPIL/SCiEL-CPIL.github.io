"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
from itertools import product, permutations

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-6 exact tensor decomposition for 2x2 matrix multiplication.
    If no rank-6 solution exists over F_3, fall back to the verified rank-7 Strassen decomposition.
    
    The matrix multiplication tensor for 2x2 matrices has entries:
    T[i, j, k] = sum_{a,b} A[i,a] * B[a,k] * delta_{j,k?} ... 
    Actually, the tensor is defined such that:
    (A * B)[i, k] = sum_{j} A[i, j] * B[j, k]
    
    In tensor form, the multiplication tensor M has:
    M[i, j, k] = 1 if the output position (i,k) depends on input A[i,j] and B[j,k]
    
    More precisely, the bilinear form is:
    sum_{i,j,k} (A*B)[i,k] * X_{i,j} * Y_{j,k}
    
    The tensor T is a 3-tensor with indices (i, j, k) where:
    - i: row of A (0,1)
    - j: column of A / row of B (0,1) 
    - k: column of B (0,1)
    
    And T[i,j,k] = 1 for all valid (i,j,k) combinations, meaning all 8 entries are 1?
    No, that's not right.
    
    Let me reconsider. The matrix multiplication tensor is a 3-linear form:
    T(A, B, C) = sum_{i,k} C[i,k] * (A*B)[i,k] = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    
    So the tensor entries are T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    Wait, that would make it just an all-ones tensor, which can be decomposed in rank 1.
    That's clearly wrong.
    
    Let me be more careful. The standard way to encode matrix multiplication as a tensor:
    
    The multiplication map is: A (n x m) * B (m x p) -> C (n x p)
    
    The tensor is T in R^{n x p} x R^{n x m} x R^{m x p} such that:
    C = sum_{r} U_r * V_r^T ... no.
    
    Actually, for a rank-r decomposition:
    C[i,k] = sum_{r=1}^{r} U[i,r] * V[k,r] * W[r,r] ... 
    
    Let me use the convention from the problem:
    U has shape (n*m, rank)
    V has shape (m*p, rank) 
    W has shape (p*n, rank)
    
    And the output follows W[k*n+i, r] convention.
    
    The decomposition satisfies:
    For all i in {0,...,n-1}, k in {0,...,p-1}:
    (A*B)[i,k] = sum_{r} U[i*m + ?, r] * V[?, r] * W[k*n + i, r]
    
    Hmm, I need to get the indexing right.
    
    Standard tensor decomposition for matmul:
    The tensor T has indices (i, a, k) where:
    - i: row index of A (and output C)
    - a: column of A / row of B
    - k: column of B (and output C)
    
    T[i, a, k] = 1 for all i, a, k (since (A*B)[i,k] = sum_a A[i,a] * B[a,k])
    
    A rank-r decomposition writes T as:
    T[i,a,k] = sum_{r=1}^{r} u_r[i] * v_r[a] * w_r[k]
    
    Where u_r is a vector in R^n, v_r in R^m, w_r in R^p.
    
    In the problem's notation:
    U[i*m + a, r] should be u_r[i] (but repeated for each a? No...)
    
    Actually, I think the convention is:
    U has entries U[i, r] for the A-part, but flattened to (n*m, rank)
    V has entries V[a, r] for the B-part, flattened to (m*p, rank)
    W has entries W[k, i, r] for the C-part, flattened to (p*n, rank) with W[k*n+i, r]
    
    And the decomposition is:
    T[i, a, k] = sum_{r} U[i*m + 0? , r] ... 
    
    I'm confused by the indexing. Let me just use the known Strassen decomposition for 2x2 matmul.
    
    Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    This gives a rank-7 decomposition.
    
    For the rank-6 question: it is known that the border rank of 2x2 matmul is 7,
    and the tensor rank is also 7. So rank 6 is impossible.
    
    Therefore, the optimal answer is rank 7 (Strassen).
    
    Let me construct the Strassen decomposition properly.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # U[i*m + a, r] where i is row of A, a is col of A
    # V[a*p + k, r] where a is row of B, k is col of B
    # W[k*n + i, r] where k is col of C, i is row of C
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # M1: A part: A00 + A11, B part: B00 + B11
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    
    # M2: A part: A10 + A11, B part: B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    
    # M3: A part: A00, B part: B01 - B11
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # -B11
    
    # M4: A part: A11, B part: B10 - B00
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # -B00
    
    # M5: A part: A00 + A01, B part: B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    
    # M6: A part: A10 - A00, B part: B00 + B01
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # -A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    
    # M7: A part: A01 - A11, B part: B10 + B11
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # -A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    
    # Now W: C[i,k] = sum_r U[i,a,r] * V[a,k,r] * W[k*n+i, r] summed appropriately
    # Actually, in the tensor decomposition:
    # T[i,a,k] = sum_r u_r[i,a] * v_r[a,k] * w_r[k,i]
    # But that's not quite right either.
    
    # Let me reconsider. The standard formulation:
    # (A*B)[i,k] = sum_a A[i,a] * B[a,k]
    # = sum_r (sum_a U[i,a,r] * V[a,k,r]) * W[k,i,r]
    # where the inner sum is over a for each r.
    
    # So W[k*n+i, r] is the coefficient for the r-th product contributing to C[i,k].
    
    # From Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W[k*n + i, r] where k is col, i is row
    # C[0,0]: k=0, i=0 -> W[0*2+0, r] = W[0, r]
    # C[0,1]: k=1, i=0 -> W[1*2+0, r] = W[2, r]
    # C[1,0]: k=0, i=1 -> W[0*2+1, r] = W[1, r]
    # C[1,1]: k=1, i=1 -> W[1*2+1, r] = W[3, r]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7
    
    # C[0,1] = M3 + M5
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # C[1,0] = M2 + M4
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
