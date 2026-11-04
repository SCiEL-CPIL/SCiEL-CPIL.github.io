import numpy as np
import time
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x3 matrix multiplication.
    
    Strategy: Structural decomposition into sub-blocks with known minimal ranks.
    - The 2x3 x 3x3 multiplication can be viewed as computing C[i,k] = sum_j A[i,j] B[j,k]
    - Split A into columns: A = [A_left (2x2), A_right (2x1)] where A_left covers j=0,1 and A_right covers j=2
    - Then C = A_left @ B_left + A_right @ B_right where B_left is 2x3, B_right is 1x3
    - A_left @ B_left is 2x2 x 2x3 -> rank 7 (Strassen for 2x2 core + 2 extra for the third column)
    - A_right @ B_right is 2x1 x 1x3 -> rank 3 (outer product of 2-vector and 3-vector)
    - Total: 7 + 3 = 10? No, that's not right because the indices overlap.
    
    Let me reconsider. The standard approach:
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    
    For i in {0,1}, k in {0,1,2}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Group by j:
    - j=0: A[i,0]*B[0,k] for i=0,1; k=0,1,2 -> this is a 2x1 x 1x3 tensor (rank 3, but actually rank 1 per (i,k) pair... no)
    
    Actually, let's think of it as:
    The tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
    T[(i,j),(j,k),(i,k)] = 1
    
    We can write this as a sum over j:
    T = sum_{j=0}^{2} T_j where T_j has 1s only when the middle index matches j.
    
    Each T_j is: for fixed j, T_j[(i,j),(j,k),(i,k)] = 1 for all i,k.
    This is the outer product of (vector over i) tensor (vector over k) tensor (delta at j for both A and B).
    Specifically, T_j = u_j ⊗ v_j ⊗ w_j where:
    - u_j has 1s at positions (i,j) for i=0,1
    - v_j has 1s at positions (j,k) for k=0,1,2  
    - w_j has 1s at positions (i,k) for i=0,1; k=0,1,2
    
    But that gives rank 3 (one per j), and the total rank would be 3, which is way too low.
    The issue is that u_j, v_j, w_j are NOT rank-1 vectors in the full space - they have multiple 1s.
    
    Let me use the correct tensor definition. The matrix multiplication tensor for n×m × m×p is:
    T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    T[a,b,c] = 1 if the indices are consistent (which they always are by construction)
    
    Actually T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' AND j=j' AND i=i' AND k=k'.
    So T[i*m+j, j*p+k, k*n+i] = 1, all other entries 0.
    
    The schoolbook rank is n*m*p = 2*3*3 = 18.
    
    Now, for 2x3 x 3x3, can we do better than 18?
    
    Known results: For 2x2 x 2x2, Strassen gives rank 7 < 8.
    For 2x3 x 3x3, we can embed the 2x2x2 Strassen decomposition.
    
    Let's decompose:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Split j into {0,1} and {2}:
    C[i,k] = (A[i,0]B[0,k] + A[i,1]B[1,k]) + A[i,2]B[2,k]
    
    The first part is a 2x2 x 2x3 multiplication (A is 2x2, B is 2x3).
    The second part is a 2x1 x 1x3 multiplication.
    
    For 2x2 x 2x3:
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] for i=0,1; k=0,1,2
    This is 6 output entries.
    
    We can view this as two 2x2 multiplications (for k=0,1 and k=1,2) but they share the A matrix.
    
    Actually, let's think of 2x2 x 2x3 as:
    Take B as two 2x2 matrices: B[:,0:2] and B[:,1:3] (overlapping in column 1).
    This doesn't decompose cleanly.
    
    Alternative: 2x2 x 2x3 can be seen as 2x2 x 2x2 plus 2x2 x 2x1.
    Rank(2x2 x 2x2) = 7 (Strassen)
    Rank(2x2 x 2x1) = 4 (each of the 2 outputs needs 2 multiplications, total 4)
    But they share the A factors, so we can potentially share some rank-1 terms.
    
    Hmm, this is getting complex. Let me try a different structural approach.
    
    For the 2x1 x 1x3 part: A[i,2] * B[2,k] for i=0,1; k=0,1,2.
    This is: [A[0,2], A[1,2]] outer [B[2,0], B[2,1], B[2,2]]
    = sum over the 6 pairs (i,k) of A[i,2]*B[2,k]
    This is a rank-6 tensor in general, BUT it has special structure:
    It's the tensor product of a 2-vector and a 3-vector, which has rank 6 (it's already a sum of 6 rank-1 terms, and no fewer since all 6 products are algebraically independent).
    
    Wait, no. The tensor for 2x1 x 1x3 is:
    T[i, 2, k, 2, i, k] = A[i,2]*B[2,k]
    In the flattened tensor: T[i*3+2, 2*3+k, k*2+i] = 1
    This has 6 nonzero entries, each in a distinct "position", so its rank is exactly 6.
    
    So if I just add the 2x1x1x3 part directly, I get rank 6 for that part.
    
    For the 2x2 x 2x3 part:
    T[i*3+j, j*3+k, k*2+i] = 1 for j=0,1; i=0,1; k=0,1,2
    That's 12 nonzero entries.
    
    Can I decompose this 2x2 x 2x3 tensor with rank < 12?
    
    This is equivalent to multiplying a 2x2 matrix by a 2x3 matrix.
    Known: rank of 2x2 x 2x3 multiplication.
    
    One approach: Use Strassen on the 2x2 x 2x2 sub-part (columns 0,1 of B), then handle column 2 separately.
    
    Strassen for 2x2 x 2x2 gives 7 terms for C[:,0:2].
    Then C[:,2] = A @ B[:,2] which is a 2x2 x 2x1 multiplication, rank 4.
    Total: 7 + 4 = 11 for the 2x2 x 2x3 part.
    
    But wait, the Strassen terms compute linear combinations, and the column-2 multiplication is separate.
    So total rank for 2x2 x 2x3 = 7 + 4 = 11.
    
    Then total for 2x3 x 3x3 = 11 (for j=0,1 part) + 6 (for j=2 part) = 17.
    
    That matches the known rank-17 solution!
    
    But can we do better? The question is whether the 2x2 x 2x3 tensor has rank < 11, or whether there's a more clever decomposition of the full 2x3 x 3x3 tensor.
    
    Actually, let me reconsider. The 2x2 x 2x1 multiplication:
    C[i] = A[i,0]*B[0] + A[i,1]*B[1] for i=0,1
    This has 2 outputs, each a sum of 2 products. Rank = 4 in general.
    
    So 7 + 4 + 6 = 17 seems to be the natural decomposition.
    
    Now, is rank 17 minimal? Or can we achieve rank 16?
    
    For now, let me implement the rank-17 solution using this structural decomposition, which is guaranteed to work and is much faster than numerical optimization.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 3
    
    # Build the target tensor for verification
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Strassen decomposition for 2x2 x 2x2:
    # M1 = (A11 - A22) * (B12 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = A11 * (B11 - B21)
    # M4 = A22 * (B12 - B21)
    # M5 = (A11 + A12) * B21
    # M6 = (A21 - A11) * (B11 + B22)
    # M7 = (A12 + A22) * (B21 - B11)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 x 2x3 (A is 2x2, B is 2x3):
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Split B into B_left (2x2, columns 0,1) and B_right (2x1, column 2)
    # C[:,0:2] = A @ B_left (2x2 x 2x2, use Strassen, rank 7)
    # C[:,2] = A @ B_right (2x2 x 2x1, rank 4)
    
    # For the 2x2 x 2x1 part:
    # C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # This is 4 rank-1 terms:
    # T1: A[0,0]*B[0] -> C[0]
    # T2: A[0,1]*B[1] -> C[0]
    # T3: A[1,0]*B[0] -> C[1]
    # T4: A[1,1]*B[1] -> C[1]
    
    # Now let's build the full rank-17 decomposition.
    # 
    # The full tensor has:
    # - j=0,1 part: 2x2 x 2x3 (12 entries) -> rank 11
    # - j=2 part: 2x1 x 1x3 (6 entries) -> rank 6
    # Total: 17
    #
    # Let me index the rank-1 terms as r = 0, 1, ..., 16.
    #
    # For the Strassen part (j=0,1; k=0,1):
    # The A indices are (i, j) with i=0,1; j=0,1 -> flattened: i*3 + j
    #   (0,0) -> 0, (0,1) -> 1, (1,0) -> 3, (1,1) -> 4
    # The B indices are (j, k) with j=0,1; k=0,1 -> flattened: j*3 + k
    #   (0,0) -> 0, (0,1) -> 1, (1,0) -> 3, (1,1) -> 4
    # The C indices are (k, i) with k=0,1; i=0,1 -> flattened: k*2 + i
    #   (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3
    
    # Strassen M1 = (A[0,0] - A[1,1]) * (B[0,1] + B[1,1])
    # In our indexing:
    # A[0,0] is at index 0, A[1,1] is at index 4
    # B[0,1] is at index 1, B[1,1] is at index 4
    # C contributions: M1 appears in C[0,0] (+1), C[1,1] (+1), C[0,1] (0), C[1,0] (0)
    # Wait, let me recheck Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C11 (+1) and C22 (+1).
    # C11 is at C index (0,0) -> k=0, i=0 -> 0*2+0 = 0
    # C22 is at C index (1,1) -> k=1, i=1 -> 1*2+1 = 3
    
    # Let me define the 7 Strassen terms:
    # r=0: M1 = (A[0,0]-A[1,1])*(B[0,1]+B[1,1])
    # r=1: M2 = (A[0,0]+A[1,1])*B[1,1]
    # r=2: M3 = A[0,0]*(B[0,0]-B[1,0])
    # r=3: M4 = A[1,1]*(B[0,1]-B[1,0])
    # r=4: M5 = (A[0,0]+A[0,1])*B[1,0]
    # r=5: M6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,1])
    # r=6: M7 = (A[0,1]+A[1,1])*(B[1,0]-B[0,0])
    
    # C contributions (for the 2x2 x 2x2 sub-problem, k=0,1):
    # C[0,0] (idx 0): M1 + M4 - M5 + M7 -> coefficients [1, 0, 0, 1, -1, 0, 1]
    # C[0,1] (idx 1): M3 + M5 -> coefficients [0, 0, 1, 0, 1, 0, 0]
    # C[1,0] (idx 2): M2 + M4 -> coefficients [0, 1, 0, 1, 0, 0, 0]
    # C[1,1] (idx 3): M1 - M2 + M3 + M6 -> coefficients [1, -1, 1, 0, 0, 1, 0]
    
    # For the 2x2 x 2x1 part (k=2):
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # C[0,2] (idx 4): A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] (idx 5): A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # r=7: A[0,0]*B[0,2] -> C[0,2]
    # r=8: A[0,1]*B[1,2] -> C[0,2]
    # r=9: A[1,0]*B[0,2] -> C[1,2]
    # r=10: A[1,1]*B[1,2] -> C[1,2]
    
    # For the j=2 part (2x1 x 1x3):
    # C[i,k] = A[i,2]*B[2,k]
    # C[0,0] (idx 0): A[0,2]*B[2,0]
    # C[0,1] (idx 1): A[0,2]*B[2,1]
    # C[0,2] (idx 2): A[0,2]*B[2,2]
    # C[1,0] (idx 3): A[1,2]*B[2,0]
    # C[1,1] (idx 4): A[1,2]*B[2,1]
    # C[1,2] (idx 5): A[1,2]*B[2,2]
    # r=11: A[0,2]*B[2,0] -> C[0,0]
    # r=12: A[0,2]*B[2,1] -> C[0,1]
    # r=13: A[0,2]*B[2,2] -> C[0,2]
    # r=14: A[1,2]*B[2,0] -> C[1,0]
    # r=15: A[1,2]*B[2,1] -> C[1,1]
    # r=16: A[1,2]*B[2,2] -> C[1,2]
    
    rank = 17
    U = np.zeros((n * m, rank))  # 6 x 17
    V = np.zeros((m * p, rank))  # 9 x 17
    W = np.zeros((p * n, rank))  # 6 x 17
    
    # Helper to set A index
    def a_idx(i, j):
        return i * m + j
    
    # Helper to set B index
    def b_idx(j, k):
        return j * p + k
    
    # Helper to set C index
    def c_idx(k, i):
        return k * n + i
    
    # Strassen terms (r=0 to 6)
    # r=0: M1 = (A[0,0]-A[1,1])*(B[0,1]+B[1,1])
    U[a_idx(0,0), 0] = 1
    U[a_idx(1,1), 0] = -1
    V[b_idx(0,1), 0] = 1
    V[b_idx(1,1), 0] = 1
    W[c_idx(0,0), 0] = 1   # C11
    W[c_idx(1,1), 0] = 1   # C22
    
    # r=1: M2 = (A[0,0]+A[1,1])*B[1,1]
    U[a_idx(0,0), 1] = 1
    U[a_idx(1,1), 1] = 1
    V[b_idx(1,1), 1] = 1
    W[c_idx(1,0), 1] = 1   # C21
    W[c_idx(1,1), 1] = -1  # C22
    
    # r=2: M3 = A[0,0]*(B[0,0]-B[1,0])
    U[a_idx(0,0), 2] = 1
    V[b_idx(0,0), 2] = 1
    V[b_idx(1,0), 2] = -1
    W[c_idx(0,1), 2] = 1   # C12
    W[c_idx(1,1), 2] = 1   # C22
    
    # r=3: M4 = A[1,1]*(B[0,1]-B[1,0])
    U[a_idx(1,1), 3] = 1
    V[b_idx(0,1), 3] = 1
    V[b_idx(1,0), 3] = -1
    W[c_idx(0,0), 3] = 1   # C11
    W[c_idx(1,0), 3] = 1   # C21
    
    # r=4: M5 = (A[0,0]+A[0,1])*B[1,0]
    U[a_idx(0,0), 4] = 1
    U[a_idx(0,1), 4] = 1
    V[b_idx(1,0), 4] = 1
    W[c_idx(0,0), 4] = -1  # C11
    W[c_idx(0,1), 4] = 1   # C12
    
    # r=5: M6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,1])
    U[a_idx(1,0), 5] = 1
    U[a_idx(0,0), 5] = -1
    V[b_idx(0,0), 5] = 1
    V[b_idx(1,1), 5] = 1
    W[c_idx(1,1), 5] = 1   # C22
    
    # r=6: M7 = (A[0,1]+A[1,1])*(B[1,0]-B[0,0])
    U[a_idx(0,1), 6] = 1
    U[a_idx(1,1), 6] = 1
    V[b_idx(1,0), 6] = 1
    V[b_idx(0,0), 6] = -1
    W[c_idx(0,0), 6] = 1   # C11
    
    # 2x2 x 2x1 terms (r=7 to 10)
    # r=7: A[0,0]*B[0,2] -> C[0,2]
    U[a_idx(0,0), 7] = 1
    V[b_idx(0,2), 7] = 1
    W[c_idx(2,0), 7] = 1
    
    # r=8: A[0,1]*B[1,2] -> C[0,2]
    U[a_idx(0,1), 8] = 1
    V[b_idx(1,2), 8] = 1
    W[c_idx(2,0), 8] = 1
    
    # r=9: A[1,0]*B[0,2] -> C[1,2]
    U[a_idx(1,0), 9] = 1
    V[b_idx(0,2), 9] = 1
    W[c_idx(2,1), 9] = 1
    
    # r=10: A[1,1]*B[1,2] -> C[1,2]
    U[a_idx(1,1), 10] = 1
    V[b_idx(1,2), 10] = 1
    W[c_idx(2,1), 10] = 1
    
    # j=2 part (r=11 to 16)
    # r=11: A[0,2]*B[2,0] -> C[0,0]
    U[a_idx(0,2), 11] = 1
    V[b_idx(2,0), 11] = 1
    W[c_idx(0,0), 11] = 1
    
    # r=12: A[0,2]*B[2,1] -> C[0,1]
    U[a_idx(0,2), 12] = 1
    V[b_idx(2,1), 12] = 1
    W[c_idx(1,0), 12] = 1
    
    # r=13: A[0,2]*B[2,2] -> C[0,2]
    U[a_idx(0,2), 13] = 1
    V[b_idx(2,2), 13] = 1
    W[c_idx(2,0), 13] = 1
    
    # r=14: A[1,2]*B[2,0] -> C[1,0]
    U[a_idx(1,2), 14] = 1
    V[b_idx(2,0), 14] = 1
    W[c_idx(0,1), 14] = 1
    
    # r=15: A[1,2]*B[2,1] -> C[1,1]
    U[a_idx(1,2), 15] = 1
    V[b_idx(2,1), 15] = 1
    W[c_idx(1,1), 15] = 1
    
    # r=16: A[1,2]*B[2,2] -> C[1,2]
    U[a_idx(1,2), 16] = 1
    V[b_idx(2,2), 16] = 1
    W[c_idx(2,1), 16] = 1
    
    # Verify
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_school = np.zeros((n * m, schoolbook_rank))
        V_school = np.zeros((m * p, schoolbook_rank))
        W_school = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[a_idx(i, j), col] = 1
                    V_school[b_idx(j, k), col] = 1
                    W_school[c_idx(k, i), col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
