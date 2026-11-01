"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    deadline = time.time() + 130.0

    # Build a rank-39 decomposition using block-Strassen for 2x2x2 blocks.
    # Strategy:
    #   - Split inner dimension m=3 into m1=2, m2=1
    #   - Split outer dimension p=7 into p1=2, p2=5
    #   - The 2x2x2 block (rows 0-1, cols 0-1 of inner, output cols 0-1) uses Strassen (rank 7)
    #   - All other blocks use schoolbook
    #
    # Block breakdown:
    #   A is 2x3, B is 3x7
    #   Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    #   Split B = [B1; B2] where B1 is 2x7, B2 is 1x7
    #   C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x7): split B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x5
    #   A1*B1a uses Strassen (7 mults)
    #   A1*B1b uses schoolbook (4*5 = 20 mults)
    # For A2*B2 (2x1 * 1x7): schoolbook (2*1*7 = 14 mults)
    # Total: 7 + 20 + 14 = 41... that's not 39.
    #
    # Let me try a different split:
    #   Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    #   Split B = [B1; B2] where B1 is 2x7, B2 is 1x7
    #   C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x7): 
    #   Strassen works on 2x2 blocks. Write B1 as 7 columns of 2-vectors.
    #   Actually, let's think of it differently.
    #   A1 is 2x2, B1 is 2x7. The product is 2x7.
    #   We can use Strassen on each 2x2 sub-block of B1, but B1 has 7 columns.
    #   Split B1 into B1a (2x2) + B1b (2x5).
    #   A1*B1a: Strassen, 7 mults
    #   A1*B1b: schoolbook, 2*2*5 = 20 mults
    #   Total for A1*B1: 27
    #
    # For A2*B2 (2x1 * 1x7): schoolbook, 2*1*7 = 14 mults
    # Total: 27 + 14 = 41. Still not 39.
    #
    # Alternative: split differently.
    #   Split A into 1+1 rows? No, A is 2x3.
    #
    # Let me try: split inner dimension as 1+2 instead of 2+1.
    #   A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    #   B = [B1; B2] where B1 is 1x7, B2 is 2x7
    #   C = A1*B1 + A2*B2
    #   A1*B1: 2x1 * 1x7 = 14 mults (schoolbook)
    #   A2*B2: 2x2 * 2x7
    #     Split B2 = [B2a | B2b] where B2a is 2x2, B2b is 2x5
    #     A2*B2a: Strassen, 7 mults
    #     A2*B2b: schoolbook, 2*2*5 = 20 mults
    #     Total: 27
    #   Grand total: 14 + 27 = 41. Same.
    #
    # Hmm. Let me think about this more carefully.
    # 
    # Actually, the known result from the inspiration is rank 39. Let me think about
    # how to get there. 
    #
    # Key insight: Strassen for 2x2x2 gives 7 instead of 8. The saving is 1 per 2x2x2 block.
    # For 2x3x7, if we can identify multiple 2x2x2 sub-blocks and apply Strassen to each,
    # we save 1 per block.
    #
    # 2x3x7 = 42. To get to 39, we need to save 3.
    # That means we need 3 independent 2x2x2 blocks where we can apply Strassen.
    #
    # How to get 3 such blocks?
    #   - The 2x2 part of A (rows 0-1, cols 0-1 or 1-2) times the 2x2 part of B
    #   - We have 2 choices for which 2 columns of A to use (cols 0-1 or cols 1-2)
    #   - We have 6 choices for which 2 rows of B to use (but they must align with A's columns)
    #
    # Actually, let's think of it as: the multiplication C = A*B where A is 2x3, B is 3x7.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # We can group the j-sum. For each pair of j values, we have a 2x2x2 sub-problem
    # (for fixed i-pair and k-pair). But the i-pair is always {0,1} (only 2 rows).
    # So for each pair of j's and each pair of k's, we have a 2x2x2 block.
    #
    # j pairs: (0,1), (1,2) -- 2 pairs
    # k pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6) -- 6 pairs
    # But these overlap! We can't just apply Strassen to all of them independently.
    #
    # Better approach: partition the j dimension and k dimension into non-overlapping blocks.
    # j: 3 = 2 + 1
    # k: 7 = 2 + 2 + 2 + 1
    #
    # Block (j=0,1) x (k=0,1): 2x2x2 -> Strassen, 7 mults
    # Block (j=0,1) x (k=2,3): 2x2x2 -> Strassen, 7 mults  
    # Block (j=0,1) x (k=4,5): 2x2x2 -> Strassen, 7 mults
    # Block (j=0,1) x (k=6):    2x2x1 -> schoolbook, 4 mults
    # Block (j=2) x (k=0,1):    1x2x2 -> schoolbook, 4 mults
    # Block (j=2) x (k=2,3):    1x2x2 -> schoolbook, 4 mults
    # Block (j=2) x (k=4,5):    1x2x2 -> schoolbook, 4 mults
    # Block (j=2) x (k=6):      1x1x1 -> schoolbook, 1 mult
    #
    # Total: 7+7+7+4+4+4+4+1 = 38!
    #
    # Wait, that's even better than 39! Let me verify this is correct.
    #
    # Actually, I need to be more careful. The "2x2x2" block means:
    #   2 rows of A (i=0,1), 2 columns of A / 2 rows of B (j=0,1), 2 columns of B (k=0,1)
    # The product contribution is A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1}
    # This is a 2x2x2 tensor, which has Strassen rank 7.
    #
    # For the "2x2x1" block: i in {0,1}, j in {0,1}, k={6}
    #   This is just 4 scalar multiplications (schoolbook for 2x2x1).
    #
    # For the "1x2x2" block: i={?}, j={2}, k in {0,1}
    #   Wait, i is always 2 rows. Let me reclarify.
    #
    # The tensor T[i,j,k] = A[i,j] * B[j,k] for all i,j,k.
    # We're decomposing this tensor into rank-1 terms.
    #
    # A "2x2x2 sub-block" means fixing i in {0,1}, j in {j1,j2}, k in {k1,k2}.
    # The sub-tensor has 8 entries, and Strassen shows it can be done with 7 rank-1 terms.
    #
    # A "2x2x1 sub-block" means i in {0,1}, j in {j1,j2}, k={k0}.
    # The sub-tensor has 4 entries, schoolbook rank 4.
    #
    # A "1x2x2 sub-block" would mean i={i0}, j in {j1,j2}, k in {k1,k2}.
    # But i is always 2 values in our problem. So this doesn't apply directly.
    #
    # Let me redo:
    # j partition: {0,1} and {2}
    # k partition: {0,1}, {2,3}, {4,5}, {6}
    #
    # Blocks (always i={0,1} since n=2):
    #   j={0,1}, k={0,1}: 2x2x2 -> rank 7 (Strassen)
    #   j={0,1}, k={2,3}: 2x2x2 -> rank 7 (Strassen)
    #   j={0,1}, k={4,5}: 2x2x2 -> rank 7 (Strassen)
    #   j={0,1}, k={6}:   2x2x1 -> rank 4 (schoolbook)
    #   j={2},    k={0,1}: 2x1x2 -> rank 4 (schoolbook)
    #   j={2},    k={2,3}: 2x1x2 -> rank 4 (schoolbook)
    #   j={2},    k={4,5}: 2x1x2 -> rank 4 (schoolbook)
    #   j={2},    k={6}:   2x1x1 -> rank 2 (schoolbook)
    #
    # Total: 7+7+7+4+4+4+4+2 = 39!
    #
    # Great, so rank 39 is achievable. Let me construct it.
    #
    # For each 2x2x2 Strassen block, I need to define the 7 rank-1 terms.
    # Strassen's algorithm for C = A*B where A, B are 2x2:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11)B00
    #   M3 = A00(B01 - B11)
    #   M4 = A11(B10 - B00)
    #   M5 = (A00 + A01)B11
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Each M corresponds to a product of a linear combo of A entries and a linear
    # combo of B entries. In the tensor decomposition, each M contributes one rank-1 term
    # where U has the A-coefficients, V has the B-coefficients, and W has the C-coefficients.
    #
    # For a 2x2x2 block with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
    #   A entries: A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1]
    #   B entries: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
    #   C entries: C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
    #
    # Strassen terms (in terms of (i,j) for U, (j,k) for V, (k,i) for W):
    #   M1: U has +1 at (i0,j0),(i1,j1); V has +1 at (j0,k0),(j1,k1); W has +1 at (k0,i0),(k1,i1)
    #       Wait, that's not quite right. Let me be more careful.
    #
    # Actually, in the tensor T[i,j,k] = A[i,j]*B[j,k], a rank-1 term is:
    #   sum over (i,j) of u[i,j] * A[i,j]  times  sum over (j,k) of v[j,k] * B[j,k]
    #   which contributes to C[i,k] via the product.
    #
    # But in the standard tensor decomposition, a rank-1 term is:
    #   U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and the sum over c should equal T[i,j,k] = 1 if (i,j,k) is a valid triple.
    #
    # Wait, I'm confusing two things. The matrix multiplication tensor is:
    #   T[i,j,k] = 1 for all valid (i,j,k)  [it's just the identity mapping]
    # And we want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k).
    #
    # No wait, that's not right either. The matrix multiplication tensor maps:
    #   (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # As a trilinear form, it's: sum_{i,j,k} A[i,j]*B[j,k]*C'[i,k] where C' is the dual.
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1.
    #
    # A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # means: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i*m+j, j*p+k, k*n+i] = 1
    #
    # So for the schoolbook, each term (i,j,k) contributes 1 to exactly one (i,j,k) triple.
    #
    # For Strassen, the idea is different. We're not decomposing the tensor T directly.
    # We're finding a bilinear algorithm: C = sum_c (sum_{i,j} U[i,j,c]*A[i,j]) * (sum_{j,k} V[j,k,c]*B[j,k])
    # where each product is then distributed to C via W.
    #
    # Actually, let me think about this differently. The standard way to express a bilinear
    # algorithm for matrix multiplication is:
    #   C[i,k] = sum_c W[i,k,c] * (sum_{i',j} U[i',j,c] * A[i',j]) * (sum_{j',k'} V[j',k',c] * B[j',k'])
    #
    # But that's not quite standard either. Let me use the tensor formulation properly.
    #
    # The matrix multiplication tensor for n×m × m×p -> n×p is:
    #   μ = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the basis vector in R^{n*m}, etc.
    #
    # A rank-r decomposition is: μ = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{n*p} (or R^{p*n} depending on convention).
    #
    # The task says W has shape (p*n, rank) with indexing W[k*n+i, c].
    # So W[k*n+i, c] corresponds to the coefficient for the (i,k) output entry.
    #
    # Verification: for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # For schoolbook: U[i*m+j, c] = 1 if c corresponds to (i,j,k), else 0.
    #
    # Now, for Strassen on a 2x2x2 block (i∈{0,1}, j∈{0,1}, k∈{0,1}):
    # The sub-tensor is: sum_{i,j,k∈{0,1}} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # This has rank 7 (Strassen).
    #
    # The 7 rank-1 terms for this sub-tensor:
    # Let me use the standard Strassen decomposition.
    # 
    # C00 = (A00+A11)(B00+B11) + A11(B10-B00) - A00(B01-B11) + A01(B10-B11)
    #     = M1 + M4 - M3' + M7'  where I'm being more careful with signs
    #
    # Let me just use the standard form:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_c = (linear combo of A) * (linear combo of B)
    # In the tensor decomposition, M_c contributes a rank-1 term where:
    #   u_c has the A-coefficients (in the (i,j) positions)
    #   v_c has the B-coefficients (in the (j,k) positions)
    #   w_c has the C-coefficients (in the (i,k) positions, with appropriate signs)
    #
    # For M1 = (A00+A11)(B00+B11):
    #   u: A00=1, A11=1 -> positions (0,0) and (1,1) in the 2x2 A block
    #   v: B00=1, B11=1 -> positions (0,0) and (1,1) in the 2x2 B block
    #   w: C00=1, C11=1 -> positions (0,0) and (1,1) in the 2x2 C block
    #
    # For M4 = A11(B10-B00):
    #   u: A11=1 -> position (1,1)
    #   v: B10=1, B00=-1 -> positions (1,0) and (0,0)
    #   w: C00=1, C10=1 -> positions (0,0) and (1,0)
    #
    # Etc.
    #
    # Let me code this up properly.
    
    def strassen_2x2x2(i0, j0, k0):
        """Return 7 rank-1 terms (as (U_coeffs, V_coeffs, W_coeffs) dicts) for the 2x2x2 block
        with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}."""
        terms = []
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        #    -> C[i0,k0] += 1, C[i0+1,k0+1] += 1
        u = {(i0, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        #    -> C[i0+1,k0] += 1
        u = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1}
        w = {(k0, i0+1): 1}
        terms.append((u, v, w))
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        #    -> C[i0,k0+1] += 1
        u = {(i0, j0): 1}
        v = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w = {(k0+1, i0): 1}
        terms.append((u, v, w))
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        #    -> C[i0,k0] += 1, C[i0+1,k0] += 1
        u = {(i0+1, j0+1): 1}
        v = {(j0+1, k0): 1, (j0, k0): -1}
        w = {(k0, i0): 1, (k0, i0+1): 1}
        terms.append((u, v, w))
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        #    -> C[i0,k0] -= 1, C[i0,k0+1] += 1
        u = {(i0, j0): 1, (i0, j0+1): 1}
        v = {(j0+1, k0+1): 1}
        w = {(k0, i0): -1, (k0+1, i0): 1}
        terms.append((u, v, w))
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        #    -> C[i0+1,k0+1] += 1
        u = {(i0+1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k0+1): 1}
        w = {(k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        #    -> C[i0,k0] += 1, C[i0+1,k0+1] += 1
        u = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        return terms
    
    def schoolbook_block(i_list, j_list, k_list):
        """Return rank-1 terms for a schoolbook block with given i, j, k indices."""
        terms = []
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms
    
    # Build all terms
    all_terms = []
    
    # j partition: {0,1} and {2}
    # k partition: {0,1}, {2,3}, {4,5}, {6}
    
    # Strassen blocks: j={0,1}, k in {{0,1}, {2,3}, {4,5}}
    for k0 in [0, 2, 4]:
        all_terms.extend(strassen_2x2x2(0, 0, k0))
    
    # Schoolbook blocks:
    # j={0,1}, k={6}: 2x2x1
    all_terms.extend(schoolbook_block([0, 1], [0, 1], [6]))
    
    # j={2}, k={0,1}: 2x1x2
    all_terms.extend(schoolbook_block([0, 1], [2], [0, 1]))
    
    # j={2}, k={2,3}: 2x1x2
    all_terms.extend(schoolbook_block([0, 1], [2], [2, 3]))
    
    # j={2}, k={4,5}: 2x1x2
    all_terms.extend(schoolbook_block([0, 1], [2], [4, 5]))
    
    # j={2}, k={6}: 2x1x1
    all_terms.extend(schoolbook_block([0, 1], [2], [6]))
    
    rank = len(all_terms)
    print(f"Constructed decomposition with rank {rank}")
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val
    
    # Verify
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    ok, info = verify(U, V, W)
    if not ok:
        print(f"Verification failed at {info}")
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
    else:
        print("Verification passed!")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
