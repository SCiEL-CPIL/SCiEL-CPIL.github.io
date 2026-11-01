"""Matrix multiplication 2x4 x 4x4 tensor decomposition - attempting rank 27."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # First, build the rank-28 Strassen decomposition as a baseline
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    
                    # U side: A[i, 2*a_blk + j]
                    # U flat index: i * 4 + (2*a_blk + j)
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[2*a_blk + j, 2*b + k]
                    # V flat index: (2*a_blk + j) * 4 + (2*b + k)
                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, 2*b + k]
                    # W flat index: (2*b + k) * 2 + i
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b_idx // p
                    k1 = b_idx % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b_idx][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Try to find rank-27 decomposition
    # Strategy: Use recursive block structure - view 2x4x4 as two 1x4x4 multiplications
    # or as 2x2 blocks of 1x2x2 with shared structure
    
    best_result = build_strassen_28()
    best_rank = 28
    
    # Key insight: The 2x4x4 tensor can be decomposed using the rank-7 Strassen
    # 2x2x2 result applied to block structure. 
    # 
    # View A as 2x4 = [A_top(2x2) A_right(2x2)] where A_top uses cols 0,1
    # and A_right uses cols 2,3. Similarly B = [[B00 B01],[B10 B11]] with 2x2 blocks.
    # 
    # C = A*B gives:
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    # 
    # Each 2x2x2 multiplication has rank 7, so naive block approach gives 4*7=28.
    # But we can share: the left operand blocks A[:,0:2] and A[:,2:4] are shared
    # across both output block columns.
    # 
    # Better: Use the fact that we only need 2 output rows. 
    # For each of the 4 B column blocks (each 2 columns), we compute 2x2x2 mult.
    # Total: 4 * 7 = 28 (same as before).
    # 
    # To get rank 27, we need to find one degree of freedom to eliminate.
    # 
    # Alternative approach: Direct algebraic construction
    # The tensor T_{i,j,k} = delta_{i,j,k} for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # where i is row of A, j is col of A / row of B, k is col of B.
    # 
    # Let's try a structured rank-27 construction:
    # Use 7 bilinear forms for each of the 4 "j-pairs" (j=0,1), (j=2,3) 
    # but share one form between adjacent pairs.
    
    # Actually, let's try a more systematic approach:
    # Generate all possible rank-27 decompositions with entries in {-1, -0.5, 0, 0.5, 1}
    # using a smarter search that exploits the tensor structure.
    
    # The tensor has 32 non-zero entries (one for each (i,j,k) with i in 0..1, j in 0..3, k in 0..3)
    # Wait, no: T[i,j,k] = 1 if this is a valid matrix mult term, else 0.
    # For matrix mult C[i,k] = sum_j A[i,j]*B[j,k], the tensor is T[i,j,k] = 1 for all i,j,k.
    # So all 2*4*4 = 32 entries are 1!
    
    # Let me re-examine: The matrix multiplication tensor for n x m times m x p
    # has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # So yes, all 32 entries are 1.
    
    # For rank r, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # and = 0 for invalid combinations (but there are no invalid ones here since
    # the tensor is fully dense with all 1s)
    
    # Wait, that's not right either. Let me think again.
    # The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (A_{i,j} * B_{j,k} -> C_{i,k})
    # The tensor entry T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    # For 2x4 * 4x4: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # All 32 combinations are valid, so T[i,j,k] = 1 for all 32 entries.
    
    # So we need: for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}:
    #   sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1
    
    # This is a system of 32 equations. With rank 27, we have 27*(8+16+8) = 1296 variables
    # and 32 constraints. Very underdetermined!
    
    # But we also need the decomposition to be "exact" in the sense that it correctly
    # computes matrix multiplication, which means the tensor must match exactly.
    
    # Let's try a constructive approach:
    # Start with the identity-like structure and see if we can do it in 27.
    
    # Method: Use 27 rank-1 tensors that sum to the all-ones tensor of shape (2,4,4).
    # An all-ones tensor of shape (2,4,4) has rank... let's think.
    # The all-ones tensor 1_{2,4,4} can be written as sum of rank-1 tensors.
    # Actually, the all-ones tensor has rank min(2,4,4) = 2? No, that's for matrices.
    # For tensors, the rank of the all-ones tensor of shape (n,m,p) is not simply min(n,m,p).
    
    # Actually, I think I'm confusing things. Let me re-read the problem.
    
    # The matrix multiplication tensor for multiplying an (n x m) matrix by (m x p) matrix
    # is a tensor in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} (or similar indexing).
    # The entry at position (a, b, c) where a indexes A, b indexes B, c indexes C
    # is 1 if the triple (a,b,c) corresponds to a valid multiplication term.
    
    # With the given indexing:
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (16, rank)  
    # W has shape (p*n, rank) = (8, rank)
    
    # U[i*m + j, c] corresponds to A[i, j]
    # V[j*p + k, c] corresponds to B[j, k]
    # W[k*n + i, c] corresponds to C[i, k]
    
    # The decomposition requires:
    # For all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}:
    #   sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1
    # 
    # And for all other combinations (which don't exist in this case since all are valid),
    # the sum should be 0. But since all 32 combinations are valid, we just need all 32 sums to equal 1.
    
    # So we need a rank-27 decomposition of the all-ones tensor of shape (2,4,4)
    # under the specific indexing convention.
    
    # The all-ones tensor of shape (2,4,4): what is its tensor rank?
    # It can be written as the sum of rank-1 tensors.
    # 1_{2,4,4} = sum_{i=0}^{1} sum_{j=0}^{3} sum_{k=0}^{3} e_i ⊗ e_j ⊗ e_k
    # This is trivially rank 32 (the naive decomposition).
    # Can we do better?
    
    # Yes! 1_{2,4,4} = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) is rank 1!
    # Wait, that would give U[i*4+j, c] = 1 for all i,j, V[j*4+k, c] = 1 for all j,k,
    # W[k*2+i, c] = 1 for all k,i. Then the product is 1*1*1 = 1. 
    # So rank 1 should work?!
    
    # Let me check: if U is all ones (8x1), V is all ones (16x1), W is all ones (8x1),
    # then for any (i,j,k): U[i*4+j, 0] * V[j*4+k, 0] * W[k*2+i, 0] = 1*1*1 = 1. ✓
    
    # But wait, that can't be right because the task says the naive rank is 32.
    # Let me re-read the problem statement...
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 4 matrix."
    # "the output factor follows the official trace convention W[k*n+i,r]"
    
    # I think the issue is that the tensor is NOT all ones. Let me reconsider.
    # 
    # In matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor T has entries T[i,j,k] = 1 if (i,j,k) is a valid term in the multiplication.
    # For all i, j, k in the valid ranges, T[i,j,k] = 1.
    # 
    # But the decomposition must satisfy:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k' V[j',k,c] * B[j',k]) * W[k,i,c]
    # 
    # No wait, that's not how tensor decomposition works for matrix multiplication.
    # 
    # The standard formulation: We want to write the matrix multiplication bilinear form
    # as a sum of rank-1 bilinear forms.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We want: C[i,k] = sum_c U_c[i,k] * V_c * W_c[i,k]
    # where U_c and W_c are... hmm, I'm getting confused.
    
    # Let me look at the seed code more carefully.
    # The validate function checks:
    # for i in range(n), j in range(m), k in range(p):
    #   s = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    #   assert s == 1
    # 
    # And for all OTHER (a, b, c) combinations where the indices don't form a valid
    # multiplication term, it checks that the sum is 0.
    # 
    # But what are the "other" combinations? The validate function iterates over
    # a in range(n*m), b in range(m*p), c in range(p*n) and checks if
    # i1 = a//m, j1 = a%m, j2 = b//p, k1 = b%p, i2 = c//n, k2 = c%n
    # satisfy i1==i2, j1==j2, k1==k2. If not, the sum should be 0.
    
    # So the tensor is NOT all ones! It's a sparse tensor where only certain
    # (a,b,c) triples have value 1, and all others have value 0.
    # 
    # Specifically, T[a,b,c] = 1 iff:
    #   a = i*m + j, b = j*p + k, c = k*n + i for some i,j,k
    # i.e., the row index of A matches the row index of C,
    # the col index of A matches the row index of B,
    # the col index of B matches the col index of C.
    
    # For n=2, m=4, p=4:
    # a in {0,...,7}, b in {0,...,15}, c in {0,...,7}
    # T[a,b,c] = 1 iff a%m == b//p and b%p == c%n and a//m == c//n
    # 
    # Let's count: for each (i,j,k) with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3},
    # there's exactly one (a,b,c) = (i*4+j, j*4+k, k*2+i).
    # So there are 2*4*4 = 32 non-zero entries, each equal to 1.
    # All other 8*16*8 - 32 = 1024 - 32 = 992 entries are 0.
    
    # So we need a rank-r decomposition of this sparse tensor.
    # The constraint is:
    # For all i,j,k: sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1
    # For all (a,b,c) not of the form (i*4+j, j*4+k, k*2+i): sum_c U[a,c]*V[b,c]*W[c_idx,c] = 0
    
    # This is a much harder constraint than just matching the 32 non-zero entries!
    # We also need the 992 zero entries to be satisfied.
    
    # OK so the all-ones rank-1 idea doesn't work because it would give non-zero
    # values for the "invalid" (a,b,c) triples.
    
    # Let me verify with the Strassen-28 construction in the seed code.
    # It should satisfy both the 32 = 1 constraints and the 992 = 0 constraints.
    
    # Now, to find rank 27, we need to satisfy all 1024 constraints with 27 columns.
    
    # Let's try a smarter search:
    # 1. Start from the rank-28 Strassen decomposition
    # 2. Try to find a linear dependency among the 28 columns
    # 3. If found, we can eliminate one column
    
    # The columns are vectors in R^{8+16+8} = R^{32}.
    # 28 vectors in R^{32} are likely linearly independent, so no direct dependency.
    # But we need a "tensor dependency" that preserves the decomposition.
    
    # Alternative: Try to construct rank-27 directly using the block structure.
    # 
    # View the 2x4x4 multiplication as:
    # C = [A0 A1] * B where A0, A1 are 2x2 blocks (columns 0-1 and 2-3 of A)
    # B is 4x4, partitioned into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # 
    # C[:, 0:2] = A0 * B00 + A1 * B10
    # C[:, 2:4] = A0 * B01 + A1 * B11
    # 
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # So:
    # C[:, 0:2] needs 7+7 = 14 bilinear forms
    # C[:, 2:4] needs 7+7 = 14 bilinear forms
    # Total: 28 (same as before)
    # 
    # Can we share between the two output blocks?
    # The A0 and A1 blocks are shared, but the B blocks are different.
    # 
    # Idea: Use the same 7 Strassen forms for A0*B00 and A0*B01,
    # and the same 7 for A1*B10 and A1*B11, but that's still 28.
    # 
    # To get 27, we need to find one form that can be shared or eliminated.
    
    # Let's try a different approach: use the fact that we only have 2 output rows.
    # For each j-pair (j=0,1) and (j=2,3), we can use 7 forms.
    # But maybe one form can be shared between the two j-pairs?
    
    # Actually, let's just try a more effective random search with the full
    # constraint set (both 1s and 0s).
    
    def validate_full(U, V, W, rank):
        """Check all 1024 constraints."""
        # Check the 32 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[c_idx][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check the zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c_idx in range(p * n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Try structured search for rank 27
    if time.time() < deadline - 5:
        # Use the Strassen-28 as a starting point and try to merge two columns
        # by finding a linear combination that preserves all constraints
        
        U28 = np.array(best_result["U"], dtype=float)
        V28 = np.array(best_result["V"], dtype=float)
        W28 = np.array(best_result["W"], dtype=float)
        
        # Try all pairs of columns to see if one can be expressed as a combination
        # of the others in a way that preserves the tensor structure
        for c1 in range(28):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, 28):
                if time.time() > deadline:
                    break
                
                # Try to eliminate column c2 by adding alpha * col_c2 to col_c1
                # and setting col_c2 to zero, then checking if we can repair
                # This is a rough heuristic
                
                # Alternative: try random linear combinations
                for trial in range(20):
                    if time.time() > deadline:
                        break
                    
                    random.seed(c1 * 1000 + c2 * 100 + trial)
                    alpha = random.choice([-1.0, -0.5, 0.5, 1.0])
                    
                    U = U28.copy()
                    V = V28.copy()
                    W = W28.copy()
                    
                    # Add alpha * col_c2 to col_c1
                    U[:, c1] += alpha * U[:, c2]
                    V[:, c1] += alpha * V[:, c2]
                    W[:, c1] += alpha * W[:, c2]
                    # Zero out col_c2
                    U[:, c2] = 0
                    V[:, c2] = 0
                    W[:, c2] = 0
                    
                    # Now try to repair by adjusting other columns
                    # This is a hard problem, so just check if it happens to work
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    if validate_full(U_list, V_list, W_list, 28):
                        # We have a valid rank-28 with a zero column, so effective rank <= 27
                        # Remove the zero column
                        U27 = U[:, [c for c in range(28) if c != c2]]
                        V27 = V[:, [c for c in range(28) if c != c2]]
                        W27 = W[:, [c for c in range(28) if c != c2]]
                        
                        if validate_full(U27.tolist(), V27.tolist(), W27.tolist(), 27):
                            best_rank = 27
                            best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                            break
                if best_rank == 27:
                    break
            if best_rank == 27:
                break
    
    # If still not found, try random search with full validation
    if best_rank > 27 and time.time() < deadline - 5:
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            random.seed(987654321 + attempt * 17)
            
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            U = [[rand_half_int() for _ in range(27)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(27)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(27)] for _ in range(p*n)]
            
            if validate_full(U, V, W, 27):
                best_rank = 27
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result

# EVOLVE-BLOCK-END
