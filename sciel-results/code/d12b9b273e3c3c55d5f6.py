"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries (sampled for speed, but exhaustive is better if fast enough)
    # Total entries: (p*n) * (n*m) * (m*p) = 12 * 10 * 30 = 3600
    # Rank is ~50, so 3600 * 50 = 180k ops. Very fast.
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                # Check consistency
                if i == i2 and j == j2 and k == k2:
                    # Valid case, already checked
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen 2x2x2 block
            # Indices:
            # A is 2x2: rows i=0,1; cols j=j0, j1
            # B is 2x2: rows j=j0, j1; cols k=k0, k1
            # C is 2x2: rows i=0,1; cols k=k0, k1
            
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            # C_0,k0 += P1, C_1,k0 -= P1, C_0,k1 += P1, C_1,k1 += P1 ? No, standard Strassen mapping:
            # C_00 = P1 - P2 + P4 - P5
            # C_01 = P3 + P5
            # C_10 = P2 + P4
            # C_11 = P1 - P3 - P4 + P6 (Wait, standard Strassen:
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B11)
            # M4 = A22(B12 - B22)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            # C11 = M1 - M2 + M4 - M5
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M3 - M4 + M6 (Wait, M7 is for C22? No.
            # Let's use the explicit formulas from the inspiration which worked.
            
            # Inspiration 1 used:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            #   -> U[0,j0]+=1, U[1,j1]+=1, V[j0,k0]+=1, V[j1,k1]+=1
            #   -> W[k0,0]+=1, W[k1,1]+=1 ... wait, this looks like a specific subset or modified Strassen.
            # Let's stick to the standard Strassen implementation which is known to work.
            
            # Let's map local indices 0,1 to global j0, j1 and k0, k1.
            # Local A: [[A00, A01], [A10, A11]] -> Global A[i, j] where i in {0,1}, j in {j0, j1}
            # Local B: [[B00, B01], [B10, B11]] -> Global B[j, k] where j in {j0, j1}, k in {k0, k1}
            # Local C: [[C00, C01], [C10, C11]] -> Global C[i, k] where i in {0,1}, k in {k0, k1}
            
            # M1 = (A00 + A11)(B00 + B11)
            # A00 is (0, j0), A11 is (1, j1)
            # B00 is (j0, k0), B11 is (j1, k1)
            # C00 += M1, C11 += M1, C01 -= M1? No.
            # C00 = M1 - M2 + M4 - M5
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 - M4 + M6 (Wait, standard is M1 - M2 + M4 - M5 for C11? No.
            
            # Let's use the verified code from Inspiration 1 directly, it's simpler and proven.
            
            # P1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=10):
    """
    Try to reduce the rank using structured linear algebra:
    Fix U and V to sparse {0, ±1} patterns, solve for W over rationals.
    """
    import numpy as np
    from fractions import Fraction
    
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Strategy: For each target rank R < current_rank, try to find sparse U, V patterns
    # such that the linear system for W has a solution.
    
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    # For fixed U, V, this is a linear system in W.
    
    # Number of valid (i,j,k) triples: n*m*p = 60
    # Number of W variables: p*n*R = 12*R
    
    # We need 12*R >= 60, so R >= 5. But with sparsity constraints, R must be larger.
    
    # Try to build sparse patterns from the existing 54-rank solution by merging terms.
    # First, let's check if we can find a rank-50 or rank-52 solution by trying
    # to eliminate pairs of terms that are "similar".
    
    # Alternative approach: Use the structure of the 54-rank solution and try to
    # find linear dependencies that allow rank reduction.
    
    # For each pair of columns (r1, r2), check if we can eliminate one by adjusting
    # the other. This requires solving a small linear system.
    
    # Actually, let's try a different approach: systematically try all subsets of
    # columns to remove, checking if the remaining columns can still span the tensor.
    
    # This is expensive, so let's use a greedy approach:
    # 1. Start with the 54-rank solution
    # 2. For each column r, check if it can be expressed as a linear combination
    #    of the other columns in a way that preserves the tensor structure.
    
    # The tensor structure constraint: for each (i,j,k),
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    
    # If we remove column r0, we need to find coefficients c_r for r != r0 such that:
    # U[i*m+j, r0] * V[j*p+k, r0] * W[k*n+i, r0] = sum_{r != r0} c_r * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a system of n*m*p = 60 equations in (R-1) unknowns.
    
    # Let's try this for each column and see if any can be eliminated.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    R = U_arr.shape[1]
    
    # Build the tensor matrix: T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need sum_r T[i,j,k,r] = delta(i,j,k) for all (i,j,k)
    
    # For each column r0 to eliminate, set up the system:
    # For each (i,j,k): sum_{r != r0} c_r * T[i,j,k,r] = T[i,j,k,r0]
    
    # This is a linear system A * c = b where:
    # A[i,j,k, r] = T[i,j,k,r] for r != r0
    # b[i,j,k] = T[i,j,k,r0]
    
    # If the system has a solution, we can eliminate column r0 by adding c_r * column_r to column_r0's contribution.
    # But we also need to update the other columns' coefficients.
    
    # Actually, this is more subtle. Let's think about it differently.
    
    # The decomposition is: T = sum_r u_r * v_r * w_r (outer products)
    # where u_r is a vector of length n*m, v_r of length m*p, w_r of length p*n.
    
    # If T = sum_r u_r * v_r * w_r and we want to eliminate term r0,
    # we need: u_{r0} * v_{r0} * w_{r0} = sum_{r != r0} c_r * u_r * v_r * w_r
    
    # This is a tensor equation, not a simple linear system.
    
    # Let's try a different approach: use the fact that we have a 54-rank solution
    # and try to find a 53-rank solution by checking if any two terms are "redundant".
    
    # For each pair (r1, r2), check if we can find coefficients a, b such that:
    # a * u_{r1} * v_{r1} * w_{r1} + b * u_{r2} * v_{rr2} * w_{r2} can be replaced
    # by a single term u_new * v_new * w_new.
    
    # This is very hard in general. Let's try a simpler heuristic:
    # Check if any two columns are identical or nearly identical.
    
    # Actually, let's try to use the linear algebra approach more directly.
    # Fix U and V to have specific sparse patterns, then solve for W.
    
    # For rank R, we need to choose U (n*m x R) and V (m*p x R) with entries in {0, ±1},
    # then solve for W (p*n x R) such that the tensor equation holds.
    
    # The system is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    
    # This is a linear system in W with n*m*p = 60 equations and p*n*R = 12*R unknowns.
    
    # For R = 50, we have 600 unknowns and 60 equations - very underdetermined.
    # The question is whether there exist sparse U, V such that the system is consistent.
    
    # Let's try to construct U, V using a difference set or projective plane structure.
    
    # For n=2, m=5, p=6:
    # U has 10 rows, V has 30 rows, W has 12 rows.
    
    # A simple approach: use the 54-rank solution as a starting point and try to
    # find a rank-50 solution by removing 4 terms and adjusting the rest.
    
    # This is complex. Let's instead try to verify if the 54-rank solution is already
    # minimal by checking if we can find any rank-53 solution using a randomized approach.
    
    # Randomized approach:
    # 1. Start with the 54-rank solution
    # 2. Randomly perturb the solution slightly
    # 3. Try to reduce the rank by projecting onto a lower-dimensional subspace
    
    # This is still complex. Let's just return the input as is for now, but with a
    # more thorough verification that the 54-rank solution is valid.
    
    # Actually, let's try one more thing: check if we can find a rank-50 solution
    # by using a structured approach based on the dimensions.
    
    # For 2x5x6, the known lower bound is 48 (from various results).
    # Let's try to construct a rank-48 solution using a combination of Strassen blocks.
    
    # Split j into {0,1,2}, {3,4} and k into {0,1}, {2,3}, {4,5}
    # Use 8-term Strassen for 3x2x2 blocks? No, Strassen is for 2x2x2.
    
    # Alternative: use a 3x3x3 block with rank 23 (Laderman's algorithm)
    # But our dimensions are 2x5x6, not 3x3x3.
    
    # Let's try a different split: j into {0,1,2,3,4} as {0,1}, {2,3}, {4}
    # and k into {0,1,2,3,4,5} as {0,1}, {2,3}, {4,5}
    
    # For the {0,1}x{0,1} block: 7 terms (Strassen)
    # For the {2,3}x{0,1} block: 7 terms
    # For the {4}x{0,1} block: 2 terms (schoolbook)
    # For the {0,1}x{2,3} block: 7 terms
    # For the {2,3}x{2,3} block: 7 terms
    # For the {4}x{2,3} block: 2 terms
    # For the {0,1}x{4,5} block: 7 terms
    # For the {2,3}x{4,5} block: 7 terms
    # For the {4}x{4,5} block: 2 terms
    
    # Total: 7+7+2+7+7+2+7+7+2 = 48 terms!
    
    # This is the same as the 54-rank solution but with a different split.
    # Wait, the 54-rank solution uses:
    # j_pairs = [(0, 1), (2, 3)] and k_pairs = [(0, 1), (2, 3), (4, 5)]
    # That's 2 j-pairs x 3 k-pairs = 6 blocks of 7 terms = 42 terms
    # Plus 6 terms for j=4: total 48 terms? But the code says rank=54.
    
    # Let me recount the 54-rank solution:
    # j_pairs = [(0, 1), (2, 3)]: 2 pairs
    # k_pairs = [(0, 1), (2, 3), (4, 5)]: 3 pairs
    # 2 x 3 = 6 blocks, each with 7 terms = 42 terms
    # Plus j=4 with 6 k values: 6 terms
    # Total: 42 + 6 = 48 terms
    
    # But the code asserts rank=54. Let me check the code again...
    # Ah, I see the issue. The code has a bug or the rank is actually 48.
    
    # Let's verify by counting the col increments in the code.
    
    # For each (j0, j1) in j_pairs and (k0, k1) in k_pairs:
    #   7 terms (P1 through P7)
    # That's 2 * 3 * 7 = 42 terms
    
    # For k in range(p):  # p=6
    #   2 terms each
    # That's 6 * 2 = 12 terms
    
    # Total: 42 + 12 = 54 terms
    
    # I see, the j=4 block has 6 k values, each with 2 terms (for i=0 and i=1), so 12 terms.
    
    # So the 54-rank solution is correct.
    
    # Now, can we do better? Let's try the split I described above:
    # j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}
    
    # For {0,1}x{0,1}: 7 terms
    # For {2,3}x{0,1}: 7 terms
    # For {4}x{0,1}: 2 terms (schoolbook for 1x2)
    # For {0,1}x{2,3}: 7 terms
    # For {2,3}x{2,3}: 7 terms
    # For {4}x{2,3}: 2 terms
    # For {0,1}x{4,5}: 7 terms
    # For {2,3}x{4,5}: 7 terms
    # For {4}x{4,5}: 2 terms
    
    # Total: 7+7+2+7+7+2+7+7+2 = 48 terms
    
    # This is better than 54! Let's implement this.
    
    def _build_48_rank(n, m, p):
        """
        Build 2x5x6 decomposition using block Strassen. Rank 48.
        Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
        """
        rank = 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        # Strassen 2x2x2 block
        def add_strassen_block(j0, j1, k0, k1):
            nonlocal col
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1)B_j0,k0
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3 = A_0,j0(B_j0,k1 - B_j1,k1)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4 = A_1,j1(B_j1,k0 - B_j0,k0)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1)B_j1,k1
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
        
        # Schoolbook 1x2 block
        def add_schoolbook_1x2(j, k0, k1):
            nonlocal col
            # C_0,k0 = A_0,j * B_j,k0
            U[0*m + j, col] += 1
            V[j*p + k0, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
            # C_1,k0 = A_1,j * B_j,k0
            U[1*m + j, col] += 1
            V[j*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # C_0,k1 = A_0,j * B_j,k1
            U[0*m + j, col] += 1
            V[j*p + k1, col] += 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # C_1,k1 = A_1,j * B_j,k1
            U[1*m + j, col] += 1
            V[j*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
        
        # Apply the blocks
        # j groups: {0,1}, {2,3}, {4}
        # k groups: {0,1}, {2,3}, {4,5}
        
        j_groups = [(0, 1), (2, 3), (4,)]
        k_groups = [(0, 1), (2, 3), (4, 5)]
        
        for jg in j_groups:
            for kg in k_groups:
                if len(jg) == 2 and len(kg) == 2:
                    add_strassen_block(jg[0], jg[1], kg[0], kg[1])
                elif len(jg) == 1 and len(kg) == 2:
                    add_schoolbook_1x2(jg[0], kg[0], kg[1])
                else:
                    # Should not happen
                    pass
        
        assert col == rank, f"Expected col={rank}, got {col}"
        return U, V, W
    
    # Try the 48-rank solution
    U48, V48, W48 = _build_48_rank(n, m, p)
    
    if _verify_decomposition(U48, V48, W48, n, m, p):
        return U48, V48, W48
    
    # If 48 doesn't work, try other approaches
    # For now, return the input as is
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
