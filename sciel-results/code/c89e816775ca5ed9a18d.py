import numpy as np
import time

DIMENSIONS = [2, 3, 10]
_TIME_LIMIT = 140  # safety margin before 150s kill

def _build_rank54():
    """
    Build a rank-54 decomposition for 2x3 * 3x10 matrix multiplication.
    
    Strategy: Use Strassen for the 2x2 blocks of the first 4 columns of B (indices 0-3),
    and schoolbook for the remaining 6 columns (indices 4-9).
    
    Split B into B1 (3x4) and B2 (3x6).
    For B1, split into two 2x2 blocks and use Strassen (7 mults per block = 14 total).
    For B2, use schoolbook (2*6 = 12 mults).
    Total: 14 + 12 = 26... wait that's too low.
    
    Actually, let me reconsider. The standard approach:
    - Split the 3 intermediate dimension into groups.
    - For a 2x3 * 3x10, we can view it as:
      A is 2x3, B is 3x10.
    
    Let me think about this differently. The known best rank for 2x3 * 3x10:
    - Schoolbook: 2*3*10 = 60
    - Strassen-based: 55 (as in the current program)
    - Can we do better?
    
    Actually, let's try a different splitting strategy.
    Split the m=3 dimension into m1=1 and m2=2.
    A1 = A[:, 0] (2x1), A2 = A[:, 1:] (2x2)
    B1 = B[0, :] (1x10), B2 = B[1:, :] (2x10)
    
    C = A1*B1 + A2*B2
    
    A1*B1: 2x1 * 1x10 = 20 mults (schoolbook)
    A2*B2: 2x2 * 2x10 = 2*2*10 = 40 mults (schoolbook), or use Strassen-like
    
    For 2x2 * 2x10, we can split the 10 into 5 blocks of 2:
    5 * (2x2 * 2x2) = 5 * 7 = 35 with Strassen
    
    Total: 20 + 35 = 55. Same as before.
    
    Alternatively, split m=3 into m1=2, m2=1:
    A1 = A[:, :2] (2x2), A2 = A[:, 2] (2x1)
    B1 = B[:2, :] (2x10), B2 = B[2, :] (1x10)
    
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x10, split 10 into 5 blocks of 2: 5*7 = 35
    A2*B2: 2x1 * 1x10 = 20
    
    Total: 35 + 20 = 55. Same.
    
    So the rank-55 is from this natural decomposition. To get rank-54,
    we need a more clever approach.
    
    Let me try: can we share some multiplications between the blocks?
    Or use a different algorithm for one of the sub-problems?
    
    For 2x2 * 2x2, the minimum rank is 7 (Strassen).
    For 2x1 * 1x2, the rank is 4 (it's just outer product, 2*1*2=4, and that's minimal).
    
    Wait, 2x1 * 1x10: this is like multiplying a 2-vector by a 1x10 matrix.
    The result is 2x10. Each output C[i,k] = A[i,0]*B[0,k].
    This is 2*10 = 20 rank-1 terms, and that's minimal (tensor rank = 20).
    
    So 55 seems to be the natural bound. Let me check if there's a way to do better.
    
    Actually, I recall that for small matrix multiplication, the tensor rank can sometimes
    be lower than the product of dimensions due to algebraic structure.
    
    Let me try a direct approach: use the fact that we have a 2x3 * 3x10 tensor.
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    
    We want to find the minimum rank r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    
    Known lower bounds:
    - Border rank of 2x3 * 3x10...
    
    Let me just try to construct a rank-54 decomposition by modifying the rank-55 one.
    """
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Use Strassen for the first 4 columns of B (k=0,1,2,3), split into two 2x2 blocks
    # Block 1: k=0,1
    # Block 2: k=2,3
    
    for g in range(2):
        k0 = 2 * g
        k1 = k0 + 1
        
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # 7 Strassen terms
        # Term 1: m1 = (a00+a11)*(b00+b11)
        U[u_idx_a00][col] = 1; U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1; V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1; W[w_idx_c11][col] = 1
        col += 1
        
        # Term 2: m2 = (a10+a11)*b00
        U[u_idx_a10][col] = 1; U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        W[w_idx_c10][col] = 1; W[w_idx_c11][col] = -1
        col += 1
        
        # Term 3: m3 = a00*(b01-b11)
        U[u_idx_a00][col] = 1
        V[v_idx_b01][col] = 1; V[v_idx_b11][col] = -1
        W[w_idx_c01][col] = 1; W[w_idx_c11][col] = 1
        col += 1
        
        # Term 4: m4 = a11*(b10-b00)
        U[u_idx_a11][col] = 1
        V[v_idx_b10][col] = 1; V[v_idx_b00][col] = -1
        W[w_idx_c00][col] = 1; W[w_idx_c10][col] = 1
        col += 1
        
        # Term 5: m5 = (a00+a01)*b11
        U[u_idx_a00][col] = 1; U[u_idx_a01][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c01][col] = 1; W[w_idx_c00][col] = -1
        col += 1
        
        # Term 6: m6 = (a10-a00)*(b00+b01)
        U[u_idx_a10][col] = 1; U[u_idx_a00][col] = -1
        V[v_idx_b00][col] = 1; V[v_idx_b01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 7: m7 = (a01-a11)*(b10+b11)
        U[u_idx_a01][col] = 1; U[u_idx_a11][col] = -1
        V[v_idx_b10][col] = 1; V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        col += 1
    
    # That's 14 terms for the first 4 columns.
    # Now for columns k=4 to k=9 (6 columns), use schoolbook for all 3 rows of A.
    # That's 2*3*6 = 36 terms. Total would be 14+36 = 50.
    # But wait, that's not right. Let me recount.
    
    # Actually, for k=4..9, we need to compute C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1 and k=4..9.
    # That's 2 * 3 * 6 = 36 rank-1 terms.
    # Total: 14 + 36 = 50. That's better than 55!
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

def _build_rank55():
    """Build the known rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for g in range(5):
        k0 = 2 * g
        k1 = k0 + 1
        
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # Term 1: m1 = (a00+a11)*(b00+b11)
        U[u_idx_a00][col] = 1; U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1; V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1; W[w_idx_c11][col] = 1
        col += 1
        
        # Term 2: m2 = (a10+a11)*b00
        U[u_idx_a10][col] = 1; U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        W[w_idx_c10][col] = 1; W[w_idx_c11][col] = -1
        col += 1
        
        # Term 3: m3 = a00*(b01-b11)
        U[u_idx_a00][col] = 1
        V[v_idx_b01][col] = 1; V[v_idx_b11][col] = -1
        W[w_idx_c01][col] = 1; W[w_idx_c11][col] = 1
        col += 1
        
        # Term 4: m4 = a11*(b10-b00)
        U[u_idx_a11][col] = 1
        V[v_idx_b10][col] = 1; V[v_idx_b00][col] = -1
        W[w_idx_c00][col] = 1; W[w_idx_c10][col] = 1
        col += 1
        
        # Term 5: m5 = (a00+a01)*b11
        U[u_idx_a00][col] = 1; U[u_idx_a01][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c01][col] = 1; W[w_idx_c00][col] = -1
        col += 1
        
        # Term 6: m6 = (a10-a00)*(b00+b01)
        U[u_idx_a10][col] = 1; U[u_idx_a00][col] = -1
        V[v_idx_b00][col] = 1; V[v_idx_b01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 7: m7 = (a01-a11)*(b10+b11)
        U[u_idx_a01][col] = 1; U[u_idx_a11][col] = -1
        V[v_idx_b10][col] = 1; V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        col += 1
    
    # Part 2: Schoolbook for A2 * B2
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # For each output entry C[i,k], check that sum_r U[i*m+j]*V[j*p+k]*W[k*n+i] == delta
    # Actually, the tensor entry T[i,j,k] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But wait - the convention is W[k*n+i, r], so W is indexed by (k, i)
    # The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_find_rank54(U, V, W, n, m, p, deadline):
    """
    Try to find a rank-54 solution by looking for a linear dependency
    among the 55 rank-1 tensors that allows dropping one.
    
    Strategy: For each term r, check if removing it and adjusting others
    can still cover all tensor entries. This is equivalent to checking
    if the r-th column is in the span of the others in a way that
    preserves the tensor decomposition.
    
    More concretely: try to express one term as a linear combination
    of the others such that the resulting 54-term decomposition is exact.
    """
    rank = 55
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # Each term r contributes: U[:,r] outer V[:,r] outer W[:,r]
    # We need sum_r U[:,r]⊗V[:,r]⊗W[:,r] = T (all-ones tensor)
    
    # Try dropping each term and see if we can compensate
    # For a given term to drop, we need the remaining 54 terms to still sum to T.
    # This means the dropped term must be a linear combination of the others
    # in the tensor space.
    
    # Alternative: try to find a 54-term decomposition directly by
    # checking if any term can be absorbed.
    
    # For each pair of Strassen terms within the same block, check if
    # their combined contribution can be represented as a single rank-1 tensor.
    
    # Actually, let's try a more direct approach:
    # For each term r (0..54), try to see if we can modify the remaining
    # terms to compensate. This is a linear system problem.
    
    # The constraint is: for all (i,j,k), sum_{r≠r0} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 - U[i,j,r0]*V[j,k,r0]*W[k,i,r0]
    # This is a nonlinear system in general, but if we fix the structure...
    
    # Let's try a simpler heuristic: for each term, check if it's "redundant"
    # in the sense that its contribution is already covered by others.
    
    for r0 in range(rank):
        if time.time() > deadline:
            break
        # Check if term r0 can be dropped
        # Build the "deficit" tensor: D[i,j,k] = 1 - U[i,j,r0]*V[j,k,r0]*W[k,i,r0]
        # Check if D can be written as sum of 54 rank-1 tensors from the remaining set
        # (with possibly modified coefficients)
        
        # This is complex. Instead, let's try: can we find a linear combination
        # of the other 54 terms that equals term r0?
        
        # In the tensor space of dimension 6*30*20 = 3600, each term is a vector.
        # Check if term r0 is in the span of the others.
        
        # Flatten each term into a vector of length n*m*p
        def flatten_term(u_col, v_col, w_col):
            vec = np.zeros(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[i*m*p + j*p + k] = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
            return vec
        
        # This is O(55 * 600) which is fine
        all_terms = []
        for r in range(rank):
            all_terms.append(flatten_term(U_np[:, r], V_np[:, r], W_np[:, r]))
        
        term_r0 = all_terms[r0]
        
        # Check if term_r0 is in the span of the other 54 terms
        # Use numpy least squares
        others = np.array([all_terms[r] for r in range(rank) if r != r0])  # 54 x 600
        
        # Solve: others.T @ x = term_r0 (least squares)
        # If the residual is ~0, then term_r0 is in the span
        try:
            x, residuals, rank_overs, sv = np.linalg.lstsq(others.T, term_r0, rcond=None)
            residual = np.linalg.norm(others.T @ x - term_r0)
            if residual < 1e-10:
                # Term r0 is a linear combination of others!
                # We can drop it and get a rank-54 solution
                new_U = [U[r] for r in range(rank) if r != r0]
                new_V = [V[r] for r in range(rank) if r != r0]
                new_W = [W[r] for r in range(rank) if r != r0]
                # Verify
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W
        except:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Try the rank-54 construction first
    U54, V54, W54 = _build_rank54()
    if _verify_decomposition(U54, V54, W54, n, m, p):
        return {"U": U54, "V": V54, "W": W54}
    
    # Build the rank-55 baseline
    U, V, W = _build_rank55()
    
    # Verify the baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but handle gracefully
        pass
    
    # Try to find a rank-54 solution
    result = _try_find_rank54(U, V, W, n, m, p, deadline)
    
    if result is not None:
        new_U, new_V, new_W = result
        return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fall back to rank 55
    return {"U": U, "V": V, "W": W}
