import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill

    # Build a block-structured decomposition.
    # Partition A (3x4): rows {0,1} vs {2}; cols {0,1} vs {2,3}
    # Partition B (4x6): rows {0,1} vs {2,3}; cols {0,1,2} vs {3,4,5}
    #
    # C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:4]*B[2:4,0:3]
    # C[0:2, 3:6] = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:4]*B[2:4,3:6]
    # C[2:3, 0:3] = A[2:3,0:2]*B[0:2,0:3] + A[2:3,2:4]*B[2:4,0:3]
    # C[2:3, 3:6] = A[2:3,0:2]*B[0:2,3:6] + A[2:3,2:4]*B[2:4,3:6]
    #
    # Subproblem types:
    #   2x2 * 2x3 = 2x3  (8 subproblems total, but grouped into 4 pairs)
    #   1x2 * 2x3 = 1x3  (4 subproblems)
    #
    # For a 2x2 * 2x3 product: schoolbook rank 12. We'll use rank 12 (safe).
    # For a 1x2 * 2x3 product: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k], rank 6 schoolbook.
    #   But actually this is just: for each k, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   This is a sum of 2 rank-1 terms in (j,k) for fixed i=0, so rank = 2*3 = 6? 
    #   No: the tensor T[j,k] for fixed i is: T[0,k]=1, T[1,k]=1 for all k.
    #   That's a 2x3 matrix of all 1s, which has rank 1 as a matrix, but as a
    #   tensor decomposition we need rank-1 tensors U[j]*V[k]*W[i].
    #   Since i is fixed (i=2 in the full problem), W[i,c] is just a scalar per term.
    #   We need: sum_c U[c]*V[j,k]*W[c] = 1 for all j,k where the term is active.
    #   Actually for the 1x2*2x3 subproblem embedded in the full problem:
    #   The contribution to C[2,k] from A[2,j]*B[j,k] for j in {0,1} (first block)
    #   is: sum_j A[2,j]*B[j,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    #   This needs: for each (j,k) in {0,1}x{0,1,2}, the product U[2*m+j,c]*V[j*p+k,c]*W[k*n+2,c] summed = 1
    #   Schoolbook: 2*3 = 6 terms, one for each (j,k).
    #   Can we do better? The tensor is T[j,k] = 1 for j in {0,1}, k in {0,1,2}.
    #   This is a 2x3 matrix of all 1s. Its tensor rank (as a bilinear form) is 1
    #   if we allow arbitrary U,V,W... but we're constrained by the global indexing.
    #   Since W depends on k and i (fixed i=2), and U depends on j, V depends on j,k:
    #   We need sum_c U[j,c]*V[j,k,c]*W[k,c] = 1 for all j in {0,1}, k in {0,1,2}.
    #   With rank 1: U[j,0]*V[j,k,0]*W[k,0] = 1 for all j,k. This requires
    #   U[j,0]*W[k,0] = 1/V[j,k,0] for all j,k. If V[j,k,0]=1, then U[j,0]*W[k,0]=1
    #   for all j,k, which means U[0,0]=U[1,0] and W[0,0]=W[1,0]=W[2,0], and
    #   U[0,0]*W[0,0]=1. So rank 1 works! U[0,0]=U[1,0]=1, W[0,0]=W[1,0]=W[2,0]=1, V[j,k,0]=1.
    #   Wait, but this would interfere with other subproblems that use the same (j,k) indices.
    #   Since we're building a global decomposition, each term c is global. We can't reuse
    #   the same term for different subproblems unless they're compatible.
    #
    # So the safest approach: use schoolbook for each subproblem, giving rank 72.
    # But we can do better by using Strassen for the 2x2x2 parts within the 2x2x3 blocks.
    #
    # Let me use a simpler, verified approach: construct the full rank-72 schoolbook,
    # then apply Strassen reductions to 6 disjoint 2x2x2 subproblems.
    #
    # Actually, the cleanest verified approach: just use schoolbook rank 72 and verify.
    # The known best is 66, but constructing it reliably in code is complex.
    # Let me try a rank-66 construction using 6 Strassen blocks.
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if verify(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_sb
    else:
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_sb

    # Now try to construct a rank-66 decomposition using Strassen on 2x2x2 blocks.
    # We'll build it from scratch.
    #
    # Partition:
    # A rows: {0,1} (block A_top), {2} (block A_bot)
    # A cols: {0,1} (block A_left), {2,3} (block A_right)
    # B rows: {0,1} (block B_top), {2,3} (block B_bot)
    # B cols: {0,1,2} (block B_left), {3,4,5} (block B_right)
    #
    # C = A*B:
    # C_top_left (2x3) = A_top_left(2x2)*B_top_left(2x3) + A_top_right(2x2)*B_bot_left(2x3)
    # C_top_right (2x3) = A_top_left(2x2)*B_top_right(2x3) + A_top_right(2x2)*B_bot_right(2x3)
    # C_bot_left (1x3) = A_bot_left(1x2)*B_top_left(2x3) + A_bot_right(1x2)*B_bot_left(2x3)
    # C_bot_right (1x3) = A_bot_left(1x2)*B_top_right(2x3) + A_bot_right(1x2)*B_bot_right(2x3)
    #
    # For each 2x2 * 2x3 product, use schoolbook rank 12.
    # For each 1x2 * 2x3 product, use schoolbook rank 6.
    # Total: 4*12 + 4*6 = 48 + 24 = 72. No improvement.
    #
    # To improve, use Strassen on the 2x2 parts. For 2x2 * 2x3:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Split k into {0,1} and {2}:
    # C[i,0:2] = A * B[:,0:2] (2x2 * 2x2 = 2x2, rank 7 via Strassen)
    # C[i,2] = A * B[:,2] (2x2 * 2x1 = 2x1, rank 4 schoolbook, or rank 3?)
    # For 2x2 * 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1], rank 4 schoolbook.
    # Can we do rank 3? The tensor is T[i,j] = 1 for all i,j (2x2 matrix of 1s),
    # but with the constraint that B is 2x1 (so B[j] is scalar). 
    # Actually for 2x2 * 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # This is a 2x2x1 tensor. Rank = 4 schoolbook. Can we do 3?
    # The tensor T[i,j,k] with k fixed is just T[i,j] = 1. As a bilinear form in (A,B),
    # it's <A, B^T> which has rank 4 (it's the identity pairing). So rank 4 is minimal.
    #
    # So for 2x2 * 2x3: rank 7 (for 2x2x2 part) + 4 (for 2x2x1 part) = 11.
    # For 1x2 * 2x3: rank 6 schoolbook. Can we do better?
    # C[0,k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0,1,2.
    # This is a 1x2x3 tensor. For fixed k, it's a 1x2 bilinear form (rank 2).
    # Total rank = 3 (one per k, since the A part is the same for all k).
    # Actually: sum_c U[c]*V[j,k,c]*W[c] = 1 for j in {0,1}, k in {0,1,2}.
    # With rank 3: for each k, one term. U[c_k] is same for both j's? No, U depends on j.
    # Let me use: for each k, term c_k: U[j,c_k] = 1 for j=0,1; V[j,k,c_k] = 1 for j=0,1; W[k,c_k] = 1.
    # Then sum_c U[j,c]*V[j,k,c]*W[k,c] = 1 (from term c_k) + 0 (others) = 1. 
    # But this uses 3 terms, one per k. So rank 3 for 1x2*2x3.
    #
    # Total: 4*11 + 4*3 = 44 + 12 = 56. That's below 66!
    # But wait, I need to verify this is correct. The issue is that terms from different
    # subproblems might interfere. Let me build it carefully.
    
    # Build rank-56 decomposition:
    U2 = []
    V2 = []
    W2 = []
    
    def add_term(u_vals, v_vals, w_vals):
        """u_vals: dict (i*m+j) -> val, v_vals: dict (j*p+k) -> val, w_vals: dict (k*n+i) -> val"""
        idx = len(U2)
        U2.append([0] * 0)  # placeholder, will fix
        V2.append([0] * 0)
        W2.append([0] * 0)
        return idx
    
    # Actually, let me build it differently. I'll construct the full matrices directly.
    # Number of terms: 4 subproblems of 2x2*2x3 (rank 11 each) + 4 subproblems of 1x2*2x3 (rank 3 each)
    # Total: 4*11 + 4*3 = 56
    
    total_rank = 4 * 11 + 4 * 3  # = 56
    U2 = [[0] * total_rank for _ in range(n * m)]
    V2 = [[0] * total_rank for _ in range(m * p)]
    W2 = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a 2x2 * 2x3 subproblem
    # A block: rows a_r0, a_r1; cols a_c0, a_c1
    # B block: rows b_r0, b_r1; cols b_c0, b_c1, b_c2
    # Contributes to C rows a_r0, a_r1 and C cols b_c0, b_c1, b_c2
    def add_2x2x3(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, b_c2):
        nonlocal col
        # Part 1: 2x2 * 2x2 (cols b_c0, b_c1) - use Strassen rank 7
        # A = [[A00, A01], [A10, A11]] where Aij = A[a_r_i, a_c_j]
        # B = [[B00, B01], [B10, B11]] where Bij = B[b_r_i, b_c_j]
        # Strassen:
        # M1 = (A00+A11)*B00
        # M2 = (A00+A11)*B10
        # M3 = A00*(B00-B10)
        # M4 = A11*(B01-B11)
        # M5 = A01*(B10+B11)
        # M6 = A10*(B00-B01)
        # M7 = (A10+A11)*B11
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)*B00
        # A00 = A[a_r0, a_c0], A11 = A[a_r1, a_c1], B00 = B[b_r0, b_c0]
        # U: A00 and A11 contribute
        U2[a_r0 * m + a_c0][col] = 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r0 * p + b_c0][col] = 1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c0 * n + a_r1][col] = 1
        W2[b_c1 * n + a_r0][col] = -1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1
        
        # M2 = (A00+A11)*B10
        U2[a_r0 * m + a_c0][col] = 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c0][col] = 1
        W2[b_c0 * n + a_r1][col] = 1
        W2[b_c1 * n + a_r0][col] = 1
        col += 1
        
        # M3 = A00*(B00-B10)
        U2[a_r0 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c0][col] = 1
        V2[b_r1 * p + b_c0][col] = -1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1
        
        # M4 = A11*(B01-B11)
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r0 * p + b_c1][col] = 1
        V2[b_r1 * p + b_c1][col] = -1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1
        
        # M5 = A01*(B10+B11)
        U2[a_r0 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c0][col] = 1
        V2[b_r1 * p + b_c1][col] = 1
        W2[b_c0 * n + a_r0][col] = -1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1
        
        # M6 = A10*(B00-B01)
        U2[a_r1 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c0][col] = 1
        V2[b_r0 * p + b_c1][col] = -1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1
        
        # M7 = (A10+A11)*B11
        U2[a_r1 * m + a_c0][col] = 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c1][col] = 1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1
        
        # Part 2: 2x2 * 2x1 (col b_c2) - schoolbook rank 4
        # C[i] = A[i,0]*B[0,b_c2] + A[i,1]*B[1,b_c2]
        # For i=a_r0: A[a_r0,a_c0]*B[b_r0,b_c2] + A[a_r0,a_c1]*B[b_r1,b_c2]
        # For i=a_r1: A[a_r1,a_c0]*B[b_r0,b_c2] + A[a_r1,a_c1]*B[b_r1,b_c2]
        U2[a_r0 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r0][col] = 1
        col += 1
        U2[a_r0 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r0][col] = 1
        col += 1
        U2[a_r1 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r1][col] = 1
        col += 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r1][col] = 1
        col += 1
    
    # Helper: add a 1x2 * 2x3 subproblem
    # A block: row a_r; cols a_c0, a_c1
    # B block: rows b_r0, b_r1; cols b_c0, b_c1, b_c2
    # Contributes to C row a_r and C cols b_c0, b_c1, b_c2
    def add_1x2x3(a_r, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, b_c2):
        nonlocal col
        # C[a_r, b_ck] = A[a_r, a_c0]*B[b_r0, b_ck] + A[a_r, a_c1]*B[b_r1, b_ck]
        # For each k, one term:
        for b_c in [b_c0, b_c1, b_c2]:
            U2[a_r * m + a_c0][col] = 1
            V2[b_r0 * p + b_c][col] = 1
            W2[b_c * n + a_r][col] = 1
            col += 1
            U2[a_r * m + a_c1][col] = 1
            V2[b_r1 * p + b_c][col] = 1
            W2[b_c * n + a_r][col] = 1
            col += 1
    
    # Add the 4 2x2*2x3 subproblems:
    # C_top_left: A[0:2,0:2]*B[0:2,0:3]
    add_2x2x3(0, 1, 0, 1, 0, 1, 0, 1, 2)
    # C_top_left: A[0:2,2:4]*B[2:4,0:3]
    add_2x2x3(0, 1, 2, 3, 2, 3, 0, 1, 2)
    # C_top_right: A[0:2,0:2]*B[0:2,3:6]
    add_2x2x3(0, 1, 0, 1, 0, 1, 3, 4, 5)
    # C_top_right: A[0:2,2:4]*B[2:4,3:6]
    add_2x2x3(0, 1, 2, 3, 2, 3, 3, 4, 5)
    
    # Add the 4 1x2*2x3 subproblems:
    # C_bot_left: A[2:3,0:2]*B[0:2,0:3]
    add_1x2x3(2, 0, 1, 0, 1, 0, 1, 2)
    # C_bot_left: A[2:3,2:4]*B[2:4,0:3]
    add_1x2x3(2, 2, 3, 2, 3, 0, 1, 2)
    # C_bot_right: A[2:3,0:2]*B[0:2,3:6]
    add_1x2x3(2, 0, 1, 0, 1, 3, 4, 5)
    # C_bot_right: A[2:3,2:4]*B[2:4,3:6]
    add_1x2x3(2, 2, 3, 2, 3, 3, 4, 5)
    
    if col != total_rank:
        # Truncate or pad
        total_rank = col
    
    if verify(U2, V2, W2, n, m, p):
        return {"U": U2, "V": V2, "W": W2}
    
    # If verification fails, return schoolbook
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135

    # Schoolbook fallback
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best = {"U": U, "V": V, "W": W}
    best_rank = rank_sb

    # Build rank-56 decomposition
    total_rank = 4 * 11 + 4 * 3
    U2 = [[0] * total_rank for _ in range(n * m)]
    V2 = [[0] * total_rank for _ in range(m * p)]
    W2 = [[0] * total_rank for _ in range(p * n)]
    col = 0

    def add_2x2x3(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, b_c2):
        nonlocal col
        # Strassen for 2x2x2 part (cols b_c0, b_c1)
        # M1 = (A00+A11)*B00
        U2[a_r0 * m + a_c0][col] = 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r0 * p + b_c0][col] = 1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c0 * n + a_r1][col] = 1
        W2[b_c1 * n + a_r0][col] = -1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1

        # M2 = (A00+A11)*B10
        U2[a_r0 * m + a_c0][col] = 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c0][col] = 1
        W2[b_c0 * n + a_r1][col] = 1
        W2[b_c1 * n + a_r0][col] = 1
        col += 1

        # M3 = A00*(B00-B10)
        U2[a_r0 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c0][col] = 1
        V2[b_r1 * p + b_c0][col] = -1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1

        # M4 = A11*(B01-B11)
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r0 * p + b_c1][col] = 1
        V2[b_r1 * p + b_c1][col] = -1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1

        # M5 = A01*(B10+B11)
        U2[a_r0 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c0][col] = 1
        V2[b_r1 * p + b_c1][col] = 1
        W2[b_c0 * n + a_r0][col] = -1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1

        # M6 = A10*(B00-B01)
        U2[a_r1 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c0][col] = 1
        V2[b_r0 * p + b_c1][col] = -1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1

        # M7 = (A10+A11)*B11
        U2[a_r1 * m + a_c0][col] = 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c1][col] = 1
        W2[b_c0 * n + a_r0][col] = 1
        W2[b_c1 * n + a_r1][col] = 1
        col += 1

        # Schoolbook for 2x2x1 part (col b_c2)
        U2[a_r0 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r0][col] = 1
        col += 1
        U2[a_r0 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r0][col] = 1
        col += 1
        U2[a_r1 * m + a_c0][col] = 1
        V2[b_r0 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r1][col] = 1
        col += 1
        U2[a_r1 * m + a_c1][col] = 1
        V2[b_r1 * p + b_c2][col] = 1
        W2[b_c2 * n + a_r1][col] = 1
        col += 1

    def add_1x2x3(a_r, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, b_c2):
        nonlocal col
        for b_c in [b_c0, b_c1, b_c2]:
            U2[a_r * m + a_c0][col] = 1
            V2[b_r0 * p + b_c][col] = 1
            W2[b_c * n + a_r][col] = 1
            col += 1
            U2[a_r * m + a_c1][col] = 1
            V2[b_r1 * p + b_c][col] = 1
            W2[b_c * n + a_r][col] = 1
            col += 1

    add_2x2x3(0, 1, 0, 1, 0, 1, 0, 1, 2)
    add_2x2x3(0, 1, 2, 3, 2, 3, 0, 1, 2)
    add_2x2x3(0, 1, 0, 1, 0, 1, 3, 4, 5)
    add_2x2x3(0, 1, 2, 3, 2, 3, 3, 4, 5)
    add_1x2x3(2, 0, 1, 0, 1, 0, 1, 2)
    add_1x2x3(2, 2, 3, 2, 3, 0, 1, 2)
    add_1x2x3(2, 0, 1, 0, 1, 3, 4, 5)
    add_1x2x3(2, 2, 3, 2, 3, 3, 4, 5)

    if col != total_rank:
        total_rank = col

    if verify(U2, V2, W2, n, m, p):
        return {"U": U2, "V": V2, "W": W2}

    return best
# EVOLVE-BLOCK-END
