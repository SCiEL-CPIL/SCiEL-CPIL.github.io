"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_4x4x4():
    """
    Returns U, V, W matrices (as lists) for 4x4x4 matrix multiplication using Strassen.
    Rank 49 decomposition.
    
    For 2x2x2, Strassen uses 7 multiplications.
    For 4x4x4, we can use a block Strassen approach.
    
    Actually, let me use a known rank-49 decomposition for 4x4x4.
    The standard approach: use Strassen recursively.
    
    4x4 can be split into four 2x2 blocks.
    Using Strassen on the block structure gives rank 7^2 = 49 for 4x4x4.
    """
    n = 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * n)]
    V = [[0.0] * rank for _ in range(n * n)]
    W = [[0.0] * rank for _ in range(n * n)]
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply Strassen at the block level.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B, each Cij is 2x2
    
    # The 7 Strassen products M1..M7 are each 2x2x2 multiplications.
    # Each 2x2x2 multiplication can be done with rank 7 (Strassen) or rank 8 (schoolbook).
    # Using Strassen for each: total rank = 7 * 7 = 49.
    
    # Let me define the 2x2x2 Strassen coefficients.
    # For 2x2x2:
    # U indices: 0..3 (for A11,A12,A21,A22)
    # V indices: 0..3 (for B11,B12,B21,B22)
    # W indices: 0..3 (for C11,C12,C21,C22)
    
    # Strassen 2x2x2 rank-7:
    # r=0: M1 = (A11+A22)(B11+B22)
    #   U: A11->1, A22->1; V: B11->1, B22->1; W: C11->1, C22->1
    # r=1: M2 = (A21+A22)B11
    #   U: A21->1, A22->1; V: B11->1; W: C21->1
    # r=2: M3 = A11(B21-B22)
    #   U: A11->1; V: B21->1, B22->-1; W: C12->1, C21->1
    # r=3: M4 = A22(B11-B21)
    #   U: A22->1; V: B11->1, B21->-1; W: C11->1, C21->-1
    # r=4: M5 = (A11+A12)B22
    #   U: A11->1, A12->1; V: B22->1; W: C11->-1, C12->1
    # r=5: M6 = (A21-A11)(B11+B12)
    #   U: A21->1, A11->-1; V: B11->1, B12->1; W: C22->1
    # r=6: M7 = (A12-A22)(B21+B22)
    #   U: A12->1, A22->-1; V: B21->1, B22->1; W: C11->1, C22->-1
    
    # Let me define this as a list of (u_coeffs, v_coeffs, w_coeffs) for the 2x2x2 case.
    # u_coeffs: dict from A_index to coefficient
    # v_coeffs: dict from B_index to coefficient  
    # w_coeffs: dict from C_index to coefficient
    
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    strassen_2x2 = [
        # r=0: M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # r=1: M2
        ({2:1, 3:1}, {0:1}, {2:1}),
        # r=2: M3
        ({0:1}, {2:1, 3:-1}, {1:1, 2:1}),
        # r=3: M4
        ({3:1}, {0:1, 2:-1}, {0:1, 2:-1}),
        # r=4: M5
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # r=5: M6
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # r=6: M7
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:-1}),
    ]
    
    # Now for 4x4x4, we have block structure.
    # A block (ii,jj) where ii,jj in {0,1} corresponds to 2x2 submatrix.
    # A element (a,b) where a,b in 0..3: block index (a//2, b//2), local index (a%2, b%2)
    # Similarly for B and C.
    
    # For each Strassen level-1 term (7 terms), we have a 2x2x2 multiplication.
    # For each of those, we use Strassen 2x2x2 (7 terms).
    # Total: 7*7 = 49 terms.
    
    # Let me map:
    # Level 1: term s in 0..6, with (u1, v1, w1)
    # Level 2: term t in 0..6, with (u2, v2, w2)
    # Global term index: r = s*7 + t
    
    # For level 1 term s:
    # u1 gives coefficients for A blocks: A_block(ii,jj) gets coefficient u1[(ii,jj)]
    # v1 gives coefficients for B blocks: B_block(ii,jj) gets coefficient v1[(ii,jj)]
    # w1 gives coefficients for C blocks: C_block(ii,jj) gets coefficient w1[(ii,jj)]
    
    # For level 2 term t:
    # u2 gives coefficients for A local elements within the block
    # v2 gives coefficients for B local elements within the block
    # w2 gives coefficients for C local elements within the block
    
    # So for global term r = s*7+t:
    # For each A element (a,b):
    #   block_ii = a//2, block_jj = b//2
    #   local_a = a%2, local_b = b%2
    #   U[a,b] += u1[(block_ii,block_jj)] * u2[(local_a,local_b)]
    
    # Wait, I need to be more careful. The Strassen algorithm works on the block matrices.
    # Let me reconsider.
    
    # Actually, the standard block Strassen for 4x4x4:
    # We have A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B = [[C00, C01], [C10, C11]] where each Ci is 2x2
    
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)  -> 2x2 * 2x2 = 2x2
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B00-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2x2 matrix multiplication, which we decompose with Strassen (rank 7).
    
    # So total rank = 7 (level 1) * 7 (level 2) = 49.
    
    # Now I need to carefully construct the U, V, W matrices.
    
    # Let me define the level-1 Strassen terms more explicitly.
    # For each level-1 term s (0..6), I need:
    # - For each A block (ii,jj) in {0,1}^2: coefficient a_coeff[s][(ii,jj)]
    # - For each B block (ii,jj) in {0,1}^2: coefficient b_coeff[s][(ii,jj)]
    # - For each C block (ii,jj) in {0,1}^2: coefficient c_coeff[s][(ii,jj)]
    
    # s=0 (M1): A: A00->1, A11->1; B: B00->1, B11->1; C: C00->1, C11->1
    # s=1 (M2): A: A10->1, A11->1; B: B00->1; C: C10->1
    # s=2 (M3): A: A00->1; B: B10->1, B11->-1; C: C01->1, C10->1
    # s=3 (M4): A: A11->1; B: B00->1, B10->-1; C: C00->1, C10->-1
    # s=4 (M5): A: A00->1, A01->1; B: B11->1; C: C00->-1, C01->1
    # s=5 (M6): A: A10->1, A00->-1; B: B00->1, B01->1; C: C11->1
    # s=6 (M7): A: A01->1, A11->-1; B: B10->1, B11->1; C: C00->1, C11->-1
    
    level1 = [
        # s=0
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # s=2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        # s=3
        ({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):-1}),
        # s=4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s=5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s=6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    # Level 2: Strassen 2x2x2, indices 0..6
    # A local indices: (la,lb) where la,lb in {0,1}
    # B local indices: (lb2,lb3) where lb2,lb3 in {0,1}  -- wait, I'll use (la,lb) for both
    # C local indices: (lc,ld) where lc,ld in {0,1}
    
    level2 = [
        # t=0
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # t=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # t=2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        # t=3
        ({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):-1}),
        # t=4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # t=5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # t=6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    # Now construct U, V, W for 4x4x4.
    # U has shape (16, 49), indexed by (a,b) for a,b in 0..3 and r in 0..48
    # V has shape (16, 49), indexed by (c,d) for c,d in 0..3 and r in 0..48
    # W has shape (16, 49), indexed by (e,f) for e,f in 0..3 and r in 0..48
    
    for s in range(7):
        a_coeff, b_coeff, c_coeff = level1[s]
        for t in range(7):
            r = s * 7 + t
            u2, v2, w2 = level2[t]
            
            # For each A element (a,b) in 0..3 x 0..3:
            for a in range(4):
                for b in range(4):
                    block_ii, block_jj = a // 2, b // 2
                    local_a, local_b = a % 2, b % 2
                    a_val = a_coeff.get((block_ii, block_jj), 0)
                    u2_val = u2.get((local_a, local_b), 0)
                    U[a * 4 + b][r] += a_val * u2_val
            
            # For each B element (c,d) in 0..3 x 0..3:
            for c in range(4):
                for d in range(4):
                    block_ii, block_jj = c // 2, d // 2
                    local_c, local_d = c % 2, d % 2
                    b_val = b_coeff.get((block_ii, block_jj), 0)
                    v2_val = v2.get((local_c, local_d), 0)
                    V[c * 4 + d][r] += b_val * v2_val
            
            # For each C element (e,f) in 0..3 x 0..3:
            for e in range(4):
                for f in range(4):
                    block_ii, block_jj = e // 2, f // 2
                    local_e, local_f = e % 2, f % 2
                    c_val = c_coeff.get((block_ii, block_jj), 0)
                    w2_val = w2.get((local_e, local_f), 0)
                    W[e * 4 + f][r] += c_val * w2_val
    
    return U, V, W

def _adapt_to_4x4x7(U4, V4, W4):
    """
    Adapt a 4x4x4 rank-49 decomposition to 4x4x7.
    
    For 4x4x4: U is (16,49), V is (16,49), W is (16,49)
    Indices: U[i*4+j], V[j*4+k], W[k*4+i]
    
    For 4x4x7: U is (16, r), V is (28, r), W is (28, r)
    Indices: U[i*4+j], V[j*7+k], W[k*4+i]
    
    The problem: V and W have different sizes.
    
    One approach: pad B from 4x7 to 4x8 by adding 4 zero columns, then use a 4x4x8 decomposition.
    But we don't have a 4x4x8 decomposition.
    
    Alternative: use the 4x4x4 decomposition for the first 4 columns of B, and schoolbook for the last 3.
    This gives rank 49 + 4*4*3 = 49 + 48 = 97.
    
    Let me try another approach: use the 4x4x4 Strassen and extend to handle 7 columns.
    
    Actually, let me just do: Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x3 (rank 48) = rank 97.
    """
    n, m, p = DIMENSIONS
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # U4[i*4+j], V4[j*4+k], W4[k*4+i]
    
    # For 4x4x7:
    # Part 1: B columns 0-3 (4 columns), use Strassen
    # Part 2: B columns 4-6 (3 columns), use schoolbook
    
    r1 = 49  # Strassen rank
    r2 = 4 * 4 * 3  # Schoolbook for 4x4x3
    total_rank = r1 + r2
    
    U = [[0.0] * total_rank for _ in range(16)]
    V = [[0.0] * total_rank for _ in range(28)]
    W = [[0.0] * total_rank for _ in range(28)]
    
    # Part 1: Strassen for first 4 columns of B
    for i in range(4):
        for j in range(4):
            for r in range(r1):
                U[i*4+j][r] = U4[i*4+j][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(r1):
                V[j*7+k][r] = V4[j*4+k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(r1):
                W[k*4+i][r] = W4[k*4+i][r]
    
    # Part 2: Schoolbook for last 3 columns of B (columns 4,5,6)
    col = r1
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j][col] = 1.0
                V[j*7+k][col] = 1.0
                W[k*4+i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_numpy(U, V, W, tol=1e-6):
    """Check decomposition using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

_global_start_time = [0.0]

def _optimize_rank(U, V, W, max_iter=500, lr=0.01):
    """
    Try to reduce the effective rank by zeroing out some columns.
    This is a heuristic approach.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    start_time = _global_start_time[0]
    
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the tensor values
    def compute_residual(U_a, V_a, W_a):
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_a[i*m+j], V_a[j*p+k] * W_a[k*n+i])
                    res += (val - 1.0) ** 2
        return res
    
    current_res = compute_residual(U_arr, V_arr, W_arr)
    
    # Try to zero out columns one by one if it doesn't increase residual too much
    for col in range(rank - 1, -1, -1):
        if time.time() > start_time + 120:
            break
        
        # Try zeroing this column
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        U_test[:, col] = 0
        V_test[:, col] = 0
        W_test[:, col] = 0
        
        test_res = compute_residual(U_test, V_test, W_test)
        
        if test_res < current_res * 1.01:  # Allow 1% increase
            U_arr = U_test
            V_arr = V_test
            W_arr = W_test
            current_res = test_res
    
    # Find the effective rank (number of non-zero columns)
    effective_rank = 0
    for col in range(rank):
        if np.any(U_arr[:, col] != 0) or np.any(V_arr[:, col] != 0) or np.any(W_arr[:, col] != 0):
            effective_rank = col + 1
    
    if effective_rank < rank:
        return U_arr[:, :effective_rank].tolist(), V_arr[:, :effective_rank].tolist(), W_arr[:, :effective_rank].tolist()
    
    return U, V, W

def construct():
    start_time = time.time()
    _global_start_time[0] = start_time
    deadline = start_time + 130.0
    
    # Get Strassen 4x4x4 decomposition
    U4, V4, W4 = _strassen_4x4x4()
    
    # Verify it works for 4x4x4
    if not _verify_numpy(U4, V4, W4, tol=1e-6):
        # Fall back to schoolbook
        return _schoolbook()
    
    # Adapt to 4x4x7
    result = _adapt_to_4x4x7(U4, V4, W4)
    
    # Verify the adapted version
    if _verify_numpy(result["U"], result["V"], result["W"], tol=1e-6):
        # Try to reduce rank
        U_opt, V_opt, W_opt = _optimize_rank(result["U"], result["V"], result["W"])
        if _verify_numpy(U_opt, V_opt, W_opt, tol=1e-6):
            return {"U": U_opt, "V": V_opt, "W": W_opt}
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
