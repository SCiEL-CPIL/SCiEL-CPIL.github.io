"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if (i,j,k) is a valid index, 0 otherwise
    # Actually, for matrix multiplication A(n,m) * B(m,p) -> C(n,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    
    # Let's verify by checking the bilinear form
    # For each output position (i,k), we need:
    # C[i,k] = sum_j sum_c U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k]
    # = sum_j A[i,j] * B[j,k] * (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])
    
    # Actually, the standard verification:
    # The tensor M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_rectangular(n, m, p):
    """
    Try to find a Strassen-like decomposition for rectangular matrices.
    For 2x4 * 4x8, we can split into blocks.
    A (2x4) = [A11 A12] where A11 is 2x2, A12 is 2x2
    B (4x8) = [B11; B21] where B11 is 2x8, B21 is 2x8
    C = A11*B11 + A12*B21
    
    We can use Strassen on the 2x2 * 2x8 part if we further split.
    Let's try: A11 (2x2) * B11 (2x8) using Strassen with B split as [B11a B11b] each 2x4
    Then A11 * B11 = A11 * B11a + A11 * B11b
    Each A11 * B11x is 2x2 * 2x4.
    
    For 2x2 * 2x4: split B as [B_top; B_bottom] each 1x4
    C_top = A11[0,:] * B_top + A11[1,:] * B_bottom  -- wait, this isn't right.
    
    Actually, let me think differently. The tensor for 2x4 * 4x8 has rank at most 64.
    Known lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    
    Let's try a block-based approach:
    Split A (2x4) into two 2x2 blocks: A = [A1, A2]
    Split B (4x8) into two 2x8 blocks: B = [B1; B2]
    C = A1*B1 + A2*B2
    
    Now A1*B1 is 2x2 * 2x8. Split B1 into four 2x2 blocks: B1 = [B11 B12 B13 B14]
    A1*B1 = A1*B11 + A1*B12 + A1*B13 + A1*B14
    Each A1*B1j is 2x2 * 2x2, which can be done with rank 7 (Strassen).
    So A1*B1 needs 4*7 = 28 multiplications.
    Similarly A2*B2 needs 28.
    Total: 56.
    
    But we can do better by sharing. Let's use the fact that we have 4 separate 2x2*2x2 multiplications.
    Actually, we can group them.
    
    Alternative: Use the Winograd-like approach or just try rank 56 first.
    """
    n, m, p = 2, 4, 8
    
    # Strategy: Split into 2x2 * 2x2 subproblems
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x8) = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    # B1 (2x8) = [B1_0 B1_1 B1_2 B1_3] each 2x2
    # B2 (2x8) = [B2_0 B2_1 B2_2 B2_3] each 2x2
    # C = A1*B1_0 + A1*B1_1 + A1*B1_2 + A1*B1_3 + A2*B2_0 + A2*B2_1 + A2*B2_2 + A2*B2_3
    # Each 2x2*2x2 can be done with Strassen (rank 7)
    # Total: 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen for 2x2 * 2x2:
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
    
    # Strassen coefficients for 2x2*2x2:
    # For input indices (i,j) where i in {0,1}, j in {0,1} for A
    # and (j,k) where j in {0,1}, k in {0,1} for B
    # Output (i,k)
    
    # Let's define the Strassen decomposition for 2x2*2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen multipliers:
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B10-B11)
    # m4 = A11*(B00-B10)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For each 2x2*2x2 subproblem, we need 7 "multiplications"
    # In the tensor decomposition, each "multiplication" corresponds to one column in U,V,W
    
    # Let's assign columns for each of the 8 subproblems
    # Subproblem (a, b) where a in {0,1} (which 2x2 block of A), b in {0,1,2,3} (which 2x2 block of B)
    
    col = 0
    for a in range(2):  # A block
        for b in range(4):  # B block
            # This subproblem computes A_a * B_b where A_a is 2x2, B_b is 2x2
            # A_a corresponds to A[a, b*2:(b+1)*2]
            # B_b corresponds to B[a*2:(a+1)*2, b*2:(b+1)*2]
            
            # Strassen multipliers for this subproblem
            # m1 = (A00+A11)*(B00+B11)
            # In terms of original indices:
            # A00 = A[a, b*2], A11 = A[a+1, b*2+1]
            # B00 = B[a*2, b*2], B11 = B[a*2+1, b*2+1]
            
            # m1
            U[a*2+0][col] = 1  # A[a, b*2]
            U[a*2+1][col] = 1  # A[a+1, b*2+1]
            V[a*2*8 + b*2][col] = 1  # B[a*2, b*2]
            V[a*2*8 + b*2+1][col] = 1  # B[a*2+1, b*2+1]
            W[0*2+0][col] = 1  # C[0,0] contribution: +1
            W[0*2+1][col] = 0
            W[1*2+0][col] = 0
            W[1*2+1][col] = 1  # C[1,1] contribution: +1
            col += 1
            
            # m2 = (A10+A11)*B00
            # A10 = A[a+1, b*2], A11 = A[a+1, b*2+1]
            # B00 = B[a*2, b*2]
            U[a*2+0][col] = 0
            U[a*2+1][col] = 1  # A[a+1, b*2]
            U[a*2+1][col] = 1  # A[a+1, b*2+1] -- wait, this is wrong
            # Let me redo this more carefully
            
            col -= 1  # back up
            
            # Let me restart with a cleaner approach
            pass
    
    # This is getting complicated. Let me use a different strategy.
    # Just return schoolbook for now and try to improve.
    return None

def _search_lower_rank(n, m, p, max_rank=None):
    """
    Search for a lower-rank decomposition using a heuristic approach.
    This is a placeholder - we'll use a more systematic method.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    # Lower bound
    lb = max(n*m, m*p, p*n)
    
    # Try ranks from lb+1 up to schoolbook
    # For now, just return schoolbook
    return None

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0  # seconds before deadline
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 64
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find better decompositions
    # Known: rank(2x4 * 4x8) is between 32 and 64
    # Let's try some structured decompositions
    
    # Attempt 1: Block Strassen with rank 56
    # Split A (2x4) into [A1 A2] (2x2 each)
    # Split B (4x8) into [B1; B2] (2x8 each)
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x8
    # Split B_i into four 2x2 blocks
    # Each 2x2*2x2 uses Strassen (7 mults)
    # Total: 2 * 4 * 7 = 56
    
    try:
        result_56 = _construct_rank56(n, m, p)
        if result_56 is not None and _verify(result_56["U"], result_56["V"], result_56["W"], n, m, p):
            best_result = result_56
            best_rank = 56
            print(f"Found valid rank 56 decomposition")
    except Exception as e:
        print(f"Rank 56 attempt failed: {e}")
    
    if time.time() > deadline - safety_margin:
        return best_result
    
    # Attempt 2: Try rank 48 or other structured approaches
    # For now, return best found
    return best_result

def _construct_rank56(n, m, p):
    """
    Construct a rank-56 decomposition using block Strassen.
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x8) = [B1; B2] where B1, B2 are 2x8
    C = A1*B1 + A2*B2
    
    For each A_i * B_i (2x2 * 2x8):
    Split B_i into four 2x2 blocks: B_i = [B_i0 B_i1 B_i2 B_i3]
    A_i * B_i = A_i*B_i0 + A_i*B_i1 + A_i*B_i2 + A_i*B_i3
    Each 2x2*2x2 uses Strassen (rank 7)
    Total: 2 * 4 * 7 = 56
    """
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(g-h)
    # m4 = d(e-g)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For a 2x2*2x2 subproblem at position (ai, bj) where:
    # A block: rows ai*2 to ai*2+1, cols bj*2 to bj*2+1
    # B block: rows ai*2 to ai*2+1, cols bj*2 to bj*2+1
    # Output: rows ai*2 to ai*2+1, cols bj*2 to bj*2+1
    
    col = 0
    for ai in range(2):  # A block index (0 or 1)
        for bj in range(4):  # B block index (0,1,2,3)
            # This computes A[ai*2:ai*2+2, bj*2:bj*2+2] * B[ai*2:ai*2+2, bj*2:bj*2+2]
            # Wait, B is 4x8, so B[ai*2:ai*2+2, bj*2:bj*2+2] makes sense
            # A is 2x4, so A[ai*2:ai*2+2, ...] doesn't work since A only has 2 rows
            # A is 2x4, so we can't split it into 2x2 blocks vertically
            # A = [A_top, A_bottom] doesn't work, A has 2 rows
            # So A[0, bj*2:bj*2+2] and A[1, bj*2:bj*2+2]
            # Actually, A is 2x4, so for block bj, we take A[:, bj*2:bj*2+2] which is 2x2
            # B is 4x8, for block (ai, bj), we take B[ai*2:ai*2+2, bj*2:bj*2+2] which is 2x2
            
            # A_block = A[:, bj*2:bj*2+2] (2x2)
            # B_block = B[ai*2:ai*2+2, bj*2:bj*2+2] (2x2)
            # C_block = A_block * B_block (2x2)
            # This contributes to C[:, bj*2:bj*2+2]
            
            # But C = A*B where A is 2x4, B is 4x8
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # For fixed k in [bj*2, bj*2+2), C[i,k] = sum_j A[i,j] * B[j,k]
            # = A[i, bj*2] * B[bj*2, k] + A[i, bj*2+1] * B[bj*2+1, k] + ... 
            # Wait, j goes from 0 to 3, so:
            # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
            
            # So if we group by A's columns:
            # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
            # = A1[i, :2] * B1[:, k] + A2[i, :2] * B2[:, k]
            # where A1 = A[:, 0:2], A2 = A[:, 2:4], B1 = B[0:2, :], B2 = B[2:4, :]
            
            # So C = A1 * B1 + A2 * B2
            # A1 is 2x2, B1 is 2x8
            # A1 * B1: for each column k of B1, compute A1 * B1[:, k]
            # B1[:, k] is a 2x1 vector
            # A1 * B1[:, k] is 2x1
            
            # To use Strassen, we need 2x2 * 2x2
            # Split B1 into four 2x2 blocks: B1 = [B1_0 B1_1 B1_2 B1_3]
            # A1 * B1 = A1 * B1_0 + A1 * B1_1 + A1 * B1_2 + A1 * B1_3
            # Each A1 * B1_j is 2x2 * 2x2
            
            # So for ai=0 (A1, B1):
            #   For bj in 0..3: compute A1 * B1_j using Strassen
            # For ai=1 (A2, B2):
            #   For bj in 0..3: compute A2 * B2_j using Strassen
            
            # A1 = A[:, 0:2], so A1[i, j] = A[i, j] for i in {0,1}, j in {0,1}
            # A2 = A[:, 2:4], so A2[i, j] = A[i, j+2] for i in {0,1}, j in {0,1}
            
            # B1 = B[0:2, :], so B1[i, k] = B[i, k] for i in {0,1}, k in {0..7}
            # B1_j = B1[:, j*2:j*2+2] = B[0:2, j*2:j*2+2]
            
            # B2 = B[2:4, :], so B2[i, k] = B[i+2, k] for i in {0,1}, k in {0..7}
            # B2_j = B2[:, j*2:j*2+2] = B[2:4, j*2:j*2+2]
            
            # So for (ai, bj):
            # A_block[i, j] = A[i, bj*2 + j] if ai==0, else A[i, bj*2 + j] -- wait
            # If ai==0: A_block = A1, so A_block[i, j] = A[i, j] for j in {0,1}
            #   But we're doing bj-th block, so A_block should be A[:, bj*2:bj*2+2]
            #   A_block[i, j] = A[i, bj*2 + j]
            # If ai==1: A_block = A2, but A2 is A[:, 2:4]
            #   For bj-th block of B2, we need A2 * B2_bj
            #   A2[i, j] = A[i, 2 + j]
            #   But B2_bj = B[2:4, bj*2:bj*2+2]
            #   So the product is A2 * B2_bj, which gives a 2x2 matrix
            #   This contributes to C[:, bj*2:bj*2+2]
            
            # Actually, I think I'm overcomplicating this.
            # Let me just define it directly.
            
            # For the subproblem (ai, bj):
            # We want to compute the contribution to C from A's columns in [bj*2, bj*2+2) 
            # and B's rows in [ai*2, ai*2+2)
            
            # Wait, that's not right either. Let me go back to basics.
            
            # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
            
            # Group by j:
            # C[i, k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
            
            # Group j into pairs:
            # C[i, k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
            
            # First term: A[i,0:2] * B[0:2, k]
            # Second term: A[i,2:4] * B[2:4, k]
            
            # For fixed k, let's write:
            # C[i, k] = sum_{j=0}^{1} A[i, j] * B[j, k] + sum_{j=2}^{3} A[i, j] * B[j, k]
            
            # Now group k into pairs:
            # For k in {0,1}: C[i, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
            # For k in {2,3}: C[i, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
            # etc.
            
            # So C = A_left * B_top + A_right * B_bottom
            # where A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
            #       B_top = B[0:2, :] (2x8), B_bottom = B[2:4, :] (2x8)
            
            # Now A_left * B_top is 2x2 * 2x8
            # Split B_top into four 2x2 blocks: B_top = [B_top_0 B_top_1 B_top_2 B_top_3]
            # A_left * B_top = A_left * B_top_0 + A_left * B_top_1 + A_left * B_top_2 + A_left * B_top_3
            
            # Each A_left * B_top_j is 2x2 * 2x2, use Strassen (7 mults)
            # Total for A_left * B_top: 4 * 7 = 28
            # Similarly for A_right * B_bottom: 28
            # Total: 56
            
            # Now let's implement this.
            pass
    
    # This is getting too complex inline. Let me use a different approach.
    # I'll construct the Strassen decomposition directly.
    
    # For each of the 8 subproblems (2 A-blocks * 4 B-blocks), use Strassen
    # A-block 0: A[:, 0:2], A-block 1: A[:, 2:4]
    # B-block j for A-block 0: B[0:2, j*2:j*2+2]
    # B-block j for A-block 1: B[2:4, j*2:j*2+2]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            # A_sub = A[:, a_block*2 : a_block*2+2] (2x2)
            # B_sub = B[a_block*2 : a_block*2+2, b_block*2 : b_block*2+2] (2x2)
            # Wait, that's not right. B_sub should be B[?, b_block*2:b_block*2+2]
            # For a_block=0: B_sub = B[0:2, b_block*2:b_block*2+2]
            # For a_block=1: B_sub = B[2:4, b_block*2:b_block*2+2]
            
            # C_sub = A_sub * B_sub (2x2)
            # This contributes to C[:, b_block*2:b_block*2+2]
            
            # Strassen for A_sub * B_sub:
            # A_sub = [[a, b], [c, d]] where a=A[0, a_block*2], b=A[0, a_block*2+1], c=A[1, a_block*2], d=A[1, a_block*2+1]
            # B_sub = [[e, f], [g, h]] where e=B[a_block*2, b_block*2], f=B[a_block*2, b_block*2+1], g=B[a_block*2+1, b_block*2], h=B[a_block*2+1, b_block*2+1]
            
            # m1 = (a+d)(e+h)
            U[0][col] = 1  # a = A[0, a_block*2]
            U[1][col] = 1  # d = A[1, a_block*2+1]
            V[a_block*2*8 + b_block*2][col] = 1  # e = B[a_block*2, b_block*2]
            V[a_block*2*8 + b_block*2 + 1][col] = 1  # h = B[a_block*2+1, b_block*2+1]
            # C[0, b_block*2] += m1, C[1, b_block*2+1] += m1
            W[b_block*2*2 + 0][col] = 1  # C[0, b_block*2]
            W[b_block*2*2 + 3][col] = 1  # C[1, b_block*2+1]
            col += 1
            
            # m2 = (c+d)e
            U[0][col] = 0
            U[1][col] = 1  # c = A[1, a_block*2]
            U[1][col] = 1  # d = A[1, a_block*2+1] -- ERROR: can't have two 1s in same cell
            # This approach is flawed because Strassen requires sums/differences
            # which means U, V, W entries can be -1, 0, 1
            # But we can't have U[1][col] = 1 twice
            
            col -= 1  # back up
            break
        break
    
    # The issue is that Strassen uses linear combinations, so U, V, W can have entries -1, 0, 1
    # But each (row, col) pair can only have one value
    # So for m1 = (a+d)(e+h), we need:
    # U[0, col] = 1 (for a)
    # U[1, col] = 1 (for d)
    # V[0, col] = 1 (for e) -- but V's row is different
    # Actually, U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (16, rank)
    
    # For m1 = (A[0, a_block*2] + A[1, a_block*2+1]) * (B[a_block*2, b_block*2] + B[a_block*2+1, b_block*2+1])
    # U[0, col] = 1  (A[0, a_block*2])
    # U[1, col] = 1  (A[1, a_block*2+1])
    # V[a_block*2*8 + b_block*2, col] = 1  (B[a_block*2, b_block*2])
    # V[a_block*2*8 + b_block*2+1, col] = 1  (B[a_block*2+1, b_block*2+1])
    # This is valid! U[0] and U[1] are different rows.
    
    # Let me redo this properly.
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            # A_sub indices: A[0, a_block*2], A[0, a_block*2+1], A[1, a_block*2], A[1, a_block*2+1]
            # B_sub indices: B[a_block*2, b_block*2], B[a_block*2, b_block*2+1], B[a_block*2+1, b_block*2], B[a_block*2+1, b_block*2+1]
            # C_sub indices: C[0, b_block*2], C[0, b_block*2+1], C[1, b_block*2], C[1, b_block*2+1]
            
            a00 = 0 * m + a_block * 2  # A[0, a_block*2]
            a01 = 0 * m + a_block * 2 + 1  # A[0, a_block*2+1]
            a10 = 1 * m + a_block * 2  # A[1, a_block*2]
            a11 = 1 * m + a_block * 2 + 1  # A[1, a_block*2+1]
            
            b00 = a_block * 2 * p + b_block * 2  # B[a_block*2, b_block*2]
            b01 = a_block * 2 * p + b_block * 2 + 1  # B[a_block*2, b_block*2+1]
            b10 = (a_block * 2 + 1) * p + b_block * 2  # B[a_block*2+1, b_block*2]
            b11 = (a_block * 2 + 1) * p + b_block * 2 + 1  # B[a_block*2+1, b_block*2+1]
            
            c00 = b_block * 2 * n + 0  # C[0, b_block*2]
            c01 = b_block * 2 * n + 1  # C[0, b_block*2+1]
            c10 = b_block * 2 * n + 2  # C[1, b_block*2]
            c11 = b_block * 2 * n + 3  # C[1, b_block*2+1]
            
            # m1 = (a00 + a11) * (b00 + b11)
            U[a00][col] = 1
            U[a11][col] = 1
            V[b00][col] = 1
            V[b11][col] = 1
            W[c00][col] = 1
            W[c11][col] = 1
            col += 1
            
            # m2 = (a10 + a11) * b00
            U[a10][col] = 1
            U[a11][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1
            col += 1
            
            # m3 = a00 * (b10 - b11)
            U[a00][col] = 1
            V[b10][col] = 1
            V[b11][col] = -1
            W[c01][col] = 1
            col += 1
            
            # m4 = a11 * (b00 - b10)
            U[a11][col] = 1
            V[b00][col] = 1
            V[b10][col] = -1
            W[c00][col] = 1
            W[c10][col] = 1
            col += 1
            
            # m5 = (a00 + a01) * b11
            U[a00][col] = 1
            U[a01][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1
            W[c01][col] = 1
            col += 1
            
            # m6 = (a10 - a00) * (b00 + b01)
            U[a10][col] = 1
            U[a00][col] = -1
            V[b00][col] = 1
            V[b01][col] = 1
            W[c11][col] = 1
            col += 1
            
            # m7 = (a01 - a11) * (b10 + b11)
            U[a01][col] = 1
            U[a11][col] = -1
            V[b10][col] = 1
            V[b11][col] = 1
            W[c00][col] = 1
            W[c11][col] = -1
            col += 1
    
    # Verify col == 56
    assert col == 56, f"Expected 56 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
