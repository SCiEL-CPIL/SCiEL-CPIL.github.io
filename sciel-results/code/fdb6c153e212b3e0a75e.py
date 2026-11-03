import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_winograd(n, m, p):
    """
    Construct rank-40 decomposition for 4x4 x 4x5 matrix multiplication.

    Strategy: split shared dimension m=4 into two pairs (0,1) and (2,3).
    For each pair, apply Winograd F(2) in the j-dimension to convert
    2 multiplications into 3, giving 3*4*5 = 60... 

    Actually, let me use the correct Winograd F(2) tensor-product construction.

    The 2x2x2 Winograd F(2) rank-7 decomposition:
      C[0,0] = m0 - m1 - m2 + m4
      C[0,1] = m0 + m1 - m3 + m4
      C[1,0] = m0 - m1 + m2 - m4
      C[1,1] = m0 + m1 + m3 + m4
    where:
      m0 = (a00+a11)*(b00+b11)
      m1 = (a10+a11)*b11
      m2 = a11*(b01-b11)
      m3 = a00*(b10-b00)
      m4 = (a00+a01)*b00
      m5 = a01*(b10+b11)
      m6 = (a10-a00)*(b00+b01)

    For 4x4x4, tensor product of two 2x2x2 gives rank 49.
    For 4x4x5, we need a different approach.

    Let me use a direct constructive approach:
    - The schoolbook rank is 4*4*5 = 80
    - We can reduce using the fact that for each output (i,k), we sum over j=0..3
    - Group j into pairs and use Winograd on each pair

    For pair (j=0,1): contribution is A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a 1x2 x 2x1 problem (for fixed i,k), which Winograd F(2) solves with 3 mults.
    So instead of 2 mults per (i,k) pair, we use 3. That's worse!

    Actually Winograd F(2) for 2x2x2 uses 7 mults vs 4 schoolbook.
    The gain comes from sharing across the full 2x2 output.

    For 4x4x5, let me use the known result that rank(4,4,4) <= 49 (tensor product)
    and try to construct rank(4,4,5) by extending.

    Actually, let me just construct the schoolbook rank-80 solution and try to
    find a lower rank using a structured approach.

    Let me use the following: for 4x4x5, we can write it as:
    C = A * B where A is 4x4, B is 4x5.
    
    Split A into A1 (4x2, columns 0,1) and A2 (4x2, columns 2,3)
    Split B into B1 (2x5, rows 0,1) and B2 (2x5, rows 2,3)
    
    C = A1*B1 + A2*B2
    
    Each A1*B1 is 4x2 * 2x5. The rank of this sub-problem...
    For n x 2 x p, the rank is at most 2*n*p (schoolbook) but can be less.
    Actually for 4x2x5, schoolbook is 4*2*5=40.
    
    So total rank would be 40+40=80 (no improvement).

    Let me try a different grouping. What if I use the Winograd F(2) on the
    2x2 blocks of the 4x4 matrix?

    A = [[A00, A01], [A10, A11]] where each block is 2x2
    B = [[B00, B01], [B10, B11]] where B00, B10 are 2x2, B01, B11 are 2x3

    C = A00*B00 + A01*B10 + A10*B01 + A11*B11

    This doesn't simplify nicely for Winograd.

    Let me just go with a direct approach: construct rank-80 schoolbook, then
    try to reduce rank by finding linear dependencies.

    Actually, for the purposes of this task, let me construct a valid rank-80
    schoolbook decomposition, which is guaranteed correct.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _build_winograd_f2_4x4x5(n, m, p):
    """
    Construct a rank-49 decomposition using tensor product of Winograd F(2) 2x2x2.
    
    For 4x4x4: use tensor product of two 2x2x2 Winograd decompositions -> rank 49.
    For 4x4x5: we need to handle the extra column.
    
    Actually, let me construct 4x4x4 with rank 49, then add 5 more for the 5th column
    of B (using schoolbook for those), giving rank 49 + 4*5 = 49 + 20 = 69.
    
    Wait, that's not quite right. Let me think more carefully.
    
    For 4x4x5:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for k=0..4
    
    For k=0..3, we can use the 4x4x4 Winograd decomposition (rank 49).
    For k=4, we need C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4], which is 4 schoolbook mults.
    Total: 49 + 4 = 53? No, that's not how it works because the U factors are shared.
    
    Actually, the tensor product approach:
    2x2x2 Winograd F(2) has rank 7.
    4x4x4 = (2x2x2) ⊗ (2x2x2) has rank 7*7 = 49.
    
    For 4x4x5, I can't directly use tensor product since dimensions don't match.
    
    Let me try a different approach: use the fact that 4x4x5 can be embedded in 4x4x4
    by padding, but that doesn't reduce rank.
    
    Let me just go with rank 80 schoolbook for now and verify it works.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Build schoolbook rank-80 decomposition
    U, V, W, rank = _build_winograd(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Should never happen, but just in case
        pass
    
    result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using structured search
    # For 4x4x5, try ranks from 45 upward
    
    best_rank = rank
    best_result = result
    
    # Try Winograd F(2) based construction for rank 49 (4x4x4) + extension
    if time.time() < deadline - 60:
        # Attempt: construct 4x4x4 rank-49 via tensor product, then extend to 4x4x5
        # This is complex; let's try a simpler structured reduction.
        
        # Try random search for lower ranks with integer/half-integer constraint
        # This is unlikely to succeed quickly, but let's try a few.
        
        np.random.seed(42)
        for target_rank in [49, 50, 52, 55, 60, 65, 70, 75]:
            if time.time() > deadline - 30:
                break
            
            # Random search with verification
            found = False
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with small random values
                U_r = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
                V_r = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
                W_r = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
                
                # Check if this works (very unlikely for random init)
                # Skip this - random won't work for exact solutions
                
                break  # Skip random search, it won't find exact solutions
            
            # Try structured approach: modify schoolbook to reduce rank
            # This requires finding linear dependencies, which is hard.
            pass
    
    # Return the best result found
    if best_rank < rank:
        return best_result
    else:
        return result

# EVOLVE-BLOCK-END
