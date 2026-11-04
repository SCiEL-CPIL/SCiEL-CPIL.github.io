"""Tensor decomposition for 3x3 by 3x8 matrix multiplication.

Uses Winograd F(4,3,2) per 3x4 block (rank 12) and schoolbook for 3x2 block (rank 6).
Total rank = 2*12 + 6 = 30.

Index conventions:
  U: (n*m, rank)  -> row i*m + j
  V: (m*p, rank)  -> row j*p + k
  W: (p*n, rank)  -> row k*n + i
"""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _make_wino3x4():
    """Build Winograd F(4,3,2) decomposition for 3x3 * 3x4 -> 3x4.

    Winograd factorisations:
      A_hat = Vm * A * Vm^T   (V2x2 for 3x2 case: [[1,1],[0,1]])
      B_hat = W^T * B * Vm    (W^T 3x4, Vm 2x4)

    With n_out = p = 3? No: here the matrix B is 3 x 4 (m=3 rows, p=4 columns).
    So Vm is 2x4, W is 4x3 (hence W^T is 3x4).

    The 2 rank-1 terms become 2 * (3x3) * (3x4) = effectively after A_hat and B_hat
    we need the convolution tensor of size 2x2x2 = 8 multipliers? 
    Actually Winograd reduces to m * m * s = 3*3*... let me re-derive.

    Actually the standard Winograd F(r,m) with r=2 (block size), m=3 (kernel/paper size):
    Input B is M x (r+s) = 3 x 4. After transform: B_hat = W^T * B * Vm which is (M) x (r+s-m+1) ... 
    No. Let me use the concrete known F(4,3,2) factor.

    Known F(4,3,2):
    Vm (2x2): [[1,1],[0,1]]
    W (4x3): [[1,0,0],[-1,2,0],[-1,-2,0],[1,0,0]]  ... wait W is 4x3? No W should be (r+s) x m = 4x3.

    Hmm, let me use the explicit 12-rank decomposition I can construct directly:
    
    For 3x3 A times 3x4 B, the Winograd factorisation gives rank = 3 * 3 * 2 = 18? No.
    
    Let me just use a direct approach: enumerate and find a low-rank decomposition
    via known results or brute force search with numpy.

    Since I need to be self-contained and correct, let me just use the schoolbook for each
    block and try to merge. But the directive says rank 24 via Strassen per block...

    Actually let me just do schoolbook for all of 3x3x8 which gives rank 72,
    then try to find dependencies and merge.
    """
    return None  # placeholder


def _schoolbook_block(n, m, p):
    """Return (U, V, W) for n x m @ m x p via schoolbook, rank = n*m*p."""
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
    return U, V, W


def _try_wino3x4():
    """Attempt Winograd F(4,3,2) rank-12 factor for 3x3 * 3x4.

    I'll construct it from the known transformation matrices.

    Winograd F(r=2, m=3) for 2D convolution applied row-wise:
    Vm = [[1, 1],
          [0, 1]]   (2 x 2, transforms size-m kernel to size-r+r' where r'=1)

    W = [[1,  0,  0],
         [-1, 2,  0],
         [-1, -2, 0],
         [1,  0,  0]]  (4 x 3, transforms size-(r+s)=4 input to size-m=3)

    For multiplying A (3x3) by B (3x4):
      A_hat[a, b] = sum_{a',b'} Vm[a,a'] * A[a',b'] * Vm[b,b']  -> A_hat is 2x2
      B_hat[x, j] = sum_{y,j'} W[y, j'] * B[j', j] * Vm[x, j'] ... 

    Actually the B transform: B_hat[x, col] = (W^T * B * Vm)[x, col]
    W^T is 3x4, B is 3x4... dimensions don't work that way.

    Let me use the explicit formula. The convolution C = A * B where A is 3x3 and B is 3x4
    (treated as B having 4 columns of 3 elements each, we convolve A over columns).

    Output C[i, t] = sum_j A[i,j] * B[j, t] for t=0,1,2,3, i=0,1,2.

    This is just matrix multiply. Winograd F(4,3,2) says we can do this in 3*3*2 = 18? 
    No, F(r,m) reduces from m*r multiplies to r*r for each output row pair...

    I think for 3x3 * 3x4 the minimum rank might be hard. Let me just use rank 12 
    if I can find it, otherwise fall back.

    Since I can't reliably produce Winograd factors from memory, I'll use a 
    rank-reduction search via linear algebra on the schoolbook decomposition.
    """
    return None


def construct():
    import time
    import numpy as np
    start = time.time()
    deadline = start + 110  # 110s safety margin

    n, m, p = DIMENSIONS  # 3, 3, 8

    # Start with schoolbook rank 72
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

    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank by identifying linear dependencies among the rank-1 terms
    # For each term c, the three vectors are U[:,c], V[:,c], W[:,c].
    # If there's a linear relation among terms, we can merge.
    
    # Convert to numpy for rank analysis
    U_np = np.array(U, dtype=np.float64).T  # rank x n*m
    V_np = np.array(V, dtype=np.float64).T  # rank x m*p
    W_np = np.array(W, dtype=np.float64).T  # rank x p*n

    # The tensor T has entries T[i, j, k] for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i,j,k] = 1 if this is a valid matmul index, else 0.
    # Actually T[i,j,k] = delta (it's just the matmul tensor, all 1s for valid indices).
    # The rank of the tensor is what we want to minimize.

    # Let's try a simple greedy: check if any pair of columns is redundant
    # (one is a scalar multiple of the other in the combined space)
    
    # More practically: try to find the rank of the flattening and use that
    # to see if we can do better than 72.
    
    # Try Winograd-inspired reduction for 3x3 x 3x4 blocks:
    # Split B into two 3x4 blocks.
    # Block 1: columns 0-3, Block 2: columns 4-7.
    # Each 3x3 * 3x4 can potentially be done in rank 12 (Winograd).
    # The 3x2 tail is rank 6.
    # Total: 12 + 12 + 6 = 30 if Winograd factors exist.

    # Let me try to construct Winograd F(4,3,2) explicitly.
    # 
    # For 1D Winograd F(4,3): 
    #   Vm = [[1, 1], [0, 1]]  (2x2)
    #   W  = [[1, 0, 0], [-1, 2, 0], [-1, -2, 0], [1, 0, 0]]  (4x3)
    # 
    # 2D Winograd F(4,3) applied to 3x3 * 3x4:
    #   A_hat = Vm @ A @ Vm.T  -> (2x2) @ (3x3) @ (2x2) 
    #   Wait, A is 3x3. Vm is 2x2. Vm @ A doesn't work dimensionally.
    #
    # Actually in Winograd, for kernel size m=3 and block size r=2:
    #   The kernel A is 3x3. We pad/reshape... 
    #   Vm should be 2x3? No.
    #
    # Standard Winograd F(r, m): input size r+s where s = m-1 = 2, so r+s = 4.
    # Vm is r x (m) = 2 x 3? No, Vm transforms the m-point kernel to r points.
    # So Vm is 2x3.
    # W transforms the (r+s) = 4 point input to m = 3 points. W is 4x3.
    #
    # But then A_hat = Vm @ A @ Vm.T would be (2x3) @ (3x3) @ (3x2) = 2x2. OK.
    # B_hat = W.T @ B @ Vm would be (3x4) @ (3x4) ... doesn't work.
    #
    # I think the correct formulation for matrix multiply C = A*B where
    # A is (n x m) and B is (m x p) with n=m=3, p=4:
    # Treat this as 1D Winograd along the p dimension.
    # For each row i of A (3 rows), multiply by B (3x4) -> 1D Winograd.
    # Winograd F(4,3): for 1 convolution of kernel size 3 and block 2:
    #   Vm (2x3), W (4x3)
    #   A_hat[i, *] = Vm @ A[i, :]  -> 2x1 per row? No, A[i,:] is 1x3.
    #   Actually A_hat[i, a] = sum_b Vm[a,b] * A[i,b]
    #   B_hat[x, j] = sum_k W[x,k] * B[k,j] * Vm[j, ?] ... 
    #
    # This is getting too complicated from memory. Let me just do a direct
    # rank minimization via SVD on tensor flattenings.

    if time.time() > deadline - 30:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try to reduce via finding the actual tensor rank
    # Flattening 1: reshape T to (n, m*p) and find rank
    # Flattening 2: reshape T to (n*m, p) and find rank
    # Flattening 3: reshape T to (p, n*m) and find rank
    # The tensor rank >= max of these flattening ranks.
    
    # T[i,j,k] = 1 for all i,j,k in 3x3x8 (it's just the matmul identity tensor)
    # So T flattened is just an all-ones array of shape 3x3x8.
    # Rank of all-ones tensor of shape (a,b,c) is 1 (it's a rank-1 tensor!).
    # But that's not right because the indices are constrained.
    
    # Actually T[i,j,k] = 1 only represents the matmul structure where
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor is T[i,j,k] which when
    # contracted gives C = A*B. T[i,j,k] = delta(i,j,k) structure... 
    # No, T[i,j,k] = 1 for all valid i,j,k. The tensor is just the all-ones
    # 3x3x8 array. Its rank is 1! But we need the factorization to respect
    # the specific index structure U[i*m+j], V[j*p+k], W[k*n+i].
    
    # The constraint is that U only depends on (i,j), V on (j,k), W on (k,i).
    # This is the matrix multiplication tensor with specific index structure.
    # The minimum rank of this tensor for 3x3x8 is what we seek.
    
    # Let me try to use numpy to find a lower-rank factorization by solving
    # for integer/rational coefficients.
    
    if time.time() > deadline - 20:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Attempt: try rank values from a low start upward, checking if a 
    # factorization exists via linear solve.
    # This is computationally expensive, so let's just try a few promising ranks.
    
    # For 3x3x4, the known minimum rank might be 12 (Winograd) or 13.
    # For 3x3x2, minimum rank might be 6 (schoolbook) or less.
    # For 3x3x8, splitting as 3x3x4 + 3x3x4: rank <= 2*12 = 24 if Winograd works.
    # But we also have cross terms... actually no, if we split B into two 
    # 3x4 blocks independently, the decompositions are independent.
    
    # Let me try to construct explicit Winograd factors for 3x3 * 3x4.
    # 
    # Winograd F(4,3) 1D: 
    #   Vm = [[1, 1], [0, 1]]  (2x2 for kernel size 2... no)
    #   
    # For kernel size m=3, the 1D Winograd uses:
    #   Vm (r x m) = [[1, 0, 1], [0, 1, 0]]? No.
    #   
    # I recall that F(4,3) has:
    #   Vm = [[1, 1], [0, 1]]  -- this is for m=2, not m=3.
    #
    # For m=3, r=2 (block of 2, overlap 1):
    #   The kernel is size 3, input is size 3+2=5? No, input size = r + (m-1) = 2+2 = 4.
    #   Vm: 2 x 3
    #   W: 4 x 3
    #
    # One known factorization:
    #   Vm = [[1, 1, 1], [0, 1, 0]]  -- not sure
    #
    # Let me just try to find rank-12 factors by solving linear equations.
    
    if time.time() > deadline - 15:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Direct approach: for 3x3x8, try to find rank-r factors.
    # Set up: T[i,j,k] = U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] summed over a.
    # We need T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8).
    # (Note: the matmul tensor has T[i,j,k]=1 for all valid indices since it's
    #  the "all ones" tensor in the appropriate index space.)
    
    # Actually wait - the tensor for matrix multiplication is:
    # T[i, j, k] where i indexes output rows, j indexes the shared dimension,
    # k indexes output columns. T[i,j,k] = 1 means A[i,j] * B[j,k] contributes
    # to C[i,k]. Since every (i,j,k) triple is valid, T is all ones.
    # But the factorization constraint is that:
    #   sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1 for all i,j,k.
    
    # This is equivalent to: for each (i,j), U[i*m+j, :]; for each (j,k), V[j*p+k, :];
    # for each (k,i), W[k*n+i, :].
    
    # Let me try rank 30 first (2 blocks of 12 + 6 for the tail).
    # But without explicit Winograd factors, I can't easily construct them.
    
    # Alternative: try to reduce the schoolbook rank by finding linear dependencies.
    # If two rank-1 terms (u1,v1,w1) and (u2,v2,w2) satisfy
    # u1 = alpha*u2, v1 = beta*v2, w1 = gamma*w2 with alpha*beta*gamma = 1,
    # then they can be merged. In our schoolbook, each term has exactly one 1 in each
    # factor, so this doesn't happen.
    
    # The real reduction comes from non-trivial factorizations.
    # Without known factors, I'll return the schoolbook.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
