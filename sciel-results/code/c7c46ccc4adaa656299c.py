"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if rank == 0:
        return False
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Compute decomposed tensor
    T_comp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_comp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    if np.max(np.abs(T_comp - T)) > tol:
        return False
    return True

def _make_schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W matrices for the 2x2x2 tensor with rank 7.
    
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each product, we need to express as sum_r U[i,r]*V[j,r]*W[k,r]
    # where i = a*2+b (a=row of A, b=col of A), j = b*2+c (b=row of B, c=col of B), k = c*2+a (c=row of C, a=col of C)
    
    # M1 = (A00+A11)(B00+B11)
    # This contributes to C00 and C11
    # For C00: a=0, c=0. Term: (A00+A11)*(B00+B11)
    # A00 is at i=0, B00 is at j=0, C00 is at k=0
    # A11 is at i=3, B11 is at j=3, C00 is at k=0
    # But we need a single rank-1 tensor for the whole product...
    # Actually Strassen works by computing 7 intermediate products, each of which
    # is a product of two linear combinations. Each such product gives rise to
    # a rank-1 tensor contribution.
    
    # Let me think more carefully. In Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # This is a scalar product of two sums. In the tensor decomposition framework,
    # we need to express the full multiplication as sum of rank-1 tensors.
    
    # Actually, the standard approach: each of the 7 Strassen products corresponds
    # to a rank-1 tensor in the 2x2x2 multiplication tensor.
    
    # For M1 = (A00+A11)(B00+B11):
    # The U vector has 1s at positions corresponding to A00 and A11
    # The V vector has 1s at positions corresponding to B00 and B11
    # The W vector... this is where it gets tricky.
    
    # Actually, I think the correct way is:
    # Each Strassen product M_i = u_i . A * B . v_i where u_i, v_i are coefficient vectors
    # And then C = sum_i M_i * w_i where w_i is a matrix of coefficients
    # This gives a rank-7 decomposition.
    
    # Let me use the standard formulation:
    # For 2x2 matrices A, B, C = AB:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor form, for each M_i:
    # M1: U has 1 at A00(i=0) and A11(i=3), V has 1 at B00(j=0) and B11(j=3)
    #     W: contributes to C00(k=0) with coeff +1 and C11(k=3) with coeff +1
    #     But W is a vector indexed by k=c*2+a, so C00 is k=0*2+0=0, C11 is k=1*2+1=3
    #     So W[0] = 1, W[3] = 1
    
    # Let me define properly:
    # U for M1: U[0]=1, U[3]=1 (A00, A11)
    # V for M1: V[0]=1, V[3]=1 (B00, B11)
    # W for M1: W[0]=1 (C00), W[3]=1 (C11)
    
    U[0,0] = 1; U[3,0] = 1  # M1: A00, A11
    V[0,0] = 1; V[3,0] = 1  # M1: B00, B11
    W[0,0] = 1; W[3,0] = 1  # M1: C00, C11
    
    # M2 = (A10+A11)B00
    # U: A10(i=2), A11(i=3)
    # V: B00(j=0)
    # W: C10(k=2) with +1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A00(B01-B11)
    # U: A00(i=0)
    # V: B01(j=1) with +1, B11(j=3) with -1
    # W: C01(k=1) with +1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # M4 = A11(B10-B00)
    # U: A11(i=3)
    # V: B10(j=2) with +1, B00(j=0) with -1
    # W: C00(k=0) with +1, C10(k=2) with +1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (A00+A10)B11
    # U: A00(i=0), A10(i=2)
    # V: B11(j=3)
    # W: C00(k=0) with -1, C01(k=1) with +1
    U[0,4] = 1; U[2,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A11-A00)(B00+B10)
    # U: A11(i=3) with +1, A00(i=0) with -1
    # V: B00(j=0), B10(j=2)
    # W: C11(k=3) with +1
    U[3,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01(i=1) with +1, A11(i=3) with -1
    # V: B10(j=2), B11(j=3)
    # W: C00(k=0) with +1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _block_decompose(n, m, p, block_sizes, strassen_threshold=2):
    """
    Decompose matrix multiplication using block Strassen where possible.
    block_sizes: tuple of (n_blocks, m_blocks, p_blocks) specifying how to partition.
    For simplicity, we'll try specific block configurations.
    """
    # For now, handle the specific case of 2x5x5 with blocks 2+2+1
    # This is complex; let's try a different approach.
    pass

def _try_block_strassen(n, m, p):
    """
    Try to decompose 2x5x5 using block Strassen with 2+2+1 partitioning.
    
    Partition A (2x5) into rows [0,1] (no block on n since n=2)
    Partition B (5x5) into 2+2+1 blocks along both dimensions
    Partition C (5x2) into 2+2+1 blocks along columns, 2 blocks along rows (n=2)
    
    Actually, for Strassen we need square blocks. Let me reconsider.
    
    For 2x5*5x5:
    - A is 2x5: can partition columns into 2+2+1
    - B is 5x5: can partition rows and columns into 2+2+1
    - C is 5x2: can partition rows into 2+2+1, columns into 2 (no split needed for n=2)
    
    This gets complicated. Let me try a simpler approach:
    Just use the full schoolbook but try to find linear dependencies.
    """
    return None

def _gaussian_elimination_reduce(U, V, W, n, m, p, target_rank, time_budget=30.0):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 tensors.
    Start from schoolbook and try to eliminate terms.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Start from schoolbook
    sb = _make_schoolbook(n, m, p)
    U = np.array(sb['U'], dtype=float)
    V = np.array(sb['V'], dtype=float)
    W = np.array(sb['W'], dtype=float)
    rank = U.shape[1]
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Try to find a set of indices to keep that still spans the tensor
    # The tensor has rank at most 50. We want to find the minimal subset.
    # Use matricization: flatten one mode and check rank.
    
    # Matricize along first mode: shape (n*m, m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # Get the column space of T_mat
    # Each column r of our decomposition contributes U[:,r] * (V[:,r] ⊗ W[:,r]).T
    # So the effective "columns" are V[:,r] ⊗ W[:,r]
    
    # We want to find a minimal set of these vectors that spans the column space.
    # This is equivalent to finding a maximal independent set.
    
    # Compute V⊗W for each rank term
    vecs = []
    for r in range(rank):
        vec = np.kron(V[:,r], W[:,r])
        vecs.append(vec)
    
    # Find a maximal independent set
    mat = np.column_stack(vecs)
    # Use QR with column pivoting
    from scipy.linalg import qr
    Q, R, piv = qr(mat, mode='complete', pivoting=True)
    
    # The rank of mat
    tol = 1e-10
    diag = np.abs(np.diag(R))
    effective_rank = np.sum(diag > tol)
    
    if effective_rank < rank:
        # We can reduce
        # Keep only the first effective_rank columns (in permuted order)
        keep = piv[:effective_rank]
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        
        if _verify(U_new, V_new, W_new, n, m, p, tol=1e-8):
            return U_new, V_new, W_new
    
    return None

def _try_direct_reduction(n, m, p, target_rank, time_budget=60.0):
    """
    Try to find a decomposition of exactly target_rank by optimizing
    from a random or structured initialization.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Try multiple random initializations
    best_result = None
    n_restarts = 5
    
    for restart in range(n_restarts):
        if not still_going():
            break
        
        np.random.seed(42 + restart * 104729)
        
        # Initialize with random values, scaled
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Use L-BFGS style optimization with careful step size
        lr = 0.02
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(3000):
            if not still_going():
                break
            
            # Compute loss and gradients
            diff = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                diff += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            diff -= T
            
            loss = np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(target_rank):
                grad_U[:,r] = 2 * np.einsum('ijk,j,k->i', diff, V[:,r], W[:,r])
                grad_V[:,r] = 2 * np.einsum('ijk,i,k->j', diff, U[:,r], W[:,r])
                grad_W[:,r] = 2 * np.einsum('ijk,i,j->k', diff, U[:,r], V[:,r])
            
            # Gradient step with adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 0:
                step_size = lr * 0.1 / grad_norm
                U -= step_size * grad_U
                V -= step_size * grad_V
                W -= step_size * grad_W
            
            # Decay learning rate
            if step % 500 == 0 and step > 0:
                lr *= 0.8
        
        if best_loss < 1e-10:
            if _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
                return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with schoolbook - always correct, rank = n*m*p
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Strategy 1: Try block Strassen with 2+2+1 partitioning
    # For 2x5*5x5, partition the 5 dimension into 2+2+1
    # This creates 3x3 = 9 blocks of various sizes
    # 2x2 * 2x2 blocks can use Strassen (rank 7 instead of 8)
    # 2x1, 1x2, 1x1 blocks use schoolbook
    
    # Let's construct this explicitly
    # A is 2x5, partition columns as [0,1], [2,3], [4]
    # B is 5x5, partition rows and columns as [0,1], [2,3], [4]
    # C is 5x2, partition rows as [0,1], [2,3], [4], columns as [0,1]
    
    # Block (i,j) of A (2x2, 2x2, 2x1, 1x2, 1x2, 1x1) multiplies
    # Block (j,k) of B (2x2, 2x2, 2x1, 2x2, 2x2, 2x1, 2x1, 2x1, 1x1)
    # to contribute to block (i,k) of C
    
    # This is getting very complex. Let me try a simpler approach:
    # Just try direct optimization for ranks 49, 48, 47, 46
    
    # Strategy 2: Direct optimization for reduced ranks
    for target_rank in [49, 48, 47, 46, 45]:
        if not still_going():
            break
        remaining = deadline - elapsed()
        if remaining < 15:
            break
        
        budget = min(remaining * 0.4, 20.0)
        
        result = _try_direct_reduction(n, m, p, target_rank, time_budget=budget)
        if result is not None:
            U_new, V_new, W_new = result
            best = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
