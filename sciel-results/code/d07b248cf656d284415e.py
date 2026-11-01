"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition.

We try to find a decomposition with rank below the schoolbook rank of 50.
For small dimensions like 2x5x5, the true rank is unknown but likely below 50.

Strategy:
- Start with schoolbook rank 50 as a guaranteed baseline.
- Use integer/half-integer entries (0, 0.5, 1, -0.5, -1) to keep exactness.
- Attempt to find a lower-rank decomposition via a structured approach:
  We exploit the fact that C = A * B where A is 2x5 and B is 5x5.
  The output C is 2x5. Each entry C[i][k] = sum_j A[i][j] * B[j][k].
  
  A bilinear decomposition of rank r means we can write:
  C[i][k] = sum_{r=1}^{R} u_r[i,j] * A[i][j] * v_r[j,k] * B[j,k] * w_r[k,i]
  
  where u_r, v_r, w_r are fixed tensors (the factor matrices).

  For the schoolbook, each term isolates one (i,j,k) triple.
  
  To reduce rank, we need to combine terms. One natural approach:
  For a fixed i, the row i of C is a linear combination of rows of B weighted by A[i].
  So C[i] = A[i] @ B (as a row vector).
  
  This means for each output row i, we need to compute A[i] @ B.
  If we could compute A[i] @ B using a rank-r_i bilinear algorithm for 1x5x5 matrix mult,
  then total rank would be sum of r_i for i=0,1.
  
  The rank of 1x5x5 matrix mult: C[k] = sum_j A[j] * B[j][k].
  This is just a vector-matrix product. The tensor is essentially the "identity" in j
  coupled with the row of B. Actually, this is just: for each k, C[k] = sum_j A[j] B[j,k].
  
  The bilinear complexity of 1xm x mxp is mp (schoolbook). Can we do better?
  For 1x5x5, the rank is 25 (each C[k] depends on all j). Actually no:
  C[k] = sum_j A[j] * B[j,k] for k=0..4.
  This is 5 separate dot products of length 5. The total number of bilinear terms is 25.
  But can we share computations? Since A[j] is shared across k, we might factor.
  
  Actually, the tensor for 1xm x mxp is: T_{j,k,l} where l is the output index (0..p-1),
  and the input A has m components, B has m*p components.
  T = sum_{j,k} A[j] * B[j,k] * e_k (output basis).
  This tensor has rank mp because each output coordinate k requires summing over j,
  and there's no sharing between different k's in a bilinear decomposition... 
  unless we use Strassen-like tricks. But for 1xm x mxp, the rank is exactly mp.
  
  So total rank would be 2*25 = 50, which is the schoolbook. No improvement.

  However, we might be able to do better by sharing between the two rows of A.
  Let A have rows a0 and a1. C[0] = a0 @ B, C[1] = a1 @ B.
  
  We can write:
  C[0] = a0 @ B
  C[1] = a1 @ B
  
  Consider the 2x5 matrix A. We need to compute A @ B where B is 5x5.
  
  Another approach: view B as a 5x5 matrix. We can use a rank-r decomposition for 5x5x5
  matrix multiplication, but we only use part of it. Actually, we need A (2x5) times B (5x5).
  
  Let's try a direct search with a small rank, using a genetic algorithm or local search
  over integer/half-integer entries.

  Given the time constraint, I'll implement a local search that starts from schoolbook
  and tries to merge terms.

  Actually, let me try a different approach: use the known result that for 2xnxn,
  the rank might be lower. Let me try rank 30, 35, 40, 45 and see if we can find
  a valid decomposition via local search.
"""
import time
import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]
N, M, P = 2, 5, 5
NM = N * M  # 10
MP = M * P  # 25
PN = P * N  # 10

def _schoolbook(n, m, p):
    """Guaranteed-exact schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x5x5():
    """
    Construct a rank-46 decomposition for 2x5x5 matrix multiplication using Strassen.
    
    Approach: Partition the 5 columns into blocks [0,1], [2,3], [4].
    For the 2x2 sub-blocks, use Strassen's 7-multiplication formula.
    For the remaining column, use schoolbook.
    
    A is 2x5: columns a0,a1,a2,a3,a4
    B is 5x5: rows b0..b4, each row has 5 columns
    C is 2x5: C[i,k] = sum_j A[i,j]*B[j,k]
    
    We group:
    - Columns 0,1 of A and rows 0,1 of B form a 2x2 block interaction
    - Columns 2,3 of A and rows 2,3 of B form another 2x2 block interaction  
    - Column 4 of A and row 4 of B form a 1x1 interaction
    
    For each 2x2 block, instead of 4 scalar mults, use Strassen's 7 mults.
    Total: 7 + 7 + 1*1*1 = 15 bilinear terms for the block structure.
    But we need to account for all output positions.
    
    Actually, let me think more carefully. The tensor is:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..4}, j in {0..4}
    
    This is a 2x5x5 tensor with 50 entries. The schoolbook uses 50 rank-1 terms.
    
    Strassen idea: For computing a 2x2 matrix product, we can use 7 mults instead of 8.
    Here we're computing C = A*B where A is 2x5 and B is 5x5.
    
    Let's partition B's rows into groups and A's columns into corresponding groups.
    If we group (j=0,1) together:
    C[:,k] += A[:,0:2] @ B[0:2,k]
    
    The subproblem A[:,0:2] (2x2) times B[0:2,k] (2x1) for each k.
    This is like computing a 2x2 matrix times a 2x5 matrix, but only the first 2 rows of B.
    
    Alternative cleaner approach: Think of it as 2 independent 1x5x5 problems that share structure.
    
    Let me try a different structured approach:
    - Use the fact that we only have 2 rows in A
    - Group the 5 columns of A into {0,1}, {2,3}, {4}
    - For groups {0,1} and {2,3}, apply Strassen-style sharing
    
    For group {0,1}: A[i,0], A[i,1] for i=0,1
    For group {2,3}: A[i,2], A[i,3] for i=0,1
    For group {4}: A[i,4] for i=0,1
    
    The contribution to C from group {0,1}:
    C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is a 2x2 x 2x5 subproblem (A is 2x2, B is 2x5, C is 2x5).
    Schoolbook rank for this: 2*2*5 = 20.
    
    With Strassen on the 2x2 part:
    We can compute A_2x2 * B_2x5 more efficiently.
    
    Actually, for a 2x2 times 2x5:
    Let A = [[a,b],[c,d]], B has rows [b00,b01,b02,b03,b04] and [b10,b11,b12,b13,b14]
    C[0,k] = a*B[0,k] + b*B[1,k]
    C[1,k] = c*B[0,k] + d*B[1,k]
    
    This is just 2 dot products of length 2, done 5 times = 20 mults schoolbook.
    Can Strassen help? Strassen helps for square matrix mult. For 2x2 * 2x5,
    we can view it as two independent 2x1 products... no sharing benefit.
    
    Let me try yet another approach. The key insight from the observation:
    "Using Strassen's algorithm on 2x2x2 sub-blocks reduces the rank from 50 to 46"
    
    So we need to find 4 places where we can replace 8 mults with 7, saving 4 total.
    
    Partition: Think of the 5x5 B as having a 2x2 block in the top-left corner.
    The contribution of B[0:2, 0:2] to C:
    C[i,0] += A[i,0]*B[0,0] + A[i,1]*B[1,0]
    C[i,1] += A[i,0]*B[0,1] + A[i,1]*B[1,1]
    
    This is a 2x2 (A block) times 2x2 (B block) producing 2x2 (C block).
    Schoolbook: 4 mults. Strassen: 7 mults. Wait, that's worse!
    
    I had it backwards. Strassen replaces 8 mults (for 2x2*2x2) with 7.
    But here the subproblem is 2x2 * 2x2 = 2x2, which schoolbook does in 4 mults.
    Strassen does it in 7. So Strassen is WORSE for 2x2!
    
    Strassen helps when you have larger blocks. For 2x2x2, Strassen rank is 7,
    schoolbook is 8. So we save 1 per 2x2x2 block.
    
    For our 2x5x5 problem, we can identify disjoint 2x2x2 sub-structures.
    
    Let me reconsider. The full tensor T_{i,j,k} = 1 for all i,j,k (it's the identity tensor).
    We want to decompose this into rank-1 tensors.
    
    A 2x2x2 sub-tensor (fixing i in {i0,i1}, j in {j0,j1}, k in {k0,k1}) has rank 7 via Strassen.
    Schoolbook for that sub-tensor would be 8.
    
    So if we can partition the 50 terms into some number of 2x2x2 blocks plus leftovers,
    and use Strassen for each 2x2x2 block, we save 1 per block.
    
    With 5 in each dimension, we can fit: floor(2/2)*floor(5/2)*floor(5/2) = 1*2*2 = 4 blocks of 2x2x2.
    That saves 4, giving rank 46.
    
    The 4 blocks:
    Block 1: i={0,1}, j={0,1}, k={0,1}
    Block 2: i={0,1}, j={0,1}, k={2,3}
    Block 3: i={0,1}, j={2,3}, k={0,1}
    Block 4: i={0,1}, j={2,3}, k={2,3}
    
    Remaining terms: those involving j=4 or k=4 (and not in the above blocks)
    j=4: i in {0,1}, k in {0,1,2,3,4} -> 10 terms
    k=4, j in {0,1,2,3}: i in {0,1}, j in {0,1,2,3}, k=4 -> 8 terms
    But we double counted j=4,k=4: 2 terms
    Total remaining: 10 + 8 - 2 = 16 terms
    
    Total rank: 4*7 + 16 = 28 + 16 = 44? That seems too good.
    
    Wait, I need to be more careful. The Strassen decomposition of a 2x2x2 tensor
    uses 7 rank-1 terms, but these terms are NOT simply the 8 schoolbook terms minus 1.
    They are specific linear combinations.
    
    Let me just implement the Strassen 2x2x2 decomposition explicitly.
    
    For a 2x2x2 tensor T where T_{ijk} = delta (all 1s for the identity tensor),
    Strassen gives:
    M1 = (a11+a22)(b11+b22)  -> contributes to c11, c22
    M2 = (a21+a22)*b11        -> contributes to c21, c22  
    M3 = a11*(b21-b22)        -> contributes to c12, c22
    M4 = a22*(b11-b12)        -> contributes to c11, c12
    M5 = (a11+a12)*b22        -> contributes to c11, c12
    M6 = (a21-a11)*(b11+b12)  -> contributes to c21
    M7 = (a12-a22)*(b21+b22)  -> contributes to c12
    
    Where a = A block (2x2), b = B block (2x2), c = C block (2x2).
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5 - M7
    c21 = M2 - M6
    c22 = M1 - M2 + M3 + M6
    
    Now I need to express this in terms of the factor matrices U, V, W.
    
    For the full 2x5x5 problem, let me index:
    A[i,j] for i in {0,1}, j in {0,1,2,3,4}
    B[j,k] for j in {0,1,2,3,4}, k in {0,1,2,3,4}
    C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    
    U has shape (10, rank), V has shape (25, rank), W has shape (10, rank)
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    For block 1: i={0,1}, j={0,1}, k={0,1}
    The 8 schoolbook terms are:
    (i=0,j=0,k=0), (i=0,j=0,k=1), (i=0,j=1,k=0), (i=0,j=1,k=1),
    (i=1,j=0,k=0), (i=1,j=0,k=1), (i=1,j=1,k=0), (i=1,j=1,k=1)
    
    Strassen replaces these 8 with 7 terms.
    """
    n, m, p = 2, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 10, 25, 10
    
    # We'll build the decomposition term by term
    U = [[0]*50 for _ in range(nm)]
    V = [[0]*50 for _ in range(mp)]
    W = [[0]*50 for _ in range(pn)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition for a sub-block
    # i_indices: the two i values
    # j_indices: the two j values  
    # k_indices: the two k values
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # Contributes to C[i0,k0], C[i1,k1]
        U[i0*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # Contributes to C[i1,k0], C[i1,k1]
        U[i1*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # Contributes to C[i0,k1], C[i1,k1]
        U[i0*m+j0][col] = 1
        V[j1*p+k0][col] = 1
        V[j1*p+k1][col] = -1
        W[k1*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        
        # M4 = A[i1,j1] * (B[j0,k0]-B[j0,k1])
        # Contributes to C[i0,k0], C[i0,k1]
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        V[j0*p+k1][col] = -1
        W[k0*n+i0][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # Contributes to C[i0,k0], C[i0,k1]
        U[i0*m+j0][col] = 1
        U[i0*m+j1][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # Contributes to C[i1,k0]
        U[i1*m+j0][col] = 1
        U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1
        V[j0*p+k1][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # Contributes to C[i0,k1]
        U[i0*m+j1][col] = 1
        U[i1*m+j1][col] = -1
        V[j1*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        
        return col
    
    # Add 4 Strassen blocks
    add_strassen_block(0, 1, 0, 1, 0, 1)  # Block 1
    add_strassen_block(0, 1, 0, 1, 2, 3)  # Block 2
    add_strassen_block(0, 1, 2, 3, 0, 1)  # Block 3
    add_strassen_block(0, 1, 2, 3, 2, 3)  # Block 4
    
    # Now add schoolbook terms for remaining (i,j,k) triples
    # Remaining: any triple where j=4 or k=4 (not already covered)
    covered = set()
    for i in range(2):
        for j in [0, 1, 2, 3]:
            for k in [0, 1, 2, 3]:
                covered.add((i, j, k))
    
    for i in range(2):
        for j in range(5):
            for k in range(5):
                if (i, j, k) not in covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    # Trim to actual column count
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W, "rank": col}


def _verify_exact(U, V, W, n, m, p, rng, num_trials=16):
    """Check the tensor identity exactly on random integer bases."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    for _ in range(num_trials):
        A = rng.integers(-4, 5, size=(n, m))
        B = rng.integers(-4, 5, size=(m, p))
        C_true = A @ B
        C_rec = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            u_r = Ua[:, r].reshape(n, m)
            v_r = Va[:, r].reshape(m, p)
            w_r = Wa[:, r].reshape(p, n)
            a_part = u_r * A
            b_part = v_r * B
            M = a_part @ b_part
            C_rec += M * w_r.T
        if not np.allclose(C_true, C_rec, atol=1e-9):
            return False
    return True


def _verify_exact_fast(U, V, W, n, m, p, rng, num_trials=32):
    """Faster verification using vectorized operations."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    for _ in range(num_trials):
        A = rng.integers(-4, 5, size=(n, m))
        B = rng.integers(-4, 5, size=(m, p))
        C_true = A @ B
        
        # Vectorized: for each rank term r, compute (U[:,r]*A) @ (V[:,r]*B) * W[:,r]^T
        # U[:,r] reshaped to (n,m), element-wise with A -> (n,m)
        # V[:,r] reshaped to (m,p), element-wise with B -> (m,p)
        # Product is (n,p), scaled by W[:,r] reshaped to (p,n) transposed -> (n,p)
        
        C_rec = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            u_r = Ua[:, r].reshape(n, m)
            v_r = Va[:, r].reshape(m, p)
            w_r = Wa[:, r].reshape(p, n)
            C_rec += (u_r * A) @ (v_r * B) * w_r.T
        
        if not np.allclose(C_true, C_rec, atol=1e-9):
            return False
    return True


def _try_local_search(n, m, p, target_rank, max_iters=2000, rng=None):
    """Attempt to find a rank=target_rank decomposition via local search."""
    if rng is None:
        rng = random.Random(42)
    
    # Start with a random initialization using values in {-1, 0, 1}
    U = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
    V = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
    W = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
    
    def _error(U, V, W, A, B):
        """Compute the error of the decomposition for given A, B."""
        C_true = A @ B
        C_rec = np.zeros((n, p), dtype=np.float64)
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for r in range(target_rank):
            u_r = Ua[:, r].reshape(n, m)
            v_r = Va[:, r].reshape(m, p)
            w_r = Wa[:, r].reshape(p, n)
            a_part = u_r * A
            b_part = v_r * B
            M = a_part @ b_part
            C_rec += M * w_r.T
        return np.sum((C_true - C_rec) ** 2)
    
    # Test on a fixed set of random matrices
    test_rng = np.random.default_rng(12345)
    test_mats = []
    for _ in range(5):
        A = test_rng.integers(-3, 4, size=(n, m))
        B = test_rng.integers(-3, 4, size=(m, p))
        test_mats.append((A, B))
    
    def _total_error(U, V, W):
        return sum(_error(U, V, W, A, B) for A, B in test_mats)
    
    current_U, current_V, current_W = U, V, W
    current_err = _total_error(current_U, current_V, current_W)
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_err = current_err
    
    values = [-1, -0.5, 0, 0.5, 1]
    
    for it in range(max_iters):
        # Random mutation: change one entry in one of the matrices
        mat_idx = rng.choice([0, 1, 2])
        if mat_idx == 0:
            row = rng.randrange(n*m)
            col = rng.randrange(target_rank)
            old = U[row][col]
            new = rng.choice([v for v in values if v != old])
            U[row][col] = new
            err = _total_error(U, V, W)
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                U[row][col] = old
        elif mat_idx == 1:
            row = rng.randrange(m*p)
            col = rng.randrange(target_rank)
            old = V[row][col]
            new = rng.choice([v for v in values if v != old])
            V[row][col] = new
            err = _total_error(U, V, W)
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                V[row][col] = old
        else:
            row = rng.randrange(p*n)
            col = rng.randrange(target_rank)
            old = W[row][col]
            new = rng.choice([v for v in values if v != old])
            W[row][col] = new
            err = _total_error(U, V, W)
            if err < current_err:
                current_err = err
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                W[row][col] = old
    
    if best_err < 1e-6:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 100.0  # safety margin before 150s limit
    
    rng = np.random.default_rng(2024)
    
    # Baseline: guaranteed-exact schoolbook decomposition (rank 50).
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured Strassen-based decomposition
    strassen_result = _strassen_2x5x5()
    strassen_rank = strassen_result["rank"]
    
    if _verify_exact(strassen_result["U"], strassen_result["V"], strassen_result["W"],
                     n, m, p, rng, num_trials=32):
        result = {"U": strassen_result["U"], "V": strassen_result["V"], "W": strassen_result["W"]}
        best_rank = strassen_rank
    else:
        # Strassen verification failed, fall back to schoolbook
        result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Try local search for even lower rank if time permits
    if time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(best_rank - 10, 20), -1):
            if time.time() > deadline - 15:
                break
            
            candidate = _try_local_search(n, m, p, target_rank, 
                                          max_iters=1000, 
                                          rng=random.Random(42 + target_rank))
            if candidate is not None:
                if _verify_exact(candidate["U"], candidate["V"], candidate["W"], 
                                n, m, p, rng, num_trials=32):
                    best_rank = target_rank
                    result = candidate
                    break
    
    # Final verification
    if not _verify_exact(result["U"], result["V"], result["W"], n, m, p, rng, num_trials=16):
        # Fall back to schoolbook if verification fails
        result = _schoolbook(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
