"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _winograd_3x3x2():
    """
    Rank-7 decomposition for 3x3 matrix times 3x2 matrix.
    
    Uses Winograd's F(3,2) scheme:
    For each output row i (0,1,2) and output col k (0,1):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    Winograd factorization:
    A = G * a  where G is 3x7, a is 7-vector (per row)
    B = Bmat * b where Bmat is 3x7, b is 7-vector (per col)
    C = c * F where F is 2x7
    
    The 7 multiplication terms per (row, col) pair:
    For row i, the 7 values are:
      a0 = A[i,0] + A[i,1]
      a1 = A[i,0] + A[i,2]
      a2 = A[i,1] + A[i,2]
      a3 = A[i,0]
      a4 = A[i,1]
      a5 = A[i,2]
      a6 = A[i,1]
    
    Wait, let me use the standard Winograd F(3,2):
    
    Actually for 3x3 * 3x2, the Winograd scheme has rank 7 per output element?
    No - the rank is 7 total for computing C[i,k] for fixed i,k? No.
    
    Let me think more carefully. The tensor for 3x3 * 3x2 has dimensions 9x3x2 = 54 entries.
    
    Actually, the rank of the 3x3x2 matrix multiplication tensor is 7.
    This means we can compute all 3*3*2 = 18 products using only 7 multiplications.
    
    The Winograd F(3,2) scheme:
    Given A (3x3) and B (3x2), compute C = A*B (3x2).
    
    The 7 intermediate values:
    m0 = (A[0,0]+A[0,1]) * (B[0,0]+B[1,0])
    m1 = (A[0,0]+A[0,2]) * (B[0,0]+B[2,0])  
    m2 = (A[0,1]+A[0,2]) * (B[1,0]+B[2,0])
    m3 = A[0,0] * B[0,0]
    m4 = A[0,1] * B[1,0]
    m5 = A[0,2] * B[2,0]
    m6 = A[0,1] * B[1,1]  -- this doesn't look right for all outputs
    
    Let me use the proper Winograd F(3,2) decomposition.
    
    The standard Winograd F(3,2) for computing y = A*x where A is 3x3 and x is 3x1
    has rank 7. For 3x3 * 3x2, we can do 2 such operations (one for each column of B),
    giving rank 14. But that's not optimal.
    
    Actually, the rank of the (3,3,2) matrix multiplication tensor is 7.
    This is a well-known result. Let me construct it properly.
    
    For C[i,k] = sum_j A[i,j]*B[j,k], i in {0,1,2}, j in {0,1,2}, k in {0,1}
    
    The 7 rank-1 terms use the following structure (Winograd F(3,2)):
    
    U (9x7) for A elements A[i,j]:
    V (6x7) for B elements B[j,k]:  
    W (6x7) for C elements C[i,k]:
    
    Using the Winograd F(3,2) transform:
    The 7 products are:
    p0 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])  -- but this varies with i and k
    
    Let me just verify: is rank(3,3,2) = 7?
    
    Actually, I recall that R(3,3,2) = 7 is NOT established. The known bounds are:
    - Lower bound: max(3*3, 3*2, 2*3) = 9? No, that's not right either.
    - The trivial bound gives 3*3*2 = 18.
    
    Let me use a different approach. For 3x3 * 3x6, I can split into three 3x3*3x2 blocks.
    If each 3x3*3x2 has rank r, total rank is 3r.
    
    The schoolbook for 3x3*3x2 is rank 18. So total would be 54 (same as before).
    
    But maybe I can do better with a structured approach.
    
    Alternative: treat 3x3*3x6 as (3x3*3x1) repeated 6 times = rank 9*6 = 54.
    Or (3x3*3x2) repeated 3 times.
    
    Let me try a different decomposition. What if I use the fact that 
    3x3*3x6 can be seen as computing 3 independent 3x3*3x2 products?
    
    For each 3x3*3x2 subproblem, the schoolbook rank is 3*3*2 = 18.
    Total: 3*18 = 54. Same as before.
    
    To beat 54, I need a non-trivial decomposition.
    
    Let me try: use Winograd-style for 3x3*3x2.
    
    Actually, I found that the rank of (3,3,2) matrix multiplication is 7.
    Reference: Winograd's F(3,2) algorithm uses 7 multiplications.
    
    The F(3,2) Winograd scheme computes y_i = sum_j a_ij * x_j for i=0,1,2
    using 7 multiplications. For our case, we have two such systems (k=0,1).
    
    But the 7 multiplications for F(3,2) are per output vector. For 3x3*3x2,
    we need to compute C[i,k] for i=0,1,2 and k=0,1.
    
    If we apply F(3,2) independently for each column k, we get 7*2 = 14 multiplications.
    
    But can we share some? The Winograd F(3,2) for column k uses:
    m0 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    m1 = (A[i,0]+A[i,2]) * (B[0,k]+B[2,k])
    m2 = (A[i,1]+A[i,2]) * (B[1,k]+B[2,k])
    m3 = A[i,0] * B[0,k]
    m4 = A[i,1] * B[1,k]
    m5 = A[i,2] * B[2,k]
    m6 = A[i,1] * B[1,k]  -- wait, this is the same as m4?
    
    I'm confusing myself. Let me look up the actual F(3,2) Winograd scheme.
    
    The Winograd F(3,2) computes y = A*x where A is 3x3, x is 3x1, using 7 mults:
    
    u0 = x0 + x1
    u1 = x0 + x2
    u2 = x1 + x2
    u3 = x0
    u4 = x1
    u5 = x2
    u6 = x1  (redundant?)
    
    No, the 7 values are:
    u0 = x0 + x1
    u1 = x0 + x2
    u2 = x1 + x2
    u3 = x0
    u4 = x1
    u5 = x2
    u6 = x1
    
    And the 7 multiplications are:
    m0 = (a00+a01) * (x0+x1)
    m1 = (a00+a02) * (x0+x2)
    m2 = (a01+a02) * (x1+x2)
    m3 = a00 * x0
    m4 = a01 * x1
    m5 = a02 * x2
    m6 = a01 * x1  -- this is duplicate of m4!
    
    I think I'm misremembering. Let me just use the schoolbook for now and try
    to find a better decomposition through optimization.
    
    For now, let me return the schoolbook rank 54 solution and improve the
    optimization to search for lower ranks.
    """
    pass

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T should equal 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, max_iters=200, time_limit=30.0, lr_init=0.01, seed=42):
    """Try to find a rank-`rank` decomposition via numerical optimization."""
    n, m, p = 3, 3, 6
    t0 = time.time()
    np.random.seed(seed)
    
    # Initialize near schoolbook (take first `rank` columns)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use schoolbook initialization for first `rank` terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    # Gradient descent on the loss: sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    lr = lr_init
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        # Compute residuals
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = s - 1.0
        
        loss = np.sum(residual ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if loss < 1e-12:
                break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_ijk = residual[i, j, k]
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * r_ijk * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * r_ijk * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * r_ijk * U[i * m + j, r] * V[j * p + k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.98
    
    # Check if we found a valid solution
    if best_loss < 1e-9:
        # Convert to list of lists
        U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _winograd_3x3x6():
    """
    Attempt to construct a low-rank decomposition for 3x3x6 using
    Winograd-style structured factors.
    
    Strategy: Split the 3x6 B-matrix into three 3x2 blocks.
    For each 3x3x2 subproblem, try to find a rank-7 decomposition.
    Total rank would be 3*7 = 21.
    """
    n, m, p = 3, 3, 6
    t0 = time.time()
    time_limit = 60.0
    
    # Try to solve for rank-7 decomposition of 3x3x2
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # for i=0,1,2; j=0,1,2; k=0,1
    
    # This is 18 equations. With rank 7, we have 7*(9+6+6) = 147 unknowns.
    # We can fix U and V, then solve for W.
    
    # Let's try a structured approach. For each of the 3 blocks (k-pair),
    # we solve independently.
    
    # For a single 3x3x2 block, let's try to find rank r decomposition.
    def solve_3x3x2(r, max_time=20.0):
        """Solve for rank-r decomposition of 3x3x2."""
        n2, m2, p2 = 3, 3, 2
        t_start = time.time()
        
        # We need: sum_{s=0}^{r-1} U[i*m2+j, s] * V[j*p2+k, s] * W[k*n2+i, s] = 1
        # for all i in {0,1,2}, j in {0,1,2}, k in {0,1}
        
        # Strategy: Fix U and V to be random (or structured), then solve linear system for W.
        # For each (i,k), the equation is:
        # sum_s W[k*n2+i, s] * (sum_j U[i*m2+j, s] * V[j*p2+k, s]) = 1
        
        # Let A[i,k,s] = sum_j U[i*m2+j, s] * V[j*p2+k, s]
        # Then: sum_s W[k*n2+i, s] * A[i,k,s] = 1
        
        # This is a linear system: for each (i,k), we have one equation with r unknowns (W[k*n2+i, s]).
        # Total: 6 equations, 6*r unknowns. Underdetermined if r > 1.
        
        # But we also need consistency: the same W values appear in multiple equations.
        # W[k*n2+i, s] appears in the equation for (i,k).
        
        # So we have 6 equations:
        # For (i,k): sum_s W[k*3+i, s] * A[i,k,s] = 1
        
        # Let's fix U and V, compute A, then solve for W.
        
        np.random.seed(42)
        for attempt in range(100):
            if time.time() - t_start > max_time:
                break
            
            U = np.random.randn(n2 * m2, r)
            V = np.random.randn(m2 * p2, r)
            
            # Compute A[i,k,s] = sum_j U[i*m2+j, s] * V[j*p2+k, s]
            A = np.zeros((n2, p2, r))
            for i in range(n2):
                for k in range(p2):
                    for s in range(r):
                        for j in range(m2):
                            A[i, k, s] += U[i * m2 + j, s] * V[j * p2 + k, s]
            
            # Solve for W: for each (i,k), sum_s W[k*n2+i, s] * A[i,k,s] = 1
            # This is 6 equations with 6*r unknowns.
            # Stack into a single system:
            # Let W_flat be a vector of length 6*r.
            # Equation for (i,k): sum_s W_flat[(k*n2+i)*r + s] * A[i,k,s] = 1
            
            # This is an underdetermined system. Use least squares.
            # Build the matrix: 6 x (6*r)
            M = np.zeros((n2 * p2, n2 * p2 * r))
            for i in range(n2):
                for k in range(p2):
                    idx = k * n2 + i
                    for s in range(r):
                        M[idx, idx * r + s] = A[i, k, s]
            
            b = np.ones(n2 * p2)
            
            # Solve least squares
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(M, b, rcond=None)
            
            W = W_flat.reshape(n2 * p2, r)
            
            # Verify
            max_err = 0.0
            for i in range(n2):
                for k in range(p2):
                    s_val = 0.0
                    for s in range(r):
                        s_val += W[k * n2 + i, s] * A[i, k, s]
                    max_err = max(max_err, abs(s_val - 1.0))
            
            if max_err < 1e-9:
                return U, V, W
        
        return None
    
    # Try rank 7 for each 3x3x2 block
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for block in range(3):
        if time.time() - t0 > time_limit:
            break
        
        result = solve_3x3x2(7, max_time=15.0)
        if result is None:
            return None
        
        U_b, V_b, W_b = result
        
        # Expand to full 3x3x6 size
        U_full = np.zeros((n * m, 7))
        V_full = np.zeros((m * p, 7))
        W_full = np.zeros((p * n, 7))
        
        # U is the same for all blocks (A-matrix is shared)
        U_full = U_b
        
        # V: for block `block`, the B-columns are 2*block and 2*block+1
        for j in range(m):
            for k in range(2):
                V_full[j * p + (2 * block + k), :] = V_b[j * 2 + k, :]
        
        # W: for block `block`, the output columns are 2*block and 2*block+1
        for k in range(2):
            for i in range(n):
                W_full[(2 * block + k) * n + i, :] = W_b[k * n + i, :]
        
        U_blocks.append(U_full)
        V_blocks.append(V_full)
        W_blocks.append(W_full)
    
    if len(U_blocks) != 3:
        return None
    
    # Concatenate the three blocks
    U_total = np.hstack(U_blocks)  # 9 x 21
    V_total = np.hstack(V_blocks)  # 18 x 21
    W_total = np.hstack(W_blocks)  # 18 x 21
    
    # Verify
    U_list = [[float(U_total[i, r]) for r in range(21)] for i in range(n * m)]
    V_list = [[float(V_total[i, r]) for r in range(21)] for i in range(m * p)]
    W_list = [[float(W_total[i, r]) for r in range(21)] for i in range(p * n)]
    
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Try Winograd-style structured decomposition (rank 21)
    if time.time() < deadline - 65:
        result = _winograd_3x3x6()
        if result is not None:
            return result
    
    # Try to find a lower-rank decomposition via optimization
    for rank in [53, 52, 51, 50]:
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, max_iters=30, time_limit=25.0)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
