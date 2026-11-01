"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition
    # The matrix multiplication tensor M_{ijk} = A_{ij} * B_{jk} where A is 2x2, B is 2x15
    # Tensor indices: i (row of A, 0..1), j (col of A / row of B, 0..1), k (col of B, 0..14)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]  (for the multiplication)
    # Actually the convention is:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # And W defines the output: W[k*n+i, c] such that the product equals the matrix multiplication tensor.
    # The tensor T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the "multiplication" tensor where each term is a product)
    # Actually, the tensor is: T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in the output)
    # The output is C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,15)
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k]
    # With W defining how the rank-1 factors combine to give the output.
    # 
    # Standard tensor decomposition for matrix multiplication:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the product is the "trilinear form"
    # 
    # Wait, let me re-read the contract. The factor matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # and they "exactly decompose the matrix multiplication tensor".
    # The standard formulation: C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 (coefficient of A[i,j] * B[j,k])
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. Let me think again.
    #
    # The matrix multiplication tensor is T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # Actually, the standard way: T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    # But the contract says W has shape (p*n, rank), so the third index is k*n+i.
    #
    # Let me use the standard formulation:
    # T[i,j,k] = 1 for all valid indices
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... no.
    #
    # Actually, in the standard tensor decomposition for matmul:
    # T ∈ R^{nm} ⊗ R^{mp} ⊗ R^{np} (or R^{pn})
    # T[(i,j), (j,k), (i,k)] = 1
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i (following the W[k*n+i,r] convention)
    #
    # So we need: for all i,j,k: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # This is a bilinear/trilinear system. Let me set up the equations.
    # Variables: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for c = 0..R-1
    # Equations: for each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a nonlinear system (trilinear). Hard to solve directly.
    #
    # Alternative approach: Use the fact that for 2x2x15, we can try to find a decomposition
    # by fixing some structure. 
    #
    # Let's try a different approach: use the known result that the rank of 2x2xp matmul is at most 2p+2
    # for p >= 2 (from literature on matrix multiplication tensor ranks).
    # For 2x2x15: rank <= 2*15+2 = 32
    #
    # Let me try to construct a rank-32 decomposition.
    #
    # One known construction: 
    # For 2x2xp, we can use the following:
    # Let the rank be 2p+2 = 32.
    # 
    # Actually, let me try a simpler approach: use a solver to find a low-rank decomposition.
    #
    # Since the system is trilinear, I'll use a heuristic:
    # 1. Fix W to be a simple pattern.
    # 2. Solve for U and V.
    #
    # Or use alternating least squares.
    
    def try_decomposition(R, time_limit=10):
        """Try to find a decomposition of rank R."""
        t_start = time.time()
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale
        U /= np.sqrt(R)
        V /= np.sqrt(R)
        W /= np.sqrt(R)
        
        # Alternating minimization: fix two, solve for one
        # This is non-convex, so we try multiple restarts
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        max_iters = 200
        for it in range(max_iters):
            if time.time() - t_start > time_limit:
                break
            
            # Update U: for each a = i*m+j, solve min sum_k W[k*n+i, c] * V[j*p+k, c] * U[a,c] = 1
            # This is still nonlinear. Let's use a simpler approach.
            
            # Actually, let's use a different strategy: 
            # For each c, the rank-1 term is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
            # We need sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
            #
            # Let's try: fix V and W, solve for U via linear least squares.
            # For each a: sum_c U[a,c] * (V[:,c] ⊙ W_mapped[:,c])[a-related] ... this is complex.
            
            # Simpler: use gradient-based optimization
            pass
        
        # Let's try a direct construction for 2x2xp
        # Known result: rank(2x2xp) = 2p+2 for p >= 2 (I believe this is correct)
        # Construction:
        # 
        # Let's use the following scheme for 2x2xp with rank 2p+2:
        # U has 4 rows (i=0,1; j=0,1), V has 2p rows (j=0,1; k=0..p-1), W has 2p columns (k=0..p-1; i=0,1)
        #
        # I'll look for a known construction or use a solver.
        
        # Try to solve using numpy's least squares on a linearized version
        # Fix W to be identity-like, then solve for U and V.
        
        # Simplest fallback: just return None if we can't find it quickly
        return None
    
    # Try a few ranks from low to high
    for R in range(2 * p + 2, rank_schoolbook):  # Start from 32
        if time.time() > deadline - 20:
            break
        
        # Try to find a decomposition of rank R
        # Use a simple heuristic: random initialization + projection
        
        np.random.seed(123 + R)
        
        # For 2x2xp, try the following construction:
        # Rank 2p+2 = 32
        # 
        # Let me try to construct it explicitly.
        # 
        # Idea: Use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
        # For 2x2: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        #           C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        #
        # We can use rank-1 terms to share computation across k.
        #
        # Let's define 32 rank-1 terms:
        # Terms 0..14: for k=0..14, handle the "main" part
        # Terms 15..29: additional terms for the 2x2 structure
        # Terms 30..31: two more terms
        #
        # Actually, let me try a different approach: use the solver with a time limit.
        
        # For now, just return the schoolbook if we can't find better quickly
        pass
    
    # If we're running out of time, return the best we have
    if time.time() > deadline - 5:
        return best_result
    
    # Try one more thing: see if rank 32 works with a specific construction
    R = 32
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Construction attempt for 2x2xp:
    # Use 2p terms for the "diagonal" part and 2 terms for the "off-diagonal" part
    # 
    # Term c for c=0..p-1:
    #   U[0, c] = 1, U[1, c] = 1, U[2, c] = 0, U[3, c] = 0  (i=0,j=0 and i=0,j=1)
    #   V[0, c] = 1, V[p+c, c] = 1, others 0  (j=0,k=c and j=1,k=c)  
    #   W[c*2+0, c] = 1, W[c*2+1, c] = 1, others 0  (k=c,i=0 and k=c,i=1)
    # This gives: for (i=0,j=0,k=c): U[0,c]*V[0,c]*W[c*2+0,c] = 1*1*1 = 1 ✓
    #           for (i=0,j=1,k=c): U[1,c]*V[p+c,c]*W[c*2+1,c] = 1*1*1 = 1 ✓
    #           for (i=1,j=0,k=c): U[2,c]*V[0,c]*W[c*2+0,c] = 0*1*1 = 0 ✗ (should be 1)
    
    # That doesn't work. Let me try a different construction.
    
    # I'll just return the schoolbook decomposition since I can't quickly find a better one.
    # But let me at least try rank 40, 50 to see if a random search works.
    
    for R in [40, 50]:
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + R)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)  
        W = np.random.randn(p * n, R)
        
        # Check if this gives the right tensor (it won't, but let's verify the structure)
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        # This is a sanity check that will fail, but shows the structure.
        
        # Since random won't work, just continue to fallback
    
    return best_result
# EVOLVE-BLOCK-END
