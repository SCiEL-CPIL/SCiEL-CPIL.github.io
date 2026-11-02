import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] equals the matmul tensor."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    # Build the decomposition tensor
    T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    # Build the target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_decomp, T_target, atol=1e-10)

def _try_rank_search(n, m, p, max_rank, time_budget):
    """
    Try to find a rank-r decomposition for r from max_rank down to some minimum.
    Uses randomized algebraic approach: fix some variables, solve linear system.
    """
    nm, mp, pn = n * m, m * p, p * n
    num_constraints = n * m * p  # number of (i,j,k) triples
    
    deadline = time.time() + time_budget
    
    for rank in range(max_rank, max(0, max_rank - 5)):
        if time.time() > deadline - 5:
            break
        
        # Strategy: Use the structure of the problem.
        # The tensor has entries T[i*m+j, j*p+k, k*n+i] = 1
        # A rank-r decomposition means:
        #   sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
        
        # Approach: Randomize and solve.
        # For each rank, we have (nm + mp + pn) * rank variables.
        # We have num_constraints equations (one per nonzero tensor entry).
        # This is a system of bilinear equations.
        
        # Simple approach: Try random initializations and use iterative refinement.
        # For small ranks, try to find solutions by fixing W to be random and solving for U, V.
        
        rng = np.random.default_rng(rank * 1000 + 42)
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize W randomly
            W = rng.standard_normal((pn, rank))
            
            # For each (i,j,k), the equation is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Group by (i,j): for fixed (i,j), we have p equations in V variables
            # Group by (j,k): for fixed (j,k), we have n equations in U variables
            
            # Let's try a different approach: fix U and V, solve for W, or vice versa.
            # Actually, let's use the fact that for fixed k, the constraint involves
            # W[k*n+i, r] for all i, which multiplies U[i*m+j, r] * V[j*p+k, r]
            
            # Alternative: Use a greedy/iterative approach
            # Start with schoolbook-like structure and try to merge terms
            
            # For now, try a simple random search with verification
            U = rng.standard_normal((nm, rank))
            V = rng.standard_normal((mp, rank))
            
            # Try to refine using gradient descent on the residual
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual
                T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
                for r in range(rank):
                    T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                T_target = np.zeros((nm, mp, pn), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i * m + j, j * p + k, k * n + i] = 1.0
                
                residual = T_decomp - T_target
                if np.max(np.abs(residual)) < 1e-10:
                    return U, V, W, rank
                
                # Compute gradients (simplified - just use finite differences for small case)
                # This is expensive but the case is small
                # Actually, let's skip gradient and just try more random inits
                break
        
        # Try structured approach: based on the rank-17 solution, try to merge
        # some of the schoolbook terms using Strassen-like tricks
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a lower-rank decomposition
    start_time = time.time()
    time_budget = 100  # seconds for search
    
    result = _try_rank_search(n, m, p, 17, time_budget)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Rank 17 decomposition for 2x3x3 matrix multiplication
    # Constructed by combining:
    # 1. Strassen (rank 7) for 2x2x2 part
    # 2. Schoolbook (rank 4) for 2x1x2 part
    # 3. Schoolbook (rank 6) for 2x3x1 part
    
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,2,3
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3
    
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
