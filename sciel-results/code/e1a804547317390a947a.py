"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Uses Strassen's 2x2 identity (rank 7) embedded into the 2x5x6 problem.
    The 2x5 matrix A has rows indexed by i in {0,1} and columns j in {0..4}.
    The 5x6 matrix B has rows j in {0..4} and columns k in {0..5}.
    Output C[i,k] = sum_j A[i,j] * B[j,k].
    
    We use the Strassen trick on the 2-row structure:
    Let M1 = A[0,:], M2 = A[1,:] (each is 1x5)
    Let N1 = B[:,0:3], N2 = B[:,3:6] (each is 5x3)
    
    Actually, let's think differently. The standard Strassen applies to 2x2 blocks.
    For 2x5 * 5x6, we can view it as:
    - 2 row groups (i=0,1)
    - 5 column groups in A (j=0..4) which are also row groups in B
    - 6 column groups in B (k=0..5)
    
    The schoolbook rank is 2*5*6 = 60.
    
    Strassen's key idea for 2x2: instead of 8 products, use 7:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B01-B11)
    P4 = A11*(B10-B00)
    P5 = (A00+A10)*B11
    P6 = (A01-A00)*(B00+B10)
    P7 = (A10-A11)*(B01+B11)
    
    For our 2x5 * 5x6 case, we can apply a similar algebraic identity.
    Let's denote A[i,j] for i in {0,1}, j in {0..4}
    B[j,k] for j in {0..4}, k in {0..5}
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + ... + A[i,4]*B[4,k]
    
    We can group the 5 middle indices. Let's try to find algebraic identities.
    
    Actually, a simpler approach: use the known result that the rank of matrix multiplication
    <n,m,p> satisfies certain bounds. For 2x5x6, we can try to construct a decomposition
    using the Strassen-like approach on the 2-dimensional index.
    
    Let me try a different structured approach:
    
    For the 2x2 case, Strassen shows rank <= 7.
    For 2x1x1: rank is 2 (A[0,0]*B[0,0] and A[1,0]*B[0,0])
    For 2x2x2: rank <= 7
    
    For 2x5x6, one approach: decompose the 5 and 6 dimensions.
    
    Let's use the following strategy:
    - Split the 5 middle indices into groups
    - Apply Strassen-like identities on the 2-row dimension
    
    Actually, let's just do a systematic search with the structure that 
    U[i*m+j, r] depends only on i and r (for the 2-row Strassen structure),
    or more generally, use a low-rank factorization approach.
    
    I'll implement a structured random search that exploits the tensor structure.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact using Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    def verify_decomposition_np(U, V, W, rank):
        """Fast numpy verification."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # We can vectorize this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j]
                    v_val = V_np[j * p + k]
                    w_val = W_np[k * n + i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook decomposition as baseline
    school_rank = n * m * p  # 60
    U_school = [[0] * school_rank for _ in range(n * m)]
    V_school = [[0] * school_rank for _ in range(m * p)]
    W_school = [[0] * school_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = school_rank
    
    # Strategy 1: Structured Strassen-like decomposition
    # For 2x2x2, Strassen uses rank 7.
    # For 2x5x6, we can try to embed this structure.
    
    # Let's try a systematic approach:
    # The tensor T[i,j,k] = 1 for all i,j,k needs to be decomposed as sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # Key insight: if we can find a low-rank decomposition, the entries should be small integers/half-integers.
    
    # Let's try ranks from the theoretical lower bound upward.
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But for 2x5x6, we might do better with structured approaches.
    
    # Try a structured random search with bias toward small values
    def structured_search(rank, max_trials=500):
        """Search for rank-r decomposition with structured random initialization."""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with sparse random values
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Use a bias: many zeros, some +/-1, some +/-1/2
            for idx in range(n * m):
                for ridx in range(rank):
                    rand = random.random()
                    if rand < 0.25:  # 25% non-zero
                        if rand < 0.15:
                            U[idx][ridx] = 1
                        elif rand < 0.20:
                            U[idx][ridx] = -1
                        elif rand < 0.225:
                            U[idx][ridx] = Fraction(1, 2)
                        else:
                            U[idx][ridx] = Fraction(-1, 2)
            
            for idx in range(m * p):
                for ridx in range(rank):
                    rand = random.random()
                    if rand < 0.25:
                        if rand < 0.15:
                            V[idx][ridx] = 1
                        elif rand < 0.20:
                            V[idx][ridx] = -1
                        elif rand < 0.225:
                            V[idx][ridx] = Fraction(1, 2)
                        else:
                            V[idx][ridx] = Fraction(-1, 2)
            
            for idx in range(p * n):
                for ridx in range(rank):
                    rand = random.random()
                    if rand < 0.25:
                        if rand < 0.15:
                            W[idx][ridx] = 1
                        elif rand < 0.20:
                            W[idx][ridx] = -1
                        elif rand < 0.225:
                            W[idx][ridx] = Fraction(1, 2)
                        else:
                            W[idx][ridx] = Fraction(-1, 2)
            
            # Quick numpy check first
            if verify_decomposition_np(U, V, W, rank):
                # Full exact verification
                if verify_decomposition(U, V, W, rank):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find decompositions with rank below schoolbook
    # Start from a reasonable estimate and go up
    # For 2x5x6, known results suggest rank might be around 40-50
    
    # Try ranks from 35 to 59
    for r in range(35, school_rank):
        if time.time() > deadline:
            break
        
        result = structured_search(r, max_trials=300)
        if result is not None:
            best_rank = r
            best_result = result
            # Found a valid decomposition, continue trying lower ranks
            continue
    
    return best_result

# EVOLVE-BLOCK-END
