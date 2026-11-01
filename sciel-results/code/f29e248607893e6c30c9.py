"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import itertools

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p  # 60 equations
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Generate candidate rank-1 tensor supports
    # Each support is defined by choosing nonzero patterns for U, V, W
    # For efficiency, we'll use structured supports
    
    # Strategy: Generate candidate supports and find minimum weight set
    # that can span all 60 entries
    
    # First, let's try a greedy approach with rank reduction
    # Build the tensor as a matrix where each column is one entry (i,j,k)
    # and each row is one possible rank-1 support pattern
    
    # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's generate structured candidate supports
    # Type 1: Support on single (i,j,k) - trivial, rank = 60
    # Type 2: Support on a row of the output (fixed i, varying j,k)
    # Type 3: Support on a column of the output (fixed k, varying i,j)
    # Type 4: More complex patterns
    
    # Better approach: Use the fact that matrix multiplication can be decomposed
    # as sum of rank-1 tensors where each term multiplies a subset of entries
    
    # Let's try to find a decomposition using structured patterns
    
    # For 2x5x6, the standard decomposition has rank = 2*5*6 = 60
    # We want to find a smaller rank
    
    # Key insight: We can share computation across entries
    # For example, if we fix j, we can compute a 2x6 matrix product
    # which can be done with rank at most 2*6 = 12 (or less with Strassen-like methods)
    
    # Let's try a block decomposition approach
    # Split the middle dimension m=5 into blocks
    
    # Actually, let's use a more systematic approach:
    # Generate candidate rank-1 supports and use linear programming to find minimum weight
    
    # Step 1: Generate candidate supports
    # A rank-1 support is defined by which (i,j), (j,k), (k,i) indices are nonzero
    # For simplicity, let's use supports that are "separable" in some way
    
    # Let's try a different approach: use known results for small matrix multiplication
    # For 2x2x2, rank is 7 (Strassen)
    # For larger dimensions, we can use tensor product constructions
    
    # Let's try to construct a decomposition using Kronecker products
    
    # First, handle the 2x2 part using Strassen (rank 7)
    # Then extend to 2x5x6
    
    # Actually, let's just try random search with gradient descent more efficiently
    
    best_result = None
    best_rank = total_eq  # fallback
    
    # Try ranks from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But we can do better with shared computation
    
    for rank in range(30, 60):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Use alternating least squares
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W given U, V
                # For each (k,i): sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a linear system in W
                
                # Build the linear system for W
                # For each (k,i), we have: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
                # Wait, that's not right. Let me re-derive.
                
                # The equation is: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed (k,i), we have p equations (one for each j) and rank unknowns
                # But the equations are coupled through W[k*n+i,c]
                
                # Actually, for fixed (k,i), the unknowns are W[k*n+i,0], ..., W[k*n+i,rank-1]
                # And we have m equations (one for each j):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0..m-1
                
                # This is a linear system: A[k,i] @ W[k*n+i,:] = 1
                # where A[k,i][j,c] = U[i*m+j,c] * V[j*p+k,c]
                
                # Solve this linear system for each (k,i)
                for k in range(p):
                    for i in range(n):
                        # Build A matrix: shape (m, rank)
                        A = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        
                        # Solve A @ x = 1
                        try:
                            x, residuals, rank_sys, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            W[k*n+i, :] = np.random.randn(rank) * 0.1
                
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        # Build B matrix: shape (p, rank)
                        B = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        
                        try:
                            x, residuals, rank_sys, s = np.linalg.lstsq(B, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            U[i*m+j, :] = np.random.randn(rank) * 0.1
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        # Build C matrix: shape (n, rank)
                        C = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                C[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        
                        try:
                            x, residuals, rank_sys, s = np.linalg.lstsq(C, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            V[j*p+k, :] = np.random.randn(rank) * 0.1
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    found = True
                    break
        
        if found:
            # Verify and return
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(nm)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(mp)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = total_eq
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
