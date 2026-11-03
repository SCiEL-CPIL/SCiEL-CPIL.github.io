"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _build_structured_decomposition(n, m, p):
    """
    Build a structured decomposition for 3x4x5 using algebraic manipulation.
    
    Strategy: Split middle dimension 4 into two groups of 2.
    For each group, use a rank-3 decomposition (Strassen-like) for the 2-term dot product,
    but share the linear combinations across all (i,k) pairs.
    
    C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
           = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - A[i,1]*B[0,k] - A[i,0]*B[1,k]
             + (A[i,2]+A[i,3])*(B[2,k]+B[3,k]) - A[i,3]*B[2,k] - A[i,2]*B[3,k]
    
    This uses 6 linear combinations per (i,k) pair, but the A-side and B-side combinations
    are shared. However, in the tensor decomposition framework, each term c has fixed
    U[i*m+j,c], V[j*p+k,c], W[k*n+i,c] values.
    
    Let me construct it explicitly:
    For group 1 (j=0,1):
      Term 1: U[i*4+0,c]=1, U[i*4+1,c]=1; V[0*5+k,c]=1, V[1*5+k,c]=1; W[k*3+i,c]=1
      Term 2: U[i*4+1,c]=1; V[0*5+k,c]=1; W[k*3+i,c]=-1
      Term 3: U[i*4+0,c]=1; V[1*5+k,c]=1; W[k*3+i,c]=-1
    
    For group 2 (j=2,3):
      Term 4: U[i*4+2,c]=1, U[i*4+3,c]=1; V[2*5+k,c]=1, V[3*5+k,c]=1; W[k*3+i,c]=1
      Term 5: U[i*4+3,c]=1; V[2*5+k,c]=1; W[k*3+i,c]=-1
      Term 6: U[i*4+2,c]=1; V[3*5+k,c]=1; W[k*3+i,c]=-1
    
    Total rank = 6, which is much less than 60!
    But wait - this doesn't work because the same rank index c must work for ALL (i,k) simultaneously.
    The issue is that in a tensor decomposition, each "term" c defines a trilinear form that
    contributes to ALL (i,j,k) entries simultaneously.
    
    Let me reconsider. The correct construction:
    Each term c defines scalars u_{ij}^c, v_{jk}^c, w_{ki}^c such that
    sum_c u_{ij}^c * v_{jk}^c * w_{ki}^c = delta_{i,k} (for the multiplication tensor)
    
    Actually for matrix multiplication: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    (where the "1" represents the multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k])
    
    Wait, the tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all i,j,k when we think of it as
    a 3-tensor. The decomposition requires: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    
    So I need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    
    This is a system of n*m*p = 60 equations in 3*rank*(n*m + m*p + p*n) unknowns.
    For rank 6: 3*6*(12+20+15) = 3*6*47 = 846 unknowns, 60 equations. Should be solvable.
    
    Let me try rank 6 with a structured approach.
    """
    if n != 3 or m != 4 or p != 5:
        return None
    
    # Try rank 6 first
    for target_rank in [6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 24, 30, 36, 40, 45, 48, 50, 52, 54, 55, 56, 57, 58, 59]:
        result = _solve_for_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    return None

def _solve_for_rank(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition by solving the system.
    We use a structured initialization and then try to solve.
    """
    import numpy as np
    
    # Number of equations: n*m*p
    num_eq = n * m * p
    # Number of unknowns per term: n*m + m*p + p*n
    num_unknowns_per_term = n * m + m * p + p * n
    total_unknowns = target_rank * num_unknowns_per_term
    
    # Build the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a multilinear system, not linear. We need a different approach.
    
    # Alternative: fix two of the three matrices and solve for the third.
    # Or use a greedy/iterative approach.
    
    # Let's try: fix U and V randomly (or structurally), then solve for W.
    # For each k,i: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
    # This is: for each (k,i), we have m equations in target_rank unknowns (W[k*n+i, c] for c in range(target_rank))
    # But the coefficients depend on j through U[i*m+j,c]*V[j*p+k,c]
    
    # Actually, for fixed (i,k): sum_c (sum_j' U[i*m+j',c]*V[j'*p+k,c]) * W[k*n+i,c] ... no, that's not right either.
    
    # The equation is: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For fixed (i,k): we have m equations (one for each j) in target_rank unknowns (W[k*n+i,c])
    # Coefficient of W[k*n+i,c] in equation (i,j,k) is U[i*m+j,c]*V[j*p+k,c]
    
    # So for each (i,k), we have a system: A_{ik} * w_{ik} = 1_vector (length m)
    # where A_{ik}[j,c] = U[i*m+j,c]*V[j*p+k,c]
    
    # If we fix U and V, we can solve for W.
    
    # Let's try with a structured U and V.
    
    # Structured approach: use the "outer product" style but with shared combinations.
    # Initialize U and V with small integers, then solve for W.
    
    # Try several random initializations
    for attempt in range(50):
        # Initialize U and V
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        
        # Use a structured pattern: for each column c, assign values based on c
        for c in range(target_rank):
            # Assign U values
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = (c * 7 + i * 3 + j * 5) % 3 - 1  # values in {-1, 0, 1}
            
            # Assign V values
            for j in range(m):
                for k in range(p):
                    V[j * p + k][c] = (c * 11 + j * 2 + k * 7) % 3 - 1
        
        # Now solve for W: for each (i,k), solve A_{ik} * w = ones(m)
        W = [[0] * target_rank for _ in range(p * n)]
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build matrix A of shape (m, target_rank)
                A = []
                for j in range(m):
                    row = []
                    for c in range(target_rank):
                        row.append(U[i * m + j][c] * V[j * p + k][c])
                    A.append(row)
                
                # Solve A * w = ones(m) using least squares or exact
                if target_rank >= m:
                    # Use numpy least squares
                    try:
                        A_np = np.array(A, dtype=float)
                        b = np.ones(m)
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_np, b, rcond=None)
                        # Check if the solution is exact
                        if np.allclose(A_np @ w, b, atol=1e-10):
                            # Round to nearest integer if close
                            w_rounded = np.round(w)
                            if np.allclose(A_np @ w_rounded, b, atol=1e-10):
                                for c in range(target_rank):
                                    W[k * n + i][c] = int(w_rounded[c])
                            else:
                                # Keep fractional values
                                for c in range(target_rank):
                                    W[k * n + i][c] = float(w[c])
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                else:
                    # Underdetermined - use a simple solution
                    # Just set W[k*n+i][0] = 1 and hope it works (unlikely)
                    valid = False
                    break
                
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_direct_construction(n, m, p, deadline):
    """Try the structured decomposition approach."""
    if time.time() > deadline:
        return None
    return _build_structured_decomposition(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try structured construction
    if time.time() < deadline - 10:
        result = _try_direct_construction(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
