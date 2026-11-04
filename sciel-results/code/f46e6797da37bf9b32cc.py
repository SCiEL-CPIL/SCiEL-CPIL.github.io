"""Matrix multiplication 3x4x5 rank-minimal decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N

# Deadline guard: evaluator kills after 150s; keep 10s safety margin.
DEADLINE = 140.0

def _time_left():
    return DEADLINE - time.time()

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = 1 if all dims valid (trivially all 1).
    Actually T is a 3-way tensor of size n*m*p where T[i,j,k]=1 for all i,j,k in ranges.
    But the decomposition constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    # The target is just 1 for every (i,j,k)
    return None  # We just need sum_c U_ic * V_jc * W_kc = 1 for all (i,j,k)

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline):
    """
    Try to find a decomposition of given rank using random initialization + gradient-free search.
    For small ranks, we can use a direct algebraic approach.
    
    For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    This is a system of n*m*p = 60 equations in 3*r*(n*m + m*p + p*n) unknowns... no wait,
    it's 3*r*(nm+mp+pn) variables? No: r columns each with (nm + mp + pn) entries, so total variables = r*(nm+mp+pn).
    Equations: n*m*p = 60.
    
    For rank 1: 1*(12+20+15)=47 vars, 60 eqs -> overdetermined, unlikely.
    For rank 2: 2*47=94 vars, 60 eqs -> underdetermined, possible.
    
    Actually the minimum rank for matrix multiplication is known to be bounded. For 3x4x5,
    the schoolbook is 60. The lower bound is max(n*m, m*p, p*n) = max(12,20,15)=20.
    Known results: rank(3x4x5) is between 20 and 60. 
    
    Let me try a smarter approach: use the fact that for small matrices, we can set up
    a system and solve it.
    
    Strategy: For a given rank r, initialize U, V, W randomly, then use iterative refinement.
    But gradient descent on this non-convex problem is tricky.
    
    Alternative: For very small ranks, use exhaustive/random search with verification.
    
    Let me try rank from 20 upward, using random Gaussian initialization and then
    trying to "fix" the equations via least squares on subsets.
    """
    nm, mp, pn = n * m, m * p, p * n
    total_vars = r * (nm + mp + pn) if 'r' in dir() else rank * (nm + mp + pn)
    
    # For each rank, try multiple random initializations
    n_tries = max(10, int(60 * _time_left() / max(1, rank)))
    
    for attempt in range(n_tries):
        if _time_left() < 2:
            break
        
        # Random initialization
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Iterative refinement: for each (i,j,k), the constraint is
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # 
        # We can fix U and V, then solve for W, then fix U and W, solve for V, etc.
        # This is like alternating least squares but the equations are trilinear.
        #
        # Actually, for fixed U and V, the constraint on W is linear:
        # For each (i,j,k): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        # This is a linear system in W's entries (p*n*rank unknowns, n*m*p equations).
        
        # Let's do ALS: alternate fixing two factors and solving for the third.
        for iteration in range(50):
            if _time_left() < 2:
                break
            
            # Solve for W given U, V
            # Unknowns: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
            # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # 
            # For fixed (i,k), the equation over j is:
            # sum_j [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]] ... no, each (i,j,k) is one equation.
            # 
            # Let me reorganize. For fixed i and k, we have m equations (one for each j):
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1, for j=0..m-1
            # 
            # This is a system of m equations in (rank) unknowns (W[k*n+i,0..rank-1]).
            # Coefficient matrix A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
            # So A * W[k*n+i,:]^T = 1 (vector of ones length m)
            
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    # Solve A * x = 1
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = x
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Solve for V given U, W
            # For fixed j and k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each i
            # A_{i,c} = U[i*m+j,c] * W[k*n+i,c], solve A * V[j*p+k,:]^T = 1
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = x
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Solve for U given V, W
            # For fixed i and j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each k
            # A_{k,c} = V[j*p+k,c] * W[k*n+i,c], solve A * U[i*m+j,:]^T = 1
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = x
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_structural_rank(rank, n, m, p):
    """
    Try structural decompositions. For matrix multiplication, there are known families.
    
    One approach: use the fact that matrix multiplication can be decomposed as
    (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    A rank-r decomposition corresponds to r "gates" each contributing a rank-1-like term.
    
    For the specific case 3x4x5, let me try to use known small-rank constructions.
    The lower bound is max(nm, mp, pn) = 20.
    
    Let me try a different approach: parameterize U, V, W with structure.
    
    Actually, let me just try random search more aggressively for low ranks.
    """
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from schoolbook rank and try to improve
    # Schoolbook rank = n*m*p = 60
    best_result = None
    best_rank = n * m * p
    
    # Generate schoolbook solution as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank + 1):
        if _time_left() < 5:
            break
        
        result = _try_rank(rank, n, m, p, DEADLINE)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found the minimum rank, stop
    
    return best_result
