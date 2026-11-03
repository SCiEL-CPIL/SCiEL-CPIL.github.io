<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The schoolbook rank is 2*3*10 = 60.
    We attempt to find a lower rank decomposition.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    the tensor T_{i,j,k} corresponds to C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    A rank-r decomposition has:
    C_{i,k} = sum_{a=1}^r U_{i,a} * V_{j,a} * W_{k,a}  -- wait, this isn't quite right for matrix mult.
    
    Actually, the standard tensor decomposition for matrix multiplication is:
    T_{i,j,k} = sum_{a=1}^r U_{i,a} * V_{j,a} * W_{k,a}
    where the tensor T is the multiplication tensor. But for matrix multiplication,
    the structure is slightly different.
    
    Let me reconsider. The multiplication tensor for A(n x m) * B(m x p) -> C(n x p) is:
    T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}, which it always does.
    So T_{i,j,k} = 1 for all valid i,j,k? No, that's not right either.
    
    The multiplication tensor has entries T_{i,j,k} such that:
    (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The rank of this tensor is the minimum r such that we can write:
    (A*B)_{i,k} = sum_{a=1}^r U_{i,a} * (sum_j V_{j,a} * W_{j,a}) ... no.
    
    Let me use the standard formulation. A rank-r decomposition of the matrix multiplication tensor
    means we can write the multiplication as:
    C_{i,k} = sum_{a=1}^r U_{i,a} * M_a(B)
    where M_a are linear forms on B, or equivalently:
    
    C_{i,k} = sum_{a=1}^r sum_j U_{i,a} * V_{j,a} * W_{j,k,a}
    
    Hmm, let me look at the seed program's convention:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The convention in the seed is:
    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    
    And the reconstruction is:
    C_{i,k} = sum_{j} sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that gives C_{i,k} = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    For the schoolbook, each (i,j,k) gets its own column c, so:
    C_{i,k} = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
           = sum_j A_{i,j} * B_{j,k}
    
    because for column c corresponding to (i',j',k'), U[i*m+j,c] = delta_{i,i'}delta_{j,j'},
    V[j*p+k,c] = delta_{j,j'}delta_{k,k'}, W[k*n+i,c] = delta_{k,k'}delta_{i,i'}.
    
    So the sum over c picks out only the term where i=i', j=j', k=k', giving A_{i,j} * B_{j,k}.
    
    So the general rank-r decomposition satisfies:
    C_{i,k} = sum_{j=0}^{m-1} sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This must equal sum_j A_{i,j} * B_{j,k} for all A, B.
    
    This is a multilinear identity. The unknowns are U, V, W entries.
    We need: for all i, k:
    sum_j A_{i,j} * B_{j,k} = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is a polynomial identity in the variables A_{i,j} and B_{j,k}.
    By comparing coefficients of each monomial A_{i,j} * B_{j,k}, we get:
    For each (i, j, k):
    1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    And for all other monomials (which don't appear in the LHS), the coefficient must be 0.
    But since the LHS only has monomials of the form A_{i,j} * B_{j,k} (one A and one B factor),
    and the RHS is also bilinear in A and B (linear in each A_{i,j} when B is fixed, etc.),
    we just need the above equation to hold for all (i,j,k).
    
    Wait, actually the RHS is:
    sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is already summed over j. Let me rewrite:
    Let U_{i,j,c} = U[i*m+j, c], V_{j,k,c} = V[j*p+k, c], W_{k,i,c} = W[k*n+i, c].
    
    Then the condition is:
    sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c} = 1  for all i, j, k.
    
    So we need to find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of n*m*p = 60 equations in 3*r*(n*m + m*p + p*n)/... wait, the number of unknowns is
    r*(n*m + m*p + p*n) = r*(6 + 30 + 20) = r*56.
    
    For r = 60, we have 3360 unknowns and 60 equations, so many solutions exist (schoolbook is one).
    For lower r, we need to check if a solution exists.
    
    This is a system of quadratic equations (each equation is linear in each of U, V, W separately,
    but bilinear/trilinear overall). Actually, each equation is linear in the entries of U for fixed V,W,
    etc. This is a 3-linear system.
    
    Known results: For 2x3 * 3x10, the rank is not known to be less than 60 in general, but let's try
    to find a lower rank. Actually, the minimal rank for matrix multiplication is a hard open problem
    for most sizes. For 2x3 * 3x10, the best known upper bound might be 60 or slightly less.
    
    Let me try to see if rank 59 or lower is possible. Given the complexity, let me try a heuristic search
    or use the fact that we can set some entries to 0 or 1 to simplify.
    
    Actually, let me just return the schoolbook decomposition for now, as finding a lower rank is
    computationally intensive and may not be feasible within the time limit.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
=======
"""Program for matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The condition for a rank-r decomposition is:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We use a structured approach: try to find solutions with small rational/integer entries.
    For the 2x3 * 3x10 case, we attempt ranks below 60 using randomized search
    with rational reconstruction.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 55 down to 60, looking for valid decompositions
    # Start with schoolbook (rank 60) as guaranteed fallback
    for rank in range(55, 61):
        # For each rank, try multiple random initializations
        for attempt in range(200):
            result = _try_rank(rank, n, m, p, attempt)
            if result is not None:
                return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _try_rank(rank, n, m, p, seed_offset):
    """
    Try to find a rank-r decomposition using alternating least squares
    with rational entry constraint.
    """
    np.random.seed(42 + seed_offset)
    
    # Initialize with small random values
    U = np.random.uniform(-1, 1, (n * m, rank))
    V = np.random.uniform(-1, 1, (m * p, rank))
    W = np.random.uniform(-1, 1, (p * n, rank))
    
    # Alternating minimization: fix two, solve for the third
    for iteration in range(500):
        # Solve for U: for each (i,j,c), we need
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is a system: for each (i,j), we have p equations in r unknowns (U[i*m+j, :])
        # But it's nonlinear because V and W depend on c too.
        
        # Actually, let's reformulate. For fixed V and W, the equations are:
        # For each (i,j,k): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
        # Let B[i*m+j, k, c] = V[j*p+k, c] * W[k*n+i, c]
        # Then: sum_c U[i*m+j, c] * B[i*m+j, k, c] = 1 for all k
        # This is a linear system in U[i*m+j, :] for each (i,j)
        
        # Build the linear system for U
        # For each (i,j), we have p equations:
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
        # Matrix form: M[i*m+j] @ U[i*m+j, :] = ones(p)
        # where M[i*m+j, k, c] = V[j*p+k, c] * W[k*n+i, c]
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build p x rank matrix
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ U[i*m+j, :] = ones(p)
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    return None
        
        # Solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    return None
        
        # Solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, residuals, rank_val, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    return None
        
        # Check convergence
        U, V, W = U_new, V_new, W_new
        
        # Verify the solution
        if _verify(U, V, W, n, m, p, tol=1e-6):
            # Round to nearest rational (integer or half-integer)
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            if _verify(U_int, V_int, W_int, n, m, p, tol=1e-10):
                # Convert to lists
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
    
    return None

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition satisfies the condition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
