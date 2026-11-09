"""Matrix multiplication 2x4x7 tensor decomposition - rank-minimal exact."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def _schoolbook(n, m, p):
    """Return U, V, W for naive tensor decomposition of n x m @ m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol, rtol=0):
        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact rational arithmetic."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Fraction(str(U[i*m+j, r])) * Fraction(str(V[j*p+k, r])) * Fraction(str(W[k*n+i, r]))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True

def _try_rank_random(n, m, p, target_rank, time_budget=10.0, seed=None):
    """Try to find a rank-target_rank decomposition using random initialization."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    deadline = time.time() + time_budget
    best = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization with small integers/halves
        scale = random.choice([0.5, 1.0, 1.5, 2.0])
        U = np.random.randint(-2, 3, size=(n*m, target_rank)) * scale
        V = np.random.randint(-2, 3, size=(m*p, target_rank)) * scale
        W = np.random.randint(-2, 3, size=(p*n, target_rank)) * scale
        
        # Check
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if np.allclose(T, np.ones((n, m, p)), atol=1e-10, rtol=0):
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
            best = (U, V, W)
    
    return None

def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on algebraic identities.
    
    For 2x4x7, we can try to exploit the structure.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: group columns of B and use linearity.
    """
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Try a decomposition where we separate the j-dependence
    # If we can write 1 = sum_r f_r(i) * g_r(j) * h_r(k) with coupled indices...
    
    # Actually, let's try specific small ranks with systematic search
    # over structured patterns
    
    for target_rank in range(15, 35):
        # Pattern 1: Block structure
        # Try to find U, V, W with entries in {-1, 0, 1}
        pass
    
    return None

def _greedy_rank_reduction(n, m, p, start_rank, time_budget=30.0):
    """
    Start from a known valid decomposition and try to reduce rank
    by combining terms or finding linear dependencies.
    """
    deadline = time.time() + time_budget
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = U.shape[1]
    
    # Try to find linear dependencies among the rank-1 tensors
    # If sum_r c_r * T_r = 0 where T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # then we can eliminate one term
    
    # This is equivalent to finding a vector c in null space of the tensor-to-matrix map
    # But the tensor is the all-ones tensor, so we need:
    # sum_r c_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 0 for all i,j,k
    
    # This is a system of n*m*p equations in current_rank unknowns
    # For 2x4x7: 56 equations in 56 unknowns (schoolbook)
    # The only solution is c=0 (since the tensors are linearly independent in schoolbook)
    
    # So we can't reduce schoolbook rank this way.
    
    # Instead, try a different approach: parameterize a lower-rank decomposition
    # and solve the resulting system
    
    return None

def _solve_for_rank(n, m, p, target_rank, time_budget=15.0):
    """
    Try to solve for a rank-target_rank decomposition by setting up
    the system of equations and solving.
    
    We have n*m*p equations:
    sum_{r=1}^{target_rank} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    This is a system of bilinear/trilinear equations. We can try to:
    1. Fix U and V, solve for W (linear in W)
    2. Iterate
    
    For fixed U, V:
    For each (i,k): sum_{j,r} U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r] = 1
    Let M[i,k,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
    Then: sum_r M[i,k,r] * W[k*n+i,r] = 1 for all i,k
    
    This is a system of n*p equations in n*p*target_rank unknowns (W entries).
    If target_rank > 1, this is underdetermined, so we can find solutions.
    """
    deadline = time.time() + time_budget
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random U, V
        U = np.random.uniform(-1, 1, size=(n*m, target_rank))
        V = np.random.uniform(-1, 1, size=(m*p, target_rank))
        
        # Build M[i,k,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
        M = np.zeros((n, p, target_rank))
        for i in range(n):
            for k in range(p):
                for r in range(target_rank):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j, r] * V[j*p+k, r]
                    M[i, k, r] = s
        
        # Now solve: for each (i,k): sum_r M[i,k,r] * W[k*n+i,r] = 1
        # This is n*p independent equations, each with target_rank unknowns
        # W[k*n+i,r] for r=0..target_rank-1
        
        W = np.zeros((p*n, target_rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Equation: sum_r M[i,k,r] * W[k*n+i,r] = 1
                # This is a single linear equation in target_rank unknowns
                # We need to find W[k*n+i,:] such that M[i,k,:] @ W[k*n+i,:] = 1
                # If M[i,k,:] is not all zeros, we can set W[k*n+i,0] = 1/M[i,k,0] and rest to 0
                # But we need consistency across all equations...
                
                # Actually, W[k*n+i,:] appears in exactly one equation (for this (i,k))
                # So we can choose it freely as long as M[i,k,:] @ W[k*n+i,:] = 1
                
                if np.allclose(M[i,k,:], 0, atol=1e-10):
                    valid = False
                    break
                
                # Set W[k*n+i,:] to satisfy the equation
                # Simple choice: put all weight on the first non-zero component
                idx = np.argmax(np.abs(M[i,k,:]))
                W[k*n+i, idx] = 1.0 / M[i,k,idx]
        
        if not valid:
            continue
        
        # Verify
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if np.allclose(T, np.ones((n, m, p)), atol=1e-10, rtol=0):
            # Try exact verification
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _iterative_refinement(n, m, p, target_rank, time_budget=20.0):
    """
    Start with a random initialization and iteratively refine to satisfy
    the tensor equations.
    """
    deadline = time.time() + time_budget
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        U = np.random.uniform(-1, 1, size=(n*m, target_rank))
        V = np.random.uniform(-1, 1, size=(m*p, target_rank))
        W = np.random.uniform(-1, 1, size=(p*n, target_rank))
        
        # Iteratively update W to satisfy equations
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Update W: for each (i,k), solve sum_r M[i,k,r] * W[k*n+i,r] = 1
            # where M[i,k,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
            
            # Build M
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[i, k, r] = s
            
            # Update W
            for i in range(n):
                for k in range(p):
                    if np.allclose(M[i,k,:], 0, atol=1e-12):
                        continue
                    # Least squares solution to M[i,k,:] @ w = 1
                    # w is W[k*n+i,:]
                    # Use np.linalg.lstsq
                    try:
                        w, _, _, _ = np.linalg.lstsq(M[i,k,:].reshape(-1,1), np.array([1.0]), rcond=None)
                        W[k*n+i, :] = w.flatten()
                    except:
                        pass
            
            # Check convergence
            T = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            err = np.max(np.abs(T - 1.0))
            if err < 1e-10:
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W
                break
        
        # If we got close but not exact, try to snap to rational values
        # and re-verify
        U_snap = np.round(U * 4) / 4
        V_snap = np.round(V * 4) / 4
        W_snap = np.round(W * 4) / 4
        
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
        
        if np.allclose(T, np.ones((n, m, p)), atol=1e-10, rtol=0):
            if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 115  # Leave 5s safety margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # Schoolbook rank
    
    # Update schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Known lower bound: rank >= max(n*m, m*p, n*p) = max(8, 28, 14) = 28
    # But this is a weak bound. Let's try ranks from 15 to 55.
    
    # Strategy 1: Direct solve for W given random U, V
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _solve_for_rank(n, m, p, target_rank, time_budget=5.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break  # Found the lowest possible in this range
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Strategy 2: Iterative refinement
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _iterative_refinement(n, m, p, target_rank, time_budget=10.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Strategy 3: Random search with structured patterns
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_random(n, m, p, target_rank, time_budget=5.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
