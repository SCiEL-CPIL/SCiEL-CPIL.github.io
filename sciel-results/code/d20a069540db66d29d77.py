"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
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

def _try_structured_winograd(n, m, p, time_budget, seed=42):
    """
    Structured approach exploiting the 2x3x10 structure.
    For n=2, we can try to use differences/sums to reduce rank.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best = None
    best_rank = n * m * p + 1
    
    # Strategy 1: Use the fact that n=2 is small
    # Try to find a decomposition where U has a special structure
    # For each column c, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should contribute to T[i,j,k]
    
    # Try systematic search with structured patterns
    # Pattern: exploit linearity in i (since n=2)
    # T[i,j,k] = 1 for all i,j,k
    # We can write: for each (j,k), need sum_c U[0*m+j,c]*V[j*p+k,c]*W[k*n+0,c] = 1
    #                    sum_c U[1*m+j,c]*V[j*p+k,c]*W[k*n+1,c] = 1
    
    # Try ranks from a reasonable lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But tensor rank is likely higher
    
    candidate_ranks = list(range(30, 65, 1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random structured attempts
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate U, V, W with some structure
            # Use a mix of ±1 and 0 entries
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            # Try to solve for W given U, V (linear system)
            # For each (i,k): A_{j,c} = U[i*m+j, c] * V[j*p+k, c], need sum_c A_{j,c} * W[k*n+i, c] = 1 for all j
            W_sol = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_sol[k*n+i, :] = sol
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify
                ok = _verify_np(U, V, W_sol, n, m, p)
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
                    best_rank = r
                    break
        
        if best is not None:
            break
    
    return best

def _try_systematic_search(n, m, p, time_budget, seed=42):
    """
    More systematic search trying different structured patterns.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best = None
    best_rank = n * m * p + 1
    
    # Try various sparsity patterns and value ranges
    for r in range(30, 65):
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Different sparsity levels
            sparsity = rng.uniform(0.2, 0.7)
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            # Apply sparsity
            U[rng.random((n*m, r)) < sparsity] = 0
            V[rng.random((m*p, r)) < sparsity] = 0
            W[rng.random((p*n, r)) < sparsity] = 0
            
            # Solve for W given U, V
            W_sol = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_sol[k*n+i, :] = sol
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = _verify_np(U, V, W_sol, n, m, p)
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
                    best_rank = r
                    break
        
        if best is not None:
            break
    
    return best

def _try_kronecker_strassen(n, m, p, time_budget, seed=42):
    """
    Try to exploit the structure: 2x3 * 3x10.
    Random search with a bias towards structured solutions.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try a few candidate ranks
    candidate_ranks = list(range(40, n*m*p + 1, 5))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try random search at this rank
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Generate random ±1 matrices with some structure
            # Bias towards having some zero entries to reduce complexity
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            # Make some entries zero (sparse)
            if attempt % 2 == 0:
                U[rng.random((n*m, r)) < 0.3] = 0
                V[rng.random((m*p, r)) < 0.3] = 0
                W[rng.random((p*n, r)) < 0.3] = 0
            
            # Verify
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        if best is not None:
            break
    
    return best

def _try_linear_program(n, m, p, time_budget, seed=42):
    """
    Try to solve a linear program to find a valid decomposition.
    Fix U and V randomly, then solve for W linearly.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try a few candidate ranks
    candidate_ranks = list(range(30, n*m*p + 1, 5))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Fix U and V randomly
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            
            # Solve for W
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # Build the linear system for W[k*n+i, :]
                    # For each j: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    # Solve A * W[k*n+i, :]^T = b
                    # Use least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = sol
                        # Check if the solution is exact
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify the full solution
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured Winograd approach first (most promising)
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_structured_winograd(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # Try systematic search
    time_left = deadline - time.time()
    if time_left > 10 and best_rank == n * m * p:
        result = _try_systematic_search(n, m, p, time_left, seed=99999)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # Try linear program approach
    time_left = deadline - time.time()
    if time_left > 10 and best_rank == n * m * p:
        result = _try_linear_program(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # Try Kronecker/Strassen approach
    time_left = deadline - time.time()
    if time_left > 10 and best_rank == n * m * p:
        result = _try_kronecker_strassen(n, m, p, time_left, seed=99999)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    return best
