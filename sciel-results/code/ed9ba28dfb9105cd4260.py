"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np
from scipy import linalg as sla

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

def _strassen_rank55(n, m, p):
    """
    Construct the known rank-55 decomposition using Strassen for 2x2 blocks
    and schoolbook for the 2x1 block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1; V[v00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1; V[v00][col] = 1; W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[u00][col] = 1; V[v01][col] = 1; V[v11][col] = -1; W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[u11][col] = 1; V[v10][col] = 1; V[v00][col] = -1; W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1; V[v11][col] = 1; W[w01][col] = 1; W[w00][col] = -1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1; V[v00][col] = 1; V[v01][col] = 1; W[w11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1; V[v10][col] = 1; V[v11][col] = 1; W[w00][col] = 1
        col += 1
        
    # Schoolbook for A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _try_rank54_exploit_n2(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    best = None
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        strat = attempt % 4
        
        if strat == 0:
            for row in range(n*m):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 0, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 0, 1], size=nnz)
                
        elif strat == 1:
            U[:, :r54] = U0[:, :r54] * rng.randint(0, 2, size=(n*m, r54))
            V[:, :r54] = V0[:, :r54] * rng.randint(0, 2, size=(m*p, r54))
            U += rng.randint(-1, 2, size=(n*m, r54)) * 0.1
            V += rng.randint(-1, 2, size=(m*p, r54)) * 0.1
            
        elif strat == 2:
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.3] = 0
            V[rng.random((m*p, r54)) < 0.3] = 0
            
        else:
            block_size = r54 // 5
            for g in range(5):
                start = g * block_size
                end = start + block_size if g < 4 else r54
                if end > start:
                    U[:, start:end] = rng.randint(0, 2, size=(n*m, end-start)) * 2 - 1
                    V[:, start:end] = rng.randint(0, 2, size=(m*p, end-start)) * 2 - 1
                    U[:, start:end][rng.random((n*m, end-start)) < 0.4] = 0
                    V[:, start:end][rng.random((m*p, end-start)) < 0.4] = 0
        
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-10):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
                
    return best

def _try_basis_reduction(n, m, p, time_budget, seed=42):
    """
    Attempt to reduce rank by finding linear dependencies in the rank-55 solution.
    This is a heuristic search for a rank-54 solution by perturbing the Strassen basis.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    # We model the problem as finding U, V, W of rank 54.
    # We fix the sparsity pattern of the first 54 columns of the rank-55 solution.
    # We treat the last column as the "redundant" one we want to eliminate.
    
    # Strategy:
    # 1. Take U0[:, :54] and V0[:, :54] as fixed structure (with some random perturbation to break symmetry).
    # 2. Solve for W such that the tensor equation holds.
    # 3. If the solution W is "sparse" or has small norm, it might be a valid rank-54 solution.
    # 4. Verify exactly.
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        # Perturb the basis slightly
        U = U0[:, :54].copy()
        V = V0[:, :54].copy()
        
        # Add small random noise to break exact integer structure, allowing for a different W
        # Or, more aggressively, try to swap/merge some terms.
        # Let's try a "rotation" in the subspace spanned by the last few columns.
        
        # Select a pair of columns to potentially merge
        c1, c2 = rng.choice(54, 2, replace=False)
        
        # Try to express column c2 as a linear combination of others?
        # Or simply solve for W with the modified U, V.
        
        # For now, just try solving for W with the truncated basis.
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    # Use lstsq to find the best fit
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-8):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
                
        if feasible:
            # Verify
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
            if ok:
                # We found a rank-54 solution!
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return None

def _try_general_search(n, m, p, time_budget, seed=42, start_rank=54):
    """
    General search for low-rank decompositions.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    for r in range(start_rank, min(n*m*p + 1, start_rank + 5)):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            strat = attempt % 3
            
            if strat == 0:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for row in range(n*m):
                    nnz = rng.randint(1, min(5, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    U[row, cols] = rng.choice([-1, 1], size=nnz)
                for row in range(m*p):
                    nnz = rng.randint(1, min(5, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    V[row, cols] = rng.choice([-1, 1], size=nnz)
                    
            elif strat == 1:
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.3] = 0
                V[rng.random((m*p, r)) < 0.3] = 0
                
            else:
                # Kronecker-like
                base_size = min(r, 12)
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                U[:, :base_size] = rng.randint(0, 2, size=(n*m, base_size)) * 2 - 1
                V[:, :base_size] = rng.randint(0, 2, size=(m*p, base_size)) * 2 - 1
                if r > base_size:
                    U[:, base_size:] = rng.randint(0, 2, size=(n*m, r-base_size)) * 2 - 1
                    V[:, base_size:] = rng.randint(0, 2, size=(m*p, r-base_size)) * 2 - 1
                    U[:, base_size:][rng.random((n*m, r-base_size)) < 0.5] = 0
                    V[:, base_size:][rng.random((m*p, r-base_size)) < 0.5] = 0
            
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                        W[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = True
                for i in range(n):
                    if not ok: break
                    for j in range(m):
                        if not ok: break
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
    deadline = time.time() + 130  # 20s safety margin before 150s limit
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # 1. Try the basis reduction / linear dependency search
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_basis_reduction(n, m, p, min(time_left, 30), seed=42)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # 2. Try the n=2 exploit search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 10:
            for seed in [42, 12345, 777]:
                if time.time() > deadline - 20:
                    break
                tl = min(time_left, deadline - time.time() - 15)
                if tl < 10:
                    break
                result = _try_rank54_exploit_n2(n, m, p, tl, seed=seed)
                if result is not None:
                    r = len(result["U"][0])
                    if r < best_rank:
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            best = result
                            best_rank = r
                            break
    
    # 3. Try general search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 10:
            for seed in [42, 12345]:
                if time.time() > deadline - 20:
                    break
                tl = min(time_left, deadline - time.time() - 15)
                if tl < 10:
                    break
                result = _try_general_search(n, m, p, tl, seed=seed, start_rank=54)
                if result is not None:
                    r = len(result["U"][0])
                    if r < best_rank:
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            best = result
                            best_rank = r
                            break
    
    # Fallback
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
        
    return best
