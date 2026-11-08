"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _schoolbook():
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


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check whether (U,V,W) is an exact decomposition of the matrix multiplication tensor.
    
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    For each (i,k), we need:
      sum_j sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 if j corresponds to the product term
    Actually, the matrix multiplication tensor T_{i,j,k} = 1 means:
      C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The decomposition says: C_{i,k} = sum_r (sum_j U[i*m+j, r] A_{i,j}) (sum_k V[j*p+k, r] B_{j,k}) W[k*n+i, r]
    
    Wait, let me re-think. The standard tensor decomposition for matrix multiplication:
    T_{i,j,k} (where i indexes rows of A / output, j indexes columns of A / rows of B, k indexes columns of B / output)
    
    T_{i,j,k} = 1 for all valid (i,j,k), and 0 otherwise.
    
    The decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
    
    But the problem uses a different indexing. Let me re-read.
    
    U: shape (n*m, rank) — indexed by (i, j) where i in [0,n), j in [0,m)
    V: shape (m*p, rank) — indexed by (j, k) where j in [0,m), k in [0,p)
    W: shape (p*n, rank) — indexed by (k, i) where k in [0,p), i in [0,n)
    
    The tensor equation: for all i, j, k:
      sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    
    (Since the matrix multiplication tensor has all ones for valid indices.)
    """
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            # sum over j and r
            total = 0.0
            for j in range(m):
                u_val = U_mat[i*m + j]  # vector of length rank
                v_val = V_mat[j*p + k]  # vector of length rank
                w_val = W_mat[k*n + i]  # vector of length rank
                # sum_r u_val[r] * v_val[r] * w_val[r]
                total += np.dot(u_val * v_val, w_val)
            if abs(total - 1.0) > tol:
                return False
    return True


def _build_system_tensors(n, m, p):
    """Precompute the system for alternating least squares.
    
    For fixed U, V: W is determined by linear equations.
    For each (i,k), we need:
      sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is linear in W entries. Let's write it as:
      For each (i,k): sum_r [sum_j U[i*m+j, r] * V[j*p+k, r]] * W[k*n+i, r] = 1
    
    Let A_{(i,k), r} = sum_j U[i*m+j, r] * V[j*p+k, r]
    Then: sum_r A_{(i,k), r} * W[k*n+i, r] = 1
    
    So for each (i,k), the equation involves only W[k*n+i, :] (a row of W).
    The system is: A_{(i,k)} dot W_row = 1, where W_row = W[k*n+i, :]
    
    This is one equation in 'rank' unknowns per (i,k) pair. We have n*p such equations,
    but each involves a different row of W. So actually each row of W is determined by 
    the equations for all i (with fixed k) or all k (with fixed i)?
    
    Wait: W is indexed by (k, i). So W[k*n + i, :] is a row. The equation for (i,k) 
    involves W[k*n + i, :]. So each equation involves exactly one row of W.
    
    For a fixed (k, i), the equation is:
      sum_r A_{(i,k), r} * W[k*n+i, r] = 1
    
    This is one scalar equation in 'rank' unknowns (the entries of W[k*n+i, :]).
    With n*p equations and n*p rows of W (each of length rank), we have an underdetermined
    system if rank > 1. We can solve in least-squares sense.
    
    Actually, each row W[k*n+i, :] appears in exactly one equation. So we have n*p independent
    equations, each constraining one row of W. Each equation is:
      a_{(i,k)} dot w_{(k,i)} = 1
    
    where a_{(i,k)} is a vector of length rank. This is underdetermined (1 equation, rank unknowns).
    We can find the minimum-norm solution: w = a / ||a||^2 * 1.
    """
    pass


def _als_decompose(n, m, p, rank, max_iter=100, tol=1e-10, seed=0):
    """Alternating least squares for fixed rank.
    
    Returns (U, V, W) as lists of lists if successful, or None.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    for it in range(max_iter):
        # Solve for W given U, V
        # For each (i, k): sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        new_W = np.zeros((p * n, rank))
        for i in range(n):
            for k in range(p):
                # Compute a[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                a = np.zeros(rank)
                for j in range(m):
                    a += U[i*m + j] * V[j*p + k]
                # Solve: a dot w = 1, minimum norm solution
                norm_sq = np.dot(a, a)
                if abs(norm_sq) < 1e-15:
                    # Singular; use pseudo-inverse fallback
                    new_W[k*n + i] = np.linalg.lstsq(a.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                else:
                    new_W[k*n + i] = a / norm_sq
        
        # Solve for U given V, W
        # For each (i, k): sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # For fixed (j, r), U[i*m+j, r] appears in equations for all k.
        # Let's rewrite: for each i, j, the unknown is U[i*m+j, :] (a vector of length rank).
        # For fixed i, j: sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let b[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
        # Then: sum_r U[i*m+j, r] * b[r] = 1
        # Minimum norm: U[i*m+j, :] = b / ||b||^2
        new_U = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                b = np.zeros(rank)
                for k in range(p):
                    b += V[j*p + k] * W[k*n + i]
                norm_sq = np.dot(b, b)
                if abs(norm_sq) < 1e-15:
                    new_U[i*m + j] = np.linalg.lstsq(b.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                else:
                    new_U[i*m + j] = b / norm_sq
        
        # Solve for V given U, W
        # For fixed j, k: sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let c[r] = sum_i U[i*m+j, r] * W[k*n+i, r]
        # Then: sum_r V[j*p+k, r] * c[r] = 1
        # Minimum norm: V[j*p+k, :] = c / ||c||^2
        new_V = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                c = np.zeros(rank)
                for i in range(n):
                    c += U[i*m + j] * W[k*n + i]
                norm_sq = np.dot(c, c)
                if abs(norm_sq) < 1e-15:
                    new_V[j*p + k] = np.linalg.lstsq(c.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                else:
                    new_V[j*p + k] = c / norm_sq
        
        U, V, W = new_U, new_V, new_W
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    total += np.dot(U[i*m + j] * V[j*p + k], W[k*n + i])
                error += (total - 1.0) ** 2
        error = np.sqrt(error / (n * p))
        if error < tol:
            break
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
        return U.tolist(), V.tolist(), W.tolist()
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    
    # Try to find lower rank decompositions
    # Try various ranks
    for rank in range(8, n * m * p + 1):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() - start_time > deadline - 5:
                break
            
            result = _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                # Verify again
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found valid decomposition for this rank, try lower
    
    return best

# EVOLVE-BLOCK-END
