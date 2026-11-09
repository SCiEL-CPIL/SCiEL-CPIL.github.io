"""Matrix multiplication 3x4x7 - rank-minimal tensor decomposition via linear algebra search."""
DIMENSIONS = [3, 4, 7]

import numpy as np
from fractions import Fraction
import time
import itertools
import sys

def _verify(U, V, W, n=3, m=4, p=7):
    """Verify the decomposition: sum_t U[:,t] outer V[:,t] outer W[:,t] = T"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    T = np.zeros((n, m, p))
    for t in range(U.shape[1]):
        u = U[:, t]  # (n*m,)
        v = V[:, t]  # (m*p,)
        w = W[:, t]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    # Expected tensor
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i, j, k] = 1
    return np.allclose(T, T_expected, atol=1e-9)

def _schoolbook(n=3, m=4, p=7):
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

def _strassen_2x2x2_block(i0, j0, k0, U, V, W, start_col, n, m, p):
    """
    Add a 2x2x2 Strassen block to the decomposition.
    """
    A00 = (i0 + 0) * m + (j0 + 0)
    A01 = (i0 + 0) * m + (j0 + 1)
    A10 = (i0 + 1) * m + (j0 + 0)
    A11 = (i0 + 1) * m + (j0 + 1)
    
    B00 = (j0 + 0) * p + (k0 + 0)
    B01 = (j0 + 0) * p + (k0 + 1)
    B10 = (j0 + 1) * p + (k0 + 0)
    B11 = (j0 + 1) * p + (k0 + 1)
    
    C00 = (k0 + 0) * n + (i0 + 0)
    C01 = (k0 + 1) * n + (i0 + 0)
    C10 = (k0 + 0) * n + (i0 + 1)
    C11 = (k0 + 1) * n + (i0 + 1)
    
    terms = [
        ({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1}),
        ({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1}),
        ({A00: 1}, {B10: 1, B11: -1}, {C01: 1, C11: 1}),
        ({A11: 1}, {B01: 1, B10: -1}, {C00: 1, C10: 1}),
        ({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1}),
        ({A10: 1, A00: -1}, {B00: 1, B10: 1}, {C11: 1}),
        ({A01: 1, A11: -1}, {B01: 1, B11: 1}, {C00: 1}),
    ]
    
    for a_coeff, b_coeff, c_coeff in terms:
        col = start_col
        for a_idx, a_val in a_coeff.items():
            U[a_idx][col] += a_val
        for b_idx, b_val in b_coeff.items():
            V[b_idx][col] += b_val
        for c_idx, c_val in c_coeff.items():
            W[c_idx][col] += c_val
        start_col += 1
    
    return start_col

def _direct_term(i, j, k, U, V, W, col, n, m, p):
    """Add a direct rank-1 term for C[i,k] += A[i,j]*B[j,k]"""
    U[i * m + j][col] += 1
    V[j * p + k][col] += 1
    W[k * n + i][col] += 1
    return col + 1

def _rank_78_construction():
    """
    Construct rank-78 decomposition using Strassen blocks + direct terms.
    """
    n, m, p = 3, 4, 7
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 6 Strassen blocks
    for (i0, j0, k0) in [(0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)]:
        col = _strassen_2x2x2_block(i0, j0, k0, U, V, W, col, n, m, p)
    
    # Remaining: C[0,6], C[1,6] (8 direct terms)
    for j in range(4):
        col = _direct_term(0, j, 6, U, V, W, col, n, m, p)
    for j in range(4):
        col = _direct_term(1, j, 6, U, V, W, col, n, m, p)
    
    # C[2,0:6] (24 direct terms)
    for j in range(4):
        for k in range(6):
            col = _direct_term(2, j, k, U, V, W, col, n, m, p)
    
    # C[2,6] (4 direct terms)
    for j in range(4):
        col = _direct_term(2, j, 6, U, V, W, col, n, m, p)
    
    assert col == 78, f"Expected 78, got {col}"
    
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _linear_algebra_search(n=3, m=4, p=7, max_rank=80, time_limit=100):
    """
    Use linear algebra to find a low-rank decomposition.
    
    The tensor T[i,j,k] = 1 for all i,j,k.
    We want to find U (nm x r), V (mp x r), W (pn x r) such that
    sum_t U[:,t] V[:,t] W[:,t] = T (as a 3-tensor)
    
    This is equivalent to: for each (i,j,k),
    sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    We can fix W and solve for U, V via linear equations.
    """
    deadline = time.time() + time_limit
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try various ranks starting from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But realistically, we need much more.
    # Let's try ranks from 60 to 80
    
    for target_rank in range(60, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        print(f"Trying rank {target_rank}...", file=sys.stderr)
        
        # Strategy: Fix W to be random (or structured), then solve for U, V
        # The equation is: for each (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        
        # This is a system of nonlinear equations. Instead, let's use a different approach.
        # 
        # Alternative: Use the fact that the matrix multiplication tensor has a specific structure.
        # We can try to find a decomposition where U and V have sparse rows.
        
        # For now, let's try a simpler approach: start from the rank-78 construction
        # and try to reduce the rank by finding linear dependencies.
        
        if target_rank >= 78:
            break
        
        # For ranks below 78, we need a more sophisticated approach.
        # Let's try to use the existing rank-78 and see if we can eliminate columns.
        
        if target_rank == 77:
            # Try to find a column that can be eliminated
            result_78 = _rank_78_construction()
            U = np.array(result_78["U"], dtype=float)
            V = np.array(result_78["V"], dtype=float)
            W = np.array(result_78["W"], dtype=float)
            
            # Check if any column is a linear combination of others
            # This is hard to do directly. Instead, let's try random perturbations.
            
            # Try removing each column and see if we can compensate
            for col_to_remove in range(78):
                if time.time() > deadline - 2:
                    break
                
                U_new = np.delete(U, col_to_remove, axis=1)
                V_new = np.delete(V, col_to_remove, axis=1)
                W_new = np.delete(W, col_to_remove, axis=1)
                
                # Check if the remaining decomposition still works
                # (It won't, since we removed a term, but let's verify)
                if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                    best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                    best_rank = 77
                    return best_result
    
    return best_result

def _try_bilinear_search(n=3, m=4, p=7, time_limit=90):
    """
    Try to find a lower rank decomposition using bilinear search.
    
    Fix one matrix (say W) to have a specific structure, then solve for U and V.
    """
    deadline = time.time() + time_limit
    best_result = None
    best_rank = 84  # schoolbook
    
    # Try fixing W to be a permutation-like matrix
    # W has shape (p*n, r) = (21, r)
    
    for r in range(60, 80):
        if time.time() > deadline - 5:
            break
        
        print(f"Bilinear search: trying rank {r}...", file=sys.stderr)
        
        # Fix W to be random with entries in {-1, 0, 1}
        # Then solve for U and V
        
        # The equation is:
        # For each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        
        # Let's fix W first. Use a structured W.
        # W[k*n+i, t] for k in 0..6, i in 0..2, t in 0..r-1
        
        # Try W as a partial identity-like matrix
        W = np.zeros((p * n, r))
        for t in range(min(r, p * n)):
            W[t, t] = 1
        
        # Now solve for U and V.
        # For each t, U[:,t] and V[:,t] are vectors.
        # The constraint is: for each (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        
        # This is still nonlinear. Let's use a different approach.
        
        # Fix U to be random, then solve for V.
        # For fixed U and W, the equation becomes:
        # For each (i,j,k): sum_t (U[i*m+j,t] * W[k*n+i,t]) * V[j*p+k,t] = 1
        # This is linear in V!
        
        # Let A[i,j,k,t] = U[i*m+j,t] * W[k*n+i,t]
        # Then: for each (i,j,k), sum_t A[i,j,k,t] * V[j*p+k,t] = 1
        
        # This is a system of nm*p = 84 equations in mp*r unknowns.
        # But V has shape (mp, r), so there are mp*r = 28*r unknowns.
        
        # For r = 60, we have 28*60 = 1680 unknowns and 84 equations. Underdetermined.
        # We can find a solution.
        
        U = np.random.randint(-1, 2, size=(n * m, r))
        # Ensure U is not all zeros
        if np.all(U == 0):
            U[0, 0] = 1
        
        # Build the linear system for V
        # Unknowns: V[j*p+k, t] for j in 0..3, k in 0..6, t in 0..r-1
        # Equations: for each (i,j,k), sum_t U[i*m+j,t] * W[k*n+i,t] * V[j*p+k,t] = 1
        
        # Let's index V as a flat vector of length mp*r
        # V_flat[idx] where idx = (j*p+k)*r + t
        
        num_eq = n * m * p  # 84
        num_unknowns = m * p * r  # 28*r
        
        A_mat = np.zeros((num_eq, num_unknowns))
        b_vec = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        val = U[i * m + j, t] * W[k * n + i, t]
                        if val != 0:
                            unknown_idx = (j * p + k) * r + t
                            A_mat[eq_idx, unknown_idx] += val
                    eq_idx += 1
        
        # Solve A_mat @ V_flat = b_vec
        # Use least squares
        try:
            V_flat, residuals, rank_svd, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is good
        residual = np.linalg.norm(A_mat @ V_flat - b_vec)
        if residual < 1e-6:
            # Reshape V_flat to V
            V = V_flat.reshape((m * p, r))
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    # First, get the rank-78 construction
    try:
        result_78 = _rank_78_construction()
        if _verify(result_78["U"], result_78["V"], result_78["W"], n, m, p):
            best_result = result_78
            best_rank = 78
        else:
            best_result = _schoolbook(n, m, p)
            best_rank = 84
    except Exception as e:
        print(f"Error in rank-78 construction: {e}", file=sys.stderr)
        best_result = _schoolbook(n, m, p)
        best_rank = 84
    
    # Try to improve with linear algebra search
    if time.time() < deadline - 10:
        try:
            improved = _linear_algebra_search(n, m, p, max_rank=80, time_limit=50)
            if improved is not None:
                if _verify(improved["U"], improved["V"], improved["W"], n, m, p):
                    new_rank = len(improved["U"][0])
                    if new_rank < best_rank:
                        best_result = improved
                        best_rank = new_rank
        except Exception as e:
            print(f"Error in linear algebra search: {e}", file=sys.stderr)
    
    # Try bilinear search
    if time.time() < deadline - 10:
        try:
            improved = _try_bilinear_search(n, m, p, time_limit=50)
            if improved is not None:
                if _verify(improved["U"], improved["V"], improved["W"], n, m, p):
                    new_rank = len(improved["U"][0])
                    if new_rank < best_rank:
                        best_result = improved
                        best_rank = new_rank
        except Exception as e:
            print(f"Error in bilinear search: {e}", file=sys.stderr)
    
    return best_result

# EVOLVE-BLOCK-END
