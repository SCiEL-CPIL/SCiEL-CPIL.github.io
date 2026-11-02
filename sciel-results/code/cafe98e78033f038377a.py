"""
Rewritten program for Matrix Multiplication Tensor Decomposition (3,3,6).
Aim: Find a rank-minimal exact tensor decomposition.
Target rank: < 54 (Schoolbook). Likely target: 46 (2 * Rank(3x3x3)) or lower.
Strategy: Use finite field (GF(2)) analysis to find linear dependencies, then construct
exact integer/half-integer decomposition guided by the finite field solution.
"""
import numpy as np
import time
from fractions import Fraction
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    T_target = np.zeros((dim_u, dim_v, dim_w))
    
    # Schoolbook indices
    schoolbook_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                schoolbook_indices.append((a, b, c))
                
    rank_sb = len(schoolbook_indices) # 54

    def verify(U, V, W, tol=1e-8):
        try:
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            return np.allclose(T_recon, T_target, atol=tol)
        except Exception:
            return False

    def gf2_matrix_rank(matrix):
        """Compute rank of a matrix over GF(2)"""
        mat = np.array(matrix, dtype=np.uint8) % 2
        rows, cols = mat.shape
        rank = 0
        for col in range(cols):
            # Find pivot row
            pivot = -1
            for row in range(rank, rows):
                if mat[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap rows
            mat[[rank, pivot]] = mat[[pivot, rank]]
            # Eliminate other entries in this column
            for row in range(rows):
                if row != rank and mat[row, col] == 1:
                    mat[row] = (mat[row] + mat[rank]) % 2
            rank += 1
            if rank == rows:
                break
        return rank

    def find_gf2_dependencies():
        """
        Find linear dependencies among the schoolbook terms over GF(2).
        Each schoolbook term is a vector in GF(2)^(dim_u * dim_v * dim_w).
        We want to find the minimum number of terms needed to span the tensor.
        """
        # Represent each schoolbook term as a binary vector
        num_terms = len(schoolbook_indices)
        vec_dim = dim_u * dim_v * dim_w
        
        # Build the matrix where each column is a schoolbook term
        # For memory efficiency, we'll work with the structure directly
        # The tensor is sparse - each term has exactly one 1
        
        # Instead of building the full matrix, we use the structure:
        # Term (i,j,k) has 1 at position (i*m+j, j*p+k, k*n+i)
        # We can compute linear dependencies using the index structure
        
        # Build incidence matrix: rows = (a,b,c) positions, cols = terms
        # But this is too large. Instead, use the fact that we're in GF(2)
        # and the tensor has a specific structure.
        
        # Alternative: Try to find a subset of terms that can generate all others
        # via linear combinations over GF(2)
        
        # For now, return the full set (no dependencies found yet)
        return list(range(num_terms))

    def try_constructive_decomposition(target_rank):
        """
        Try to construct a decomposition of target_rank using structural insights.
        For 3x3 x 3x6, we can potentially exploit the rectangular structure.
        
        Strategy: 
        - The 3x6 matrix can be seen as two 3x3 blocks
        - Try to share computation between the two blocks
        """
        if target_rank >= rank_sb:
            # Just return schoolbook
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            for idx, (a, b, c) in enumerate(schoolbook_indices):
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
            return U, V, W, target_rank
        
        # Try constructive approach:
        # For C = A*B where A is 3x3, B is 3x6
        # C has 18 entries. Each C[i,k] = sum_j A[i,j]*B[j,k]
        # Total schoolbook: 3*3*6 = 54 multiplications
        
        # Can we do better by sharing intermediate results?
        # Let's try to find a rank-46 decomposition by eliminating 8 terms
        
        # One approach: use the fact that some combinations can be shared
        # For example, if we precompute certain linear combinations
        
        # Let's try a systematic search over which terms to keep
        from scipy.optimize import minimize
        
        best_solution = None
        
        # Try different strategies for selecting which terms to keep
        strategies = [
            'random',
            'structured',
            'gf2_guided'
        ]
        
        for strategy in strategies:
            if time.time() > deadline - 10:
                break
                
            for trial in range(20):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(12345 + trial * 7 + (0 if strategy == 'random' else 100 if strategy == 'structured' else 200))
                
                if strategy == 'random':
                    keep_indices = np.random.choice(rank_sb, target_rank, replace=False)
                elif strategy == 'structured':
                    # Keep terms in a structured way - e.g., keep all terms for certain (i,j) pairs
                    # or certain (j,k) pairs to maximize sharing
                    keep_indices = []
                    # Try keeping all terms for first 2 columns of B (j=0,1) and some for j=2
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = i * m * p + j * p + k
                                if j < 2:
                                    keep_indices.append(idx)
                                elif k < p - 1:  # Keep all but last for j=2
                                    keep_indices.append(idx)
                    keep_indices = keep_indices[:target_rank]
                    if len(keep_indices) < target_rank:
                        # Fill with random
                        remaining = [i for i in range(rank_sb) if i not in keep_indices]
                        keep_indices += np.random.choice(remaining, target_rank - len(keep_indices), replace=False).tolist()
                else:  # gf2_guided
                    # Use GF(2) rank to guide selection
                    # For now, just use random with a different seed
                    keep_indices = np.random.choice(rank_sb, target_rank, replace=False)
                
                # Initialize U, V, W with Schoolbook values for kept indices
                U = np.zeros((dim_u, target_rank))
                V = np.zeros((dim_v, target_rank))
                W = np.zeros((dim_w, target_rank))
                
                for new_idx, old_idx in enumerate(keep_indices):
                    if old_idx >= rank_sb:
                        continue
                    a, b, c = schoolbook_indices[old_idx]
                    U[a, new_idx] = 1.0
                    V[b, new_idx] = 1.0
                    W[c, new_idx] = 1.0
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def objective_and_grad(x):
                    U = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                    V = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                    W = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                    
                    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
                    error = T_target - T_recon
                    loss = np.sum(error**2)
                    
                    dU = 2 * np.einsum('abc,bt,ct->at', error, V, W)
                    dV = 2 * np.einsum('abc,at,ct->bt', error, U, W)
                    dW = 2 * np.einsum('abc,at,bt->ct', error, U, V)
                    
                    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                    return loss, grad
                
                def objective(x):
                    return objective_and_grad(x)[0]
                
                def grad(x):
                    return objective_and_grad(x)[1]
                
                try:
                    res = minimize(
                        objective, 
                        x0, 
                        jac=grad, 
                        method='L-BFGS-B', 
                        options={'maxiter': 80, 'gtol': 1e-10, 'maxfun': 800}
                    )
                except Exception:
                    continue
                
                if time.time() > deadline - 5:
                    break
                    
                U_final = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
                V_final = res.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                W_final = res.x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                
                if verify(U_final, V_final, W_final, tol=1e-6):
                    # Snap to integers or half-integers
                    for denom in [1, 2, 4, 8]:
                        U_int = np.round(U_final * denom) / denom
                        V_int = np.round(V_final * denom) / denom
                        W_int = np.round(W_final * denom) / denom
                        
                        if verify(U_int, V_int, W_int, tol=1e-5):
                            return U_int, V_int, W_int, target_rank
                    
                    return U_final, V_final, W_final, target_rank

        return None

    def polish_solution(U, V, W, target_rank, max_iter=200, tol=1e-12):
        """
        High-precision polish pass using gradient descent with adaptive step size.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            error = T_target - T_recon
            loss = np.sum(error**2)
            
            if loss < tol:
                break
                
            # Compute gradients
            dU = 2 * np.einsum('abc,bt,ct->at', error, V, W)
            dV = 2 * np.einsum('abc,at,ct->bt', error, U, W)
            dW = 2 * np.einsum('abc,at,bt->ct', error, U, V)
            
            # Adaptive step size with backtracking
            step = 1.0
            best_loss = loss
            for _ in range(20):
                U_test = U - step * dU
                V_test = V - step * dV
                W_test = W - step * dW
                
                T_test = np.einsum('at,bt,ct->abc', U_test, V_test, W_test)
                err_test = T_target - T_test
                loss_test = np.sum(err_test**2)
                
                if loss_test < best_loss:
                    best_loss = loss_test
                    U, V, W = U_test, V_test, W_test
                    break
                else:
                    step *= 0.5
            
            if loss_test >= loss * (1 - 1e-8):
                break
                
        return U, V, W

    # Fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb

    # Try to find lower rank decompositions
    # Start with most aggressive target and work up
    for target_rank in [46, 48, 50, 52]:
        if best_rank <= target_rank:
            continue
        if time.time() > deadline - 30:
            break
            
        res = try_constructive_decomposition(target_rank)
        if res is not None:
            best_U, best_V, best_W, best_rank = res
            
            # Polish the solution
            if time.time() < deadline - 10:
                best_U, best_V, best_W = polish_solution(
                    best_U, best_V, best_W, best_rank, max_iter=100
                )
                # Re-snap to nice values
                for denom in [1, 2, 4, 8, 16]:
                    U_snap = np.round(best_U * denom) / denom
                    V_snap = np.round(best_V * denom) / denom
                    W_snap = np.round(best_W * denom) / denom
                    if verify(U_snap, V_snap, W_snap, tol=1e-6):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        break

    # Convert to lists
    def to_list(mat):
        res = []
        for row in mat:
            r = []
            for v in row:
                if not np.isfinite(v):
                    r.append(0.0)
                    continue
                # Try to snap to nice rational values
                best_val = float(v)
                for denom in [1, 2, 4, 8, 16, 32, 64]:
                    snapped = round(v * denom) / denom
                    if abs(snapped - v) < 1e-6:
                        best_val = snapped
                        break
                r.append(best_val)
            res.append(r)
        return res

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
