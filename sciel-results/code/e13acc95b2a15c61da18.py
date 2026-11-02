"""Optimized tensor decomposition search for 2x4 * 4x6 matrix multiplication.
Combines structured Gaussian elimination on the matmul tensor with gradient-based
refinement and half-integer rounding."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6
NM = N * M  # 8
MP = M * P  # 24
PN = P * N  # 12

def _build_target():
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)."""
    return np.ones((N, M, P), dtype=np.float64)

def _compute_residual(U, V, W, T):
    """Compute residual[i,j,k] = sum_c U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c] - T[i,j,k]"""
    rank = U.shape[1]
    residual = np.zeros((N, M, P), dtype=np.float64)
    for c in range(rank):
        u_c = U[:, c].reshape(N, M)
        v_c = V[:, c].reshape(M, P)
        w_c = W[:, c].reshape(P, N)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    residual[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    residual -= T
    return residual

def _compute_gradients(U, V, W, T):
    """Compute gradients of ||residual||^2 w.r.t. U, V, W."""
    rank = U.shape[1]
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    residual = _compute_residual(U, V, W, T)
    
    for c in range(rank):
        u_c = U[:, c].reshape(N, M)
        v_c = V[:, c].reshape(M, P)
        w_c = W[:, c].reshape(P, N)
        
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    r = residual[i, j, k]
                    dU[i * M + j, c] += 2 * r * V[j * P + k, c] * W[k * N + i, c]
                    dV[j * P + k, c] += 2 * r * U[i * M + j, c] * W[k * N + i, c]
                    dW[k * N + i, c] += 2 * r * U[i * M + j, c] * V[j * P + k, c]
    
    return dU, dV, dW

def _matmul_tensor_check(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    residual = _compute_residual(U, V, W, T)
    return np.max(np.abs(residual)) < tol

def _is_half_integer(arr, tol=1e-6):
    """Check if all entries are close to half-integers."""
    twice = arr * 2.0
    rounded = np.round(twice)
    return np.max(np.abs(twice - rounded)) < tol

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _structured_init_gaussian(r, seed=42):
    """
    Structured initialization using Gaussian elimination ideas.
    For the all-ones tensor, we exploit the fact that T[i,j,k]=1 for all (i,j,k).
    We construct rank-1 terms that each 'cover' a subset of the tensor.
    """
    rng = np.random.RandomState(seed)
    
    U_try = np.zeros((NM, r))
    V_try = np.zeros((MP, r))
    W_try = np.zeros((PN, r))
    
    # Strategy: Use a partition-based approach
    # Divide the index space into r groups and assign each group a rank-1 term
    # For the all-ones tensor, we can use simple products
    
    # Method 1: Cyclic assignment with random phases
    for c in range(r):
        phase = 2.0 * np.pi * c / r
        
        for i in range(N):
            for j in range(M):
                U_try[i * M + j, c] = 1.0 + 0.3 * np.cos(phase + i * 0.7 + j * 0.3)
        
        for j in range(M):
            for k in range(P):
                V_try[j * P + k, c] = 1.0 + 0.3 * np.cos(phase + j * 0.5 + k * 0.4)
        
        for k in range(P):
            for i in range(N):
                W_try[k * N + i, c] = 1.0 + 0.3 * np.cos(phase + k * 0.6 + i * 0.8)
    
    return U_try, V_try, W_try

def _structured_init_kronecker(r, seed=42):
    """
    Kronecker-like structured initialization.
    Exploit the tensor product structure of matrix multiplication.
    """
    rng = np.random.RandomState(seed)
    
    U_try = np.zeros((NM, r))
    V_try = np.zeros((MP, r))
    W_try = np.zeros((PN, r))
    
    # For matrix multiplication, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1
    # 
    # A natural decomposition uses the fact that we can factor through
    # intermediate dimensions. For 2x4 * 4x6, the 'middle' dimension is 4.
    # 
    # Try: use rank terms that correspond to different (i,j) or (j,k) pairs
    
    # Approach: assign each rank term to a specific (i, k) output pair
    # and distribute the summation over j
    output_pairs = [(i, k) for i in range(N) for k in range(P)]
    n_outputs = len(output_pairs)  # 12
    
    if r >= n_outputs:
        # Each output gets its own term, plus extra terms for the sum over j
        for c in range(min(r, n_outputs * M)):
            idx = c
            i_out = idx // (M * P) if idx < n_outputs * M else 0
            # Better: map c to (i, k, j)
            i_val = c // (P * M) if c < N * P * M else 0
            rem = c % (P * M)
            k_val = rem // M
            j_val = rem % M
            
            if i_val < N and k_val < P and j_val < M:
                U_try[i_val * M + j_val, c] = 1.0
                V_try[j_val * P + k_val, c] = 1.0
                W_try[k_val * N + i_val, c] = 1.0
    
    # Add some random perturbation to help optimization escape local minima
    noise_scale = 0.1
    U_try += rng.randn(NM, r) * noise_scale
    V_try += rng.randn(MP, r) * noise_scale
    W_try += rng.randn(PN, r) * noise_scale
    
    return U_try, V_try, W_try

def _random_init(r, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    U_try = rng.randn(NM, r) * 0.5
    V_try = rng.randn(MP, r) * 0.5
    W_try = rng.randn(PN, r) * 0.5
    return U_try, V_try, W_try

def _optimize(U, V, W, T, max_iters=3000, lr_init=0.01, deadline=None, use_adaptive=True):
    """Optimize using gradient descent with adaptive learning rate."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = lr_init
    best_loss = np.inf
    patience = 0
    max_patience = 300
    min_lr = 1e-10
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute loss
        residual = _compute_residual(U, V, W, T)
        loss = np.sum(residual ** 2)
        
        if loss < 1e-14:
            break
        
        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
        
        if use_adaptive:
            if patience > max_patience:
                lr *= 0.5
                patience = 0
                if lr < min_lr:
                    break
            elif it % 200 == 0 and it > 0:
                lr = min(lr * 1.2, 0.1)
        
        # Compute gradients
        dU, dV, dW = _compute_gradients(U, V, W, T)
        
        # Gradient descent step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W

def _try_rank_with_methods(r, T, deadline, max_inits=6):
    """Try to find a valid rank-r decomposition using multiple initialization strategies."""
    inits = []
    
    # Different initialization strategies
    inits.append(('gauss', _structured_init_gaussian(r, seed=r)))
    inits.append(('kronecker', _structured_init_kronecker(r, seed=r)))
    inits.append(('rand1', _random_init(r, seed=r)))
    inits.append(('rand2', _random_init(r, seed=r + 1000)))
    inits.append(('rand3', _random_init(r, seed=r + 2000)))
    inits.append(('rand4', _random_init(r, seed=r + 3000)))
    
    for name, (U0, V0, W0) in inits:
        if time.time() > deadline - 3:
            break
        
        U_opt, V_opt, W_opt = _optimize(U0, V0, W0, T, max_iters=2000, lr_init=0.005, deadline=deadline - 2)
        
        if _matmul_tensor_check(U_opt, V_opt, W_opt, T, tol=1e-9):
            # Try to round to half-integers
            U_round = _round_to_half_integer(U_opt)
            V_round = _round_to_half_integer(V_opt)
            W_round = _round_to_half_integer(W_opt)
            
            if _matmul_tensor_check(U_round, V_round, W_round, T, tol=1e-6):
                return U_round, V_round, W_round
            else:
                return U_opt, V_opt, W_opt
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    T = _build_target()
    
    # Schoolbook fallback (rank 48)
    best_rank = N * M * P  # 48
    best_U = np.zeros((NM, best_rank))
    best_V = np.zeros((MP, best_rank))
    best_W = np.zeros((PN, best_rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                best_U[i * M + j, col] = 1
                best_V[j * P + k, col] = 1
                best_W[k * N + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Lower bound considerations:
    # - Trivial: rank >= max(NM, MP, PN) = 24
    # - For matmul, known lower bounds are higher
    # - Try from 24 up to 47
    
    found = False
    for r in range(24, 48):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_with_methods(r, T, deadline)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_rank = r
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            found = True
            break
    
    # Final output preparation
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Ensure entries are close to half-integers for the final result
    # If not already rounded, try one more round
    if not _is_half_integer(U_out) or not _is_half_integer(V_out) or not _is_half_integer(W_out):
        U_round = _round_to_half_integer(U_out)
        V_round = _round_to_half_integer(V_out)
        W_round = _round_to_half_integer(W_out)
        if _matmul_tensor_check(U_round, V_round, W_round, T, tol=1e-6):
            U_out, V_out, W_out = U_round, V_round, W_round
    
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
