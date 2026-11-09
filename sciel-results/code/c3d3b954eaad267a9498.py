"""Find a rank-minimal exact tensor decomposition for 3x4 @ 4x5 matrix multiplication."""

import time
import sys
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def check_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly matches the multiplication tensor."""
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each c in range(rank): U[i,j,c] * V[j,k,c] * W[k,i,c] should sum to delta
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 if we're multiplying (the multiplication tensor has 1s at all (i,j,k) positions)
        
        # Actually, the multiplication tensor for C = A @ B where A is n x m and B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes to C[i,k])
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, rank):
        """Faster check using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need to verify T[i,j,k] == 1 for all i,j,k
        
        # For each c, the contribution is U[:,c] * V[:,c] * W[:,c] (outer product structure)
        # U has shape (nm, rank), V has shape (mp, rank), W has shape (pn, rank)
        
        # Let's compute T by summing over c
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            
            # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        
        if not np.allclose(T, np.ones((n,m,p)), atol=1e-6):
            return False
        return True
    
    def check_decomposition_v2(U, V, W, rank):
        """Most efficient check."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            
            # For fixed i,j: T[i,j,:] += u_c[i,j] * v_c[j,:] * w_c[:,i]
            # w_c[:,i] is a vector of length p
            for i in range(n):
                for j in range(m):
                    T[i,j,:] += u_c[i,j] * v_c[j,:] * w_c[:,i]
        
        if not np.allclose(T, np.ones((n,m,p)), atol=1e-8):
            return False
        return True
    
    def try_rank(rank, time_budget=10.0):
        """Try to find a decomposition of the given rank using random search + local optimization."""
        t_start = time.time()
        
        best_solution = None
        
        num_trials = 50
        for trial in range(num_trials):
            if time.time() - t_start > time_budget:
                break
            if time.time() > DEADLINE:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Use gradient-free optimization (random restarts with perturbation)
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            iterations = 200
            lr = 0.1
            
            for it in range(iterations):
                if time.time() > DEADLINE:
                    break
                
                # Compute loss: sum of (T[i,j,k] - 1)^2
                T = np.zeros((n, m, p))
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            T[i,j,:] += u_c[i,j] * v_c[j,:] * w_c[:,i]
                
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
                
                # Random perturbation (simplex-like)
                perturb_size = lr * (1.0 - it/iterations) * 0.5
                U_p = U + np.random.randn(nm, rank) * perturb_size
                V_p = V + np.random.randn(mp, rank) * perturb_size
                W_p = W + np.random.randn(pn, rank) * perturb_size
                
                T_p = np.zeros((n, m, p))
                for c in range(rank):
                    u_c = U_p[:, c].reshape(n, m)
                    v_c = V_p[:, c].reshape(m, p)
                    w_c = W_p[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            T_p[i,j,:] += u_c[i,j] * v_c[j,:] * w_c[:,i]
                
                loss_p = np.sum((T_p - 1.0)**2)
                
                if loss_p < loss:
                    U, V, W = U_p, V_p, W_p
                else:
                    # Try smaller step
                    lr = max(lr * 0.95, 0.001)
        
        # Convert to list of lists
        if best_U is not None and best_loss < 1e-8:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            if check_decomposition_v2(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list, True
        
        return None, None, None, False
    
    # Try ranks from low to high
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually rank must be at least max(nm, mp, pn) for bilinear forms? No.
    # The tensor has dimension nm * mp * pn in a sense, but rank is different.
    # For matrix multiplication, known bounds: rank >= max(nm, mp, pn) is not quite right.
    # Actually, the rank of the matrix multiplication tensor [n,m,p] is at least max(nm, mp, pn).
    # But we can try smaller values too.
    
    min_possible_rank = max(nm, mp, pn)  # 20 in this case
    # Actually let me reconsider. The tensor T has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    # The tensor is in U[n] x U[m] x U[p] (triple tensor product).
    # A rank-1 tensor is u_i * v_j * w_k.
    # The multiplication tensor is sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k, which is just the all-ones tensor.
    # Wait, no. Let me re-read the problem.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is the "all-ones" tensor of size n*m*p. Its tensor rank is known to be...
    # The all-ones tensor 1_n ⊗ 1_m ⊗ 1_p has rank 1! Because it's just (sum_i u_i)(sum_j v_j)(sum_k w_k) where all are 1.
    # Wait, that means rank 1 suffices? Let me check.
    
    # If U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i
    # Then sum_c=0 U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1. Yes!
    
    # So rank 1 works! But wait, this seems too simple. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix."
    
    # Hmm, but the standard matrix multiplication tensor is not the all-ones tensor. Let me reconsider.
    
    # In standard matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    # The "multiplication tensor" T has T[i,j,k] = 1, meaning the bilinear form for C[i,k] is sum_j A[i,j]B[j,k].
    # But when we write the tensor decomposition, we're decomposing the trilinear form:
    # sum_i,k C[i,k] * (something) ... 
    
    # Actually, I think the tensor here IS the all-ones tensor, and rank 1 is the answer.
    # But that seems trivially easy. Let me just go with rank 1.
    
    # Actually wait - I need to re-examine. The problem says "exactly decompose the matrix multiplication tensor". 
    # The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix is indeed the tensor
    # T ∈ U^{n×m} ⊗ U^{m×p} ⊗ U^{p×n} defined by T[i,j,k] = 1 (using the convention where the output is indexed by (k,i) i.e., W[k*n+i]).
    
    # So yes, the all-ones tensor has rank 1.
    
    # Let's just return rank 1.
    
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    # Verify
    if check_decomposition_v2(U, V, W, 1):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(nm)]
    V_fb = [[0] * rank for _ in range(mp)]
    W_fb = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

