"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_target_tensor():
        """Build the multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = build_target_tensor()
    
    def schoolbook_decomposition():
        """Standard schoolbook decomposition with rank = n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def verify_decomposition(U, V, W, T_target, tol=1e-10):
        """Verify if U⊗V⊗W reconstructs T_target exactly."""
        r = U.shape[1]
        T_recon = np.zeros_like(T_target)
        for t in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        return np.allclose(T_recon, T_target, atol=tol, rtol=0)
    
    def try_structured_rank_reduction(target_rank, T_target, max_iter=50, tol=1e-10):
        """
        Try to find a rank-target_rank decomposition using structured perturbations
        of the schoolbook decomposition. Uses a smooth surrogate approach.
        """
        if target_rank >= schoolbook_rank:
            return None
        
        # Start from schoolbook and try to merge terms
        # For each pair of terms that might be mergeable, try to find a single term
        # that approximates their sum
        
        # Generate all possible pairs of (i,j,k) triples
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try greedy merging: start with all terms, try to merge pairs
        # that have similar structure
        
        # Alternative: use a parameterized family of rank-r decompositions
        # where r < n*m*p, and optimize the parameters
        
        # For 3x3x6, try a structured approach based on the matrix dimensions
        # Use the fact that we can potentially share computation across k dimensions
        
        # Strategy: Try rank = n*m*p - d for small d, using random linear combinations
        # of the schoolbook terms
        
        best_result = None
        best_loss = np.inf
        
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization near schoolbook
            np.random.seed(trial * 7 + 13)
            
            # Create random mixing of schoolbook terms
            # Each new term is a linear combination of original terms
            U_t = np.zeros((n*m, target_rank))
            V_t = np.zeros((m*p, target_rank))
            W_t = np.zeros((p*n, target_rank))
            
            # Assign each original term to one of the target_rank slots
            # with random weights
            assignments = np.random.randint(0, target_rank, size=schoolbook_rank)
            weights = np.random.randn(schoolbook_rank) * 0.1
            
            for idx, (i, j, k) in enumerate(triples):
                slot = assignments[idx]
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_t[a, slot] += weights[idx]
                V_t[b, slot] += weights[idx]
                W_t[c, slot] += weights[idx]
            
            # Normalize to maintain scale
            for t in range(target_rank):
                if np.any(U_t[:, t] != 0):
                    norm = np.linalg.norm(U_t[:, t]) * np.linalg.norm(V_t[:, t]) * np.linalg.norm(W_t[:, t])
                    if norm > 1e-10:
                        scale = 1.0 / (norm ** (1.0/3.0))
                        U_t[:, t] *= scale
                        V_t[:, t] *= scale
                        W_t[:, t] *= scale
            
            # Gradient descent to refine
            lr = 0.001
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                T_recon = np.zeros_like(T_target)
                for t in range(target_rank):
                    T_recon += np.einsum('a,b,c->abc', U_t[:,t], V_t[:,t], W_t[:,t])
                
                error = T_target - T_recon
                loss = np.sum(error**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U_t.copy(), V_t.copy(), W_t.copy())
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U_t)
                dV = np.zeros_like(V_t)
                dW = np.zeros_like(W_t)
                for t in range(target_rank):
                    # M_b,c = V_b,t * W_c,t
                    M = np.outer(V_t[:,t], W_t[:,t])
                    dU[:,t] = np.einsum('abc,bc->a', error, M)
                    
                    # N_a,c = U_a,t * W_c,t
                    N = np.outer(U_t[:,t], W_t[:,t])
                    dV[:,t] = np.einsum('abc,ac->b', error, N)
                    
                    # P_a,b = U_a,t * V_b,t
                    P = np.outer(U_t[:,t], V_t[:,t])
                    dW[:,t] = np.einsum('abc,ab->c', error, P)
                
                # Adam-like update with momentum
                U_t -= lr * dU
                V_t -= lr * dV
                W_t -= lr * dW
                
                # Adaptive learning rate
                if it % 20 == 0 and it > 0:
                    lr *= 0.9
        
        # Verify best result
        if best_result is not None:
            U_best, V_best, W_best = best_result
            if verify_decomposition(U_best, V_best, W_best, T_target, tol):
                return U_best, V_best, W_best, target_rank
        
        return None
    
    # Start with schoolbook
    U, V, W, current_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), current_rank
    
    # Try progressively lower ranks with structured approach
    # For 3x3x6, theoretical lower bound is around 24-30
    # Try from rank 45 down to 25
    for target_rank in range(45, 24, -1):
        if time.time() > deadline:
            break
        
        result = try_structured_rank_reduction(target_rank, T_target, max_iter=30)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    U_final, V_final, W_final, r_final = best_U, best_V, best_W, best_rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        """Convert numpy array to list of lists, rounding to nearest half-integer if close."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-8:
                    new_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-8:
                    new_row.append(nearest_half)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
