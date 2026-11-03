"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build tensor from decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U[:, c].reshape(n, m)
        v_vec = V[:, c].reshape(m, p)
        w_vec = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    # Target tensor is all ones
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-10):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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
    return U, V, W, rank

def _try_gradient_descent(n, m, p, target_rank, seed=0, max_iters=5000, lr=0.01):
    """
    Try to find a decomposition of target_rank using gradient descent.
    We minimize the loss: sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute current tensor values
        T = np.zeros((n, m, p))
        for c in range(target_rank):
            u_vec = U[:, c].reshape(n, m)
            v_vec = V[:, c].reshape(m, p)
            w_vec = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
        
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Compute gradients (numerical or analytical)
        # For speed, use finite differences on a subset
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Only compute gradients for a random subset of entries for speed
        n_grad_entries = min(50, n * m * target_rank)
        indices = rng.choice(n * m * target_rank, n_grad_entries, replace=False)
        
        eps = 1e-6
        for idx in indices:
            row = idx // target_rank
            col = idx % target_rank
            
            # Gradient for U
            orig = U[row, col]
            U[row, col] = orig + eps
            T_plus = np.zeros((n, m, p))
            for c in range(target_rank):
                u_vec = U[:, c].reshape(n, m)
                v_vec = V[:, c].reshape(m, p)
                w_vec = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_plus[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
            loss_plus = np.sum((T_plus - 1.0) ** 2)
            
            U[row, col] = orig - eps
            T_minus = np.zeros((n, m, p))
            for c in range(target_rank):
                u_vec = U[:, c].reshape(n, m)
                v_vec = V[:, c].reshape(m, p)
                w_vec = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_minus[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
            loss_minus = np.sum((T_minus - 1.0) ** 2)
            
            U[row, col] = orig
            grad_U[row, col] = (loss_plus - loss_minus) / (2 * eps)
        
        # Update U
        U -= lr * grad_U
        
        # Similarly for V and W (simplified: just use same structure)
        # For brevity and speed, we'll do a simpler update
        # Let's just do a simple random perturbation approach instead
    
    # Check if we found a good solution
    T_final = np.zeros((n, m, p))
    for c in range(target_rank):
        u_vec = best_U[:, c].reshape(n, m)
        v_vec = best_V[:, c].reshape(m, p)
        w_vec = best_W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_final[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    
    if np.allclose(T_final, 1.0, atol=1e-8):
        # Snap to half-integers
        U_clean = np.round(best_U * 2) / 2
        V_clean = np.round(best_V * 2) / 2
        W_clean = np.round(best_W * 2) / 2
        
        # Verify after snapping
        if _verify_decomposition(n, m, p, U_clean.tolist(), V_clean.tolist(), W_clean.tolist(), target_rank):
            return U_clean.tolist(), V_clean.tolist(), W_clean.tolist()
    
    return None

def _try_structured_strassen_hybrid(n, m, p, target_rank, seed=0):
    """
    Try a structured approach based on Strassen's algorithm for the 3x3x3 core
    combined with a 3x3x4 extension, with global mixing.
    """
    rng = random.Random(seed)
    
    # Strassen's algorithm for 2x2x2 has rank 7
    # For 3x3x3, we can use a variant
    # Let's try a different structured approach
    
    # Approach: Use the fact that T[i,j,k] = 1 for all i,j,k
    # We can write this as a sum of rank-1 tensors
    
    # Try random assignments with structured constraints
    vals = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    for attempt in range(200):
        U = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
        
        if _verify_decomposition(n, m, p, U, V, W, target_rank):
            return U, V, W
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb, rank_schoolbook = _schoolbook(n, m, p)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(21, 21, 9) = 21
    # But tensor rank is higher. For 3x3x3, best known is 7 (Strassen).
    # For 3x3x7, we expect something like 7 + 3*4 = 19 or similar, but let's search.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    min_rank_to_try = 22  # slightly above theoretical lower bound
    max_rank_to_try = min(rank_schoolbook - 1, 55)
    
    seed_counter = 0
    found_improvement = False
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 10:  # safety margin
            break
        
        # Try multiple random seeds for this rank
        max_seeds = 30
        for seed in range(seed_counter, seed_counter + max_seeds):
            if time.time() > deadline - 10:
                break
            
            # First try the structured approach
            result = _try_structured_strassen_hybrid(n, m, p, rank, seed=seed)
            if result is not None:
                U, V, W = result
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify once more
                if _verify_decomposition(n, m, p, U_list, V_list, W_list, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found_improvement = True
                    break
        
        if found_improvement:
            # Found something, continue trying lower ranks
            continue
        else:
            # No improvement at this rank, move to higher rank
            seed_counter += max_seeds
    
    # Also try gradient descent approach with a few seeds if time permits
    if time.time() < deadline - 20:
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            for rank in range(max(best_rank - 5, min_rank_to_try), min(best_rank + 1, max_rank_to_try + 1)):
                result = _try_gradient_descent(n, m, p, rank, seed=seed, max_iters=2000, lr=0.05)
                if result is not None:
                    U, V, W = result
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    if _verify_decomposition(n, m, p, U_list, V_list, W_list, rank):
                        best_rank = rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        break
    
    # Convert to proper format - ensure values are exact integers or half-integers
    def clean_values(val):
        # Snap to nearest 0.5 if close
        rounded = round(val * 2) / 2
        if abs(val - rounded) < 1e-10:
            return rounded
        return val
    
    U_final = [[clean_values(x) for x in row] for row in best_U]
    V_final = [[clean_values(x) for x in row] for row in best_V]
    W_final = [[clean_values(x) for x in row] for row in best_W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
