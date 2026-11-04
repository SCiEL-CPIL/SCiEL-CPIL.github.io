import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    # Helper to check validity of a solution
    def is_valid(U, V, W, rank):
        # Compute the reconstructed tensor
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            u_reshaped = u.reshape(n, m)
            v_reshaped = v.reshape(m, p)
            w_reshaped = w.reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped, optimize=True)
        return np.allclose(Rec, 1.0, atol=1e-9)

    # Try to find a lower rank solution using structured initialization
    def try_structured(rank, max_iters=1000):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
        
        # Initialize with a pattern that might work
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Use a cyclic pattern
        for r in range(rank):
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        
        # Add some random perturbation
        U += np.random.randn(n * m, rank) * 0.1
        V += np.random.randn(m * p, rank) * 0.1
        W += np.random.randn(p * n, rank) * 0.1
        
        # Gradient descent on error
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient
            Rec = np.zeros((n, m, p))
            grad_U = np.zeros((n * m, rank))
            grad_V = np.zeros((m * p, rank))
            grad_W = np.zeros((p * n, rank))
            
            for r in range(rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                u_reshaped = u.reshape(n, m)
                v_reshaped = v.reshape(m, p)
                w_reshaped = w.reshape(p, n)
                
                term = np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped, optimize=True)
                Rec += term
                
                # Gradient for U[:, r]
                # d/dU[i,j,r] of (Rec[i,j,k] - 1)^2 = 2*(Rec[i,j,k]-1) * V[j,p+k,r] * W[k*n+i,r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            grad_U[i * m + j, r] += 2 * (Rec[i, j, k] - 1) * v_reshaped[j, k] * w_reshaped[k, i]
                
                # Gradient for V[:, r]
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            grad_V[j * p + k, r] += 2 * (Rec[i, j, k] - 1) * u_reshaped[i, j] * w_reshaped[k, i]
                
                # Gradient for W[:, r]
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            grad_W[k * n + i, r] += 2 * (Rec[i, j, k] - 1) * u_reshaped[i, j] * v_reshaped[j, k]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.allclose(Rec, 1.0, atol=1e-8):
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if is_valid(U_snap, V_snap, W_snap, rank):
                    best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                    best_rank = rank
                    return True
                elif is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
        
        return False

    # Try ranks from best_rank - 1 downwards
    current_rank = best_rank - 1
    while current_rank >= 20 and time.time() < deadline:
        if try_structured(current_rank, max_iters=200):
            # Found a solution, continue searching for lower rank
            pass
        current_rank -= 1

    # Ensure the returned values are lists of lists of floats
    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
# EVOLVE-BLOCK-END
