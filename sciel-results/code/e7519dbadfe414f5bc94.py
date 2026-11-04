import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Target: Rec[i, j, k] = 1 for all i, j, k
    # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    best_rank = n * m * p
    best_solution = None
    
    # 1. Generate schoolbook solution as fallback (Rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    # Helper to check validity of a solution
    def is_valid(U, V, W, rank):
        # Compute the reconstructed tensor
        # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Rec[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        # This is a bit tricky with einsum because of the index permutation in W (k, i) vs (i, k)
        # Let's transpose W_3d to (n, p, rank) so it is W_new[i, k, r] = W_3d[k, i, r]
        W_3d_T = W_3d.transpose(1, 0, 2) # Shape (n, p, rank)
        
        # Now Rec[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d_T[i, k, r]
        # einsum: 'ijr,jkr,ikr->ijk'
        Rec = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d_T, optimize=True)
        
        return np.allclose(Rec, 1.0, atol=1e-8)

    def compute_gradients(U, V, W, Rec, rank):
        """
        Compute gradients of 0.5 * sum (Rec - 1)^2 w.r.t U, V, W.
        Rec[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] (with W permuted)
        
        Let Error[i, j, k] = Rec[i, j, k] - 1
        d(Error^2)/dU[i, j, r] = 2 * Error[i, j, k] * V[j, k, r] * W[k, i, r]
        
        We can compute this efficiently using einsum.
        """
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        W_3d_T = W_3d.transpose(1, 0, 2) # (n, p, rank)
        
        Error = Rec - 1.0
        
        # Grad_U[i, j, r] = sum_k 2 * Error[i, j, k] * V_3d[j, k, r] * W_3d_T[i, k, r]
        # einsum: 'ijk,jkr,ikr->ijr'
        grad_U_3d = 2.0 * np.einsum('ijk,jkr,ikr->ijr', Error, V_3d, W_3d_T, optimize=True)
        
        # Grad_V[j, k, r] = sum_i 2 * Error[i, j, k] * U_3d[i, j, r] * W_3d_T[i, k, r]
        # einsum: 'ijk,ijr,ikr->jkr'
        grad_V_3d = 2.0 * np.einsum('ijk,ijr,ikr->jkr', Error, U_3d, W_3d_T, optimize=True)
        
        # Grad_W[k, i, r] = sum_j 2 * Error[i, j, k] * U_3d[i, j, r] * V_3d[j, k, r]
        # Note: We need grad for W_3d which is indexed [k, i, r]
        # grad_W_3d[k, i, r] = sum_j 2 * Error[i, j, k] * U_3d[i, j, r] * V_3d[j, k, r]
        # einsum: 'ijk,ijr,jkr->kir'
        grad_W_3d = 2.0 * np.einsum('ijk,ijr,jkr->kir', Error, U_3d, V_3d, optimize=True)
        
        return grad_U_3d.reshape(n*m, rank), grad_V_3d.reshape(m*p, rank), grad_W_3d.reshape(p*n, rank)

    def try_optimization(rank, num_restarts=10, max_iters=2000):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                return False
                
            # Initialization: Small random values or structured
            # Try initializing with small random integers/half-integers to stay close to lattice
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Optional: Try a specific structured init if rank is close to n*m*p
            # For now, random is fine, but we'll add a "schoolbook-like" sparse init for high ranks
            if rank > 40:
                # Initialize with sparse ones
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                # Add small noise to break symmetry
                U += np.random.randn(n * m, rank) * 0.01
                V += np.random.randn(m * p, rank) * 0.01
                W += np.random.randn(p * n, rank) * 0.01

            lr = 0.05
            prev_error = np.inf
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                W_3d_T = W_3d.transpose(1, 0, 2)
                
                Rec = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d_T, optimize=True)
                error = np.sum((Rec - 1.0)**2)
                
                if error < 1e-10:
                    # Snap to half-integers
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
                    break

                # Adaptive learning rate
                if error < prev_error:
                    lr = min(lr * 1.05, 0.5)
                else:
                    lr = lr * 0.8
                prev_error = error
                
                gU, gV, gW = compute_gradients(U, V, W, Rec, rank)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Periodic snapping to check for integer solutions
                if it % 100 == 0:
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if is_valid(U_snap, V_snap, W_snap, rank):
                        best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                        best_rank = rank
                        return True

        return False

    # Search for lower ranks
    # Start from best_rank - 1 down to a reasonable lower bound
    # Theoretical lower bound for 3x3x5 is not trivially 45, but let's try down to 30
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        # Try a few restarts for each rank
        if try_optimization(r, num_restarts=5, max_iters=500):
            # If we found a solution, we can continue to lower ranks, 
            # but it's unlikely to find much lower immediately.
            # We break if we found the absolute best possible? No, we just update best.
            # If we found rank r, we should try r-1, r-2...
            # But if we found 44, 43 might exist.
            # The loop structure above handles this by continuing the for loop.
            pass

    # Ensure the returned values are lists of lists of floats
    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
# EVOLVE-BLOCK-END
