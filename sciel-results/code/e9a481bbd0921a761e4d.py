import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    
    # True Tensor T
    # T[u, v, w] = 1 if indices match, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def error(U, V, W):
        # Compute sum of U_r ^ V_r ^ W_r
        # U, V, W are (dim, rank)
        # Result is (dim1, dim2, dim3)
        # We can compute this by iterating r
        Res = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            Res += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(Res - T)

    # Search for rank 23
    best_U, best_V, best_W = None, None, None
    start_time = time.time()
    deadline = start_time + 20  # 20s budget
    
    # Try a few random initializations
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Simple Gradient Descent
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
            
            # Compute Gradient
            # E = Res - T
            Res = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                Res += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            E = Res - T
            
            # Update U
            # dE/dU[i, r] = sum_{j,k} E[i,j,k] V[j,r] W[k,r]
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                # Efficient computation
                # grad_U[:, r] = np.einsum('ijk,jr,kr->ir', E, V[:, r], W[:, r])
                # But V[:, r] is a vector.
                # grad_U[:, r] = sum_{j,k} E[:, j, k] * V[j, r] * W[k, r]
                # This is (E * V_r * W_r) summed over j, k
                # We can do:
                # Temp = E * np.outer(V[:, r], W[:, r]) ? No.
                # grad_U[:, r] = np.tensordot(E, np.outer(V[:, r], W[:, r]), axes=([1,2],[0,1]))
                
                # Simpler:
                # grad_U[:, r] = np.sum(E * np.outer(V[:, r], W[:, r]), axis=(1, 2))
                # Wait, E is (n*m, m*p, p*n).
                # V[:, r] is (m*p,). W[:, r] is (p*n,).
                # Outer product is (m*p, p*n).
                # E * Outer is (n*m, m*p, p*n).
                # Sum over axis 1, 2 gives (n*m,).
                grad_U[:, r] = np.sum(E * np.outer(V[:, r], W[:, r]), axis=(1, 2))
                
                grad_V[:, r] = np.sum(E * np.outer(U[:, r], W[:, r]), axis=(0, 2))
                grad_W[:, r] = np.sum(E * np.outer(U[:, r], V[:, r]), axis=(0, 1))
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            err = error(U, V, W)
            if err < 1e-6:
                # Found a solution!
                # Round to nearest 0.5
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                if error(U_int, V_int, W_int) < 1e-9:
                    best_U, best_V, best_W = U_int, V_int, W_int
                    break
        
        if best_U is not None:
            break

    if best_U is not None:
        # Convert to list of lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to Seed
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

# EVOLVE-BLOCK-END
