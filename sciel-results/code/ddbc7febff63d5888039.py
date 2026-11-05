import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

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

    def compute_reconstruction(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def snap_to_half_integer(arr):
        """Snap values to nearest half-integer (0, 0.5, 1, 1.5, ... or negative)"""
        return np.round(arr * 2) / 2

    def gradient_descent(U_init, V_init, W_init, rank, max_iters=500, lr_init=0.01, lr_decay=0.999):
        """
        Gradient descent on the smooth objective: sum(Rec - 1)^2
        Then snap to half-integers and validate.
        """
        U = U_init.astype(float).copy()
        V = V_init.astype(float).copy()
        W = W_init.astype(float).copy()
        
        lr = lr_init
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute reconstruction and gradient
            Rec = np.zeros((n, m, p))
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                
                term = np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
                Rec += term
                
                # Gradient for U[:, r]: d/dU[i,j,r] = 2*(Rec[i,j,k]-1) * V[j,k,r] * W[k,i,r]
                # We need to sum over k
                # grad_U[i*m+j, r] = 2 * sum_k (Rec[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        # Vectorized over k
                        vk = V[j*p:(j+1)*p, r]  # shape (p,)
                        wk = W[0:n*p, r].reshape(p, n)[:, i]  # shape (p,)
                        err_k = Rec[i, j, :] - 1.0  # shape (p,)
                        grad_U[i*m+j, r] = 2.0 * np.dot(err_k, vk * wk)
                
                # Gradient for V[:, r]: d/dV[j,k,r] = 2*(Rec[i,j,k]-1) * U[i,j,r] * W[k,i,r]
                for j in range(m):
                    for k in range(p):
                        uk = U[:, r].reshape(n, m)[:, j]  # shape (n,)
                        wk = W[k*n:(k+1)*n, r]  # shape (n,)
                        err_i = Rec[:, j, k] - 1.0  # shape (n,)
                        grad_V[j*p+k, r] = 2.0 * np.dot(err_i, uk * wk)
                
                # Gradient for W[:, r]: d/dW[k,i,r] = 2*(Rec[i,j,k]-1) * U[i,j,r] * V[j,k,r]
                for k in range(p):
                    for i in range(n):
                        uk = U[:, r].reshape(n, m)[i, :]  # shape (m,)
                        vk = V[:, r].reshape(m, p)[:, k]  # shape (m,)
                        err_j = Rec[i, :, k] - 1.0  # shape (m,)
                        grad_W[k*n+i, r] = 2.0 * np.dot(err_j, uk * vk)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            err = np.max(np.abs(Rec - 1.0))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            lr *= lr_decay
        
        # Snap to half-integers
        U_snap = snap_to_half_integer(best_U)
        V_snap = snap_to_half_integer(best_V)
        W_snap = snap_to_half_integer(best_W)
        
        if is_valid(U_snap, V_snap, W_snap, rank):
            return U_snap, V_snap, W_snap, True
        
        # Also try without snapping (in case the solution is already close)
        if is_valid(best_U, best_V, best_W, rank):
            return best_U, best_V, best_W, True
        
        return None, None, None, False

    def try_structured_init(rank, pattern='cyclic', max_iters=300):
        """
        Try a structured initialization and optimize.
        """
        if rank >= best_rank:
            return False
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        if pattern == 'cyclic':
            # Cyclic pattern: each rank corresponds to a (i,j,k) triple
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
        elif pattern == 'block':
            # Block pattern
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                # Use different values for different blocks
                val = 1.0 if (i + j + k) % 2 == 0 else -1.0
                U[i * m + j, r] = val
                V[j * p + k, r] = val
                W[k * n + i, r] = val
        elif pattern == 'random_structured':
            # Random but with structure: some entries are 1, some -1, some 0
            np.random.seed(rank * 1000 + hash(pattern) % 10000)
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = np.random.choice([-1, 0, 1])
                V[j * p + k, r] = np.random.choice([-1, 0, 1])
                W[k * n + i, r] = np.random.choice([-1, 0, 1])
        
        # Add small random perturbation to break symmetry
        U += np.random.randn(n * m, rank) * 0.05
        V += np.random.randn(m * p, rank) * 0.05
        W += np.random.randn(p * n, rank) * 0.05
        
        U_opt, V_opt, W_opt, success = gradient_descent(U, V, W, rank, max_iters=max_iters)
        
        if success:
            nonlocal best_solution, best_rank
            best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
            best_rank = rank
            return True
        
        return False

    def random_search(rank, n_attempts=10, max_iters=200):
        """
        Random initialization + gradient descent.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return False
            
            np.random.seed(rank * 1000 + attempt * 7 + 42)
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], (n * m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], (m * p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], (p * n, rank))
            
            # Add small perturbation
            U += np.random.randn(n * m, rank) * 0.02
            V += np.random.randn(m * p, rank) * 0.02
            W += np.random.randn(p * n, rank) * 0.02
            
            U_opt, V_opt, W_opt, success = gradient_descent(U, V, W, rank, max_iters=max_iters)
            
            if success:
                nonlocal best_solution, best_rank
                best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                best_rank = rank
                return True
        
        return False

    # Try to find better solutions
    # Start from rank 44 down to 30
    for rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 5:
            break
        
        # Try structured initializations
        for pattern in ['cyclic', 'block', 'random_structured']:
            if time.time() > deadline - 5:
                break
            try_structured_init(rank, pattern, max_iters=200)
            if best_rank < rank:
                break
        
        # Try random search
        if best_rank >= rank and time.time() < deadline - 5:
            random_search(rank, n_attempts=5, max_iters=150)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
