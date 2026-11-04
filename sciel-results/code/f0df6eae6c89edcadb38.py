import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i

    valid_triples = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
                valid_set.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    U = np.zeros((num_A, best_rank))
    V = np.zeros((num_B, best_rank))
    W = np.zeros((num_C, best_rank))
    col = 0
    for (a, b, c) in valid_triples:
        U[a, col] = 1
        V[b, col] = 1
        W[c, col] = 1
        col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def verify_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # Check all valid triples
        for (a, b, c) in valid_triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val - 1.0) > tol:
                return False
        
        # Check a sample of invalid triples
        rng = np.random.RandomState(42)
        checked = 0
        max_checks = 1000
        while checked < max_checks:
            a = rng.randint(0, num_A)
            b = rng.randint(0, num_B)
            c = rng.randint(0, num_C)
            if (a, b, c) not in valid_set:
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val) > tol:
                    return False
                checked += 1
        return True

    def compute_loss(U, V, W):
        """Compute loss: sum of squared errors on valid + invalid triples."""
        loss = 0.0
        # Valid triples should have value 1
        for (a, b, c) in valid_triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            loss += (val - 1.0) ** 2
        
        # Invalid triples should have value 0 (sample)
        rng = np.random.RandomState(123)
        for _ in range(200):
            a = rng.randint(0, num_A)
            b = rng.randint(0, num_B)
            c = rng.randint(0, num_C)
            if (a, b, c) not in valid_set:
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                loss += val ** 2
        return loss

    def optimize(U, V, W, target_rank, max_steps=500, lr_init=0.1):
        """Optimize the decomposition using gradient descent with momentum."""
        lr = lr_init
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_loss = compute_loss(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Gradient from valid triples
            for (a, b, c) in valid_triples:
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                err = val - 1.0
                grad_U[a, :] += err * np.multiply(V[b, :], W[c, :])
                grad_V[b, :] += err * np.multiply(U[a, :], W[c, :])
                grad_W[c, :] += err * np.multiply(U[a, :], V[b, :])
            
            # Gradient from invalid triples (sample)
            rng = np.random.RandomState(step % 100)
            for _ in range(100):
                a = rng.randint(0, num_A)
                b = rng.randint(0, num_B)
                c = rng.randint(0, num_C)
                if (a, b, c) not in valid_set:
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    grad_U[a, :] += 2 * val * np.multiply(V[b, :], W[c, :])
                    grad_V[b, :] += 2 * val * np.multiply(U[a, :], W[c, :])
                    grad_W[c, :] += 2 * val * np.multiply(U[a, :], V[b, :])
            
            # Update with momentum
            vU = momentum * vU - lr * grad_U
            vV = momentum * vV - lr * grad_V
            vW = momentum * vW - lr * grad_W
            
            U = U + vU
            V = V + vV
            W = W + vW
            
            # Adaptive learning rate
            if step % 50 == 0 and step > 0:
                lr *= 0.95
            
            # Check improvement
            if step % 20 == 0:
                current_loss = compute_loss(U, V, W)
                if current_loss < best_loss - 1e-10:
                    best_loss = current_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 5:
                        break
        
        return best_U, best_V, best_W, best_loss

    # Try to find lower rank decompositions
    for target_rank in range(35, 20, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        num_restarts = 3 if target_rank >= 30 else 2
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random values
            np.random.seed(20000 + target_rank * 100 + restart)
            U_curr = np.random.randn(num_A, target_rank) * 0.3
            V_curr = np.random.randn(num_B, target_rank) * 0.3
            W_curr = np.random.randn(num_C, target_rank) * 0.3
            
            # Optimize
            U_opt, V_opt, W_opt, final_loss = optimize(
                U_curr, V_curr, W_curr, target_rank,
                max_steps=300, lr_init=0.05
            )
            
            # Verify
            if verify_decomposition(U_opt, V_opt, W_opt, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                found = True
                break
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
