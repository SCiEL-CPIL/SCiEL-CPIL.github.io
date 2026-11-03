"""Rewritten program for 3x3x7 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Build list of (a,b,c,expected) entries for the tensor."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries.append((a, b, c, 1.0))
    return entries

def _build_all_nonzero(n, m, p):
    """Set of all (a,b,c) that should be nonzero."""
    s = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s.add((a, b, c))
    return s

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130
    
    target = _build_target(n, m, p)
    nonzero_set = _build_all_nonzero(n, m, p)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 63
    
    # Build schoolbook solution
    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_fb, V_fb, W_fb = schoolbook()
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook
    
    # Verification function
    def verify(U, V, W, rank, tol=1e-7):
        for a, b, c, exp in target:
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - exp) > tol:
                return False
        # Check some random non-target entries
        rng = np.random.RandomState(99)
        for _ in range(500):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            if (a, b, c) not in nonzero_set:
                val = np.dot(U[a], V[b] * W[c])
                if abs(val) > tol:
                    return False
        return True
    
    def compute_error(U, V, W):
        err = 0.0
        for a, b, c, exp in target:
            val = np.dot(U[a], V[b] * W[c])
            err += (val - exp) ** 2
        rng = np.random.RandomState(99)
        for _ in range(200):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            if (a, b, c) not in nonzero_set:
                val = np.dot(U[a], V[b] * W[c])
                err += val ** 2
        return err
    
    # Try to find lower rank using gradient-based optimization
    # We'll try ranks from rank_schoolbook-1 down
    # Use a structured approach: initialize from schoolbook and optimize
    
    for rank in range(rank_schoolbook - 1, 25, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        found = False
        for restart in range(10):
            if time.time() - start_time > DEADLINE:
                break
            
            # Initialize
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize from schoolbook pattern (take first rank columns)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        if c < rank:
                            U[i * m + j, c] = 1.0
                            V[j * p + k, c] = 1.0
                            W[k * n + i, c] = 1.0
            
            # Add small random perturbation
            np_rng = np.random.RandomState(restart * 1000 + 42)
            U += np_rng.randn(n * m, rank) * 0.05
            V += np_rng.randn(m * p, rank) * 0.05
            W += np_rng.randn(p * n, rank) * 0.05
            
            # Gradient descent with Adam-like updates
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            mU = np.zeros_like(U)
            mV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            t = 0
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(300):
                if time.time() - start_time > DEADLINE:
                    break
                t += 1
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Target entries
                for a, b, c, exp in target:
                    T_val = np.dot(U[a], V[b] * W[c])
                    E = T_val - exp
                    dU[a] += 2 * E * (V[b] * W[c])
                    dV[b] += 2 * E * (U[a] * W[c])
                    dW[c] += 2 * E * (U[a] * V[b])
                
                # Non-target entries (sample)
                rng_iter = np.random.RandomState(iteration * 7 + restart)
                for _ in range(30):
                    a = rng_iter.randint(0, n * m)
                    b = rng_iter.randint(0, m * p)
                    c = rng_iter.randint(0, p * n)
                    if (a, b, c) not in nonzero_set:
                        T_val = np.dot(U[a], V[b] * W[c])
                        dU[a] += 2 * T_val * (V[b] * W[c])
                        dV[b] += 2 * T_val * (U[a] * W[c])
                        dW[c] += 2 * T_val * (U[a] * V[b])
                
                # Adam update
                mU = beta1 * mU + (1 - beta1) * dU
                mV = beta1 * mV + (1 - beta1) * dV
                mW = beta1 * mW + (1 - beta1) * dW
                vU = beta2 * vU + (1 - beta2) * (dU ** 2)
                vV = beta2 * vV + (1 - beta2) * (dV ** 2)
                vW = beta2 * vW + (1 - beta2) * (dW ** 2)
                mU_hat = mU / (1 - beta1 ** t)
                mV_hat = mV / (1 - beta1 ** t)
                mW_hat = mW / (1 - beta1 ** t)
                vU_hat = vU / (1 - beta2 ** t)
                vV_hat = vV / (1 - beta2 ** t)
                vW_hat = vW / (1 - beta2 ** t)
                
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                # Track best
                if iteration % 20 == 0:
                    err = compute_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Check if we've found a valid solution
                if verify(U, V, W, rank):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    found = True
                    break
            if found:
                break
        
        # If we found a solution at this rank, try to go lower
        # But if not found, continue trying lower ranks
        # Actually, let's just break when we find one since lower is better
        if found:
            continue  # Try even lower ranks
    
    return best_result
# EVOLVE-BLOCK-END
