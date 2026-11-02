"""Combined program: finite-field search + gradient descent polish for 3x3 x 3x4 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
FIELD = 13  # Small prime field for discrete search

def _verify_decomposition(U, V, W, n=3, m=3, p=4):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _loss_and_grad(U, V, W, n=3, m=3, p=4):
    """
    Compute loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    and analytical gradients for U, V, W.
    """
    rank = U.shape[1]
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i, j, k] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
    
    loss = np.sum((S - 1.0) ** 2)
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err = 2.0 * (S[i, j, k] - 1.0)
                grad_U[i * m + j] += err * V[j * p + k] * W[k * n + i]
                grad_V[j * p + k] += err * U[i * m + j] * W[k * n + i]
                grad_W[k * n + i] += err * U[i * m + j] * V[j * p + k]
    
    return loss, grad_U, grad_V, grad_W

def _structured_init(n, m, p, rank, seed=42):
    """
    Initialize U, V, W with a structured pattern that partially satisfies the constraints.
    Uses a block structure based on the (i,j,k) tensor indices.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Assign each (i,j,k) triple to a column with coefficient 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    # Add small random perturbations to break symmetry
    U += 0.01 * rng.randn(n * m, rank)
    V += 0.01 * rng.randn(m * p, rank)
    W += 0.01 * rng.randn(p * n, rank)
    
    return U, V, W

def _gradient_descent(n, m, p, rank, max_iter=5000, lr=0.05, tol=1e-12, timeout=60.0, seed=42):
    """
    Run gradient descent with analytical gradients to find a rank-r decomposition.
    """
    start_time = time.time()
    U, V, W = _structured_init(n, m, p, rank, seed=seed)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_lr = lr
    
    for iteration in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        U -= current_lr * gU
        V -= current_lr * gV
        W -= current_lr * gW
        
        new_loss, _, _, _ = _loss_and_grad(U, V, W, n, m, p)
        if new_loss > loss * 1.5:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            current_lr *= 0.5
            if current_lr < 1e-6:
                break
        else:
            if new_loss < loss * 0.9:
                current_lr = min(current_lr * 1.1, 0.1)
    
    return best_U, best_V, best_W, best_loss

def _try_rank(n, m, p, rank, timeout=45.0, num_seeds=5):
    """Try to find a valid rank-r decomposition."""
    start_time = time.time()
    
    for seed in range(num_seeds):
        if time.time() - start_time > timeout:
            break
        
        U, V, W, loss = _gradient_descent(n, m, p, rank, 
                                           max_iter=3000, lr=0.05, 
                                           tol=1e-12, timeout=timeout, seed=seed)
        
        if loss < 1e-8:
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _finite_field_search(n, m, p, rank, timeout=30.0, seed=42):
    """
    Search for a rank-r decomposition over F_13 using randomized local search.
    Returns integer entries in [0, 12] if found, else None.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with random values in F_13
    U = rng.randint(0, FIELD, size=(n * m, rank))
    V = rng.randint(0, FIELD, size=(m * p, rank))
    W = rng.randint(0, FIELD, size=(p * n, rank))
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = FIELD
    
    for iteration in range(20000):
        if time.time() - start_time > timeout:
            break
        
        # Compute error: count of (i,j,k) where sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] != 1 mod FIELD
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s = (s + U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]) % FIELD
                    if s != 1:
                        err += 1
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err == 0:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        # Pick a random variable to update
        mat = rng.randint(0, 3)
        if mat == 0:
            row = rng.randint(0, n * m)
            col = rng.randint(0, rank)
            old_val = U[row, col]
            U[row, col] = (old_val + 1) % FIELD
            i, j = divmod(row, m)
            new_err = 0
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = (s + U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]) % FIELD
                if s != 1:
                    new_err += 1
            if new_err >= err:
                U[row, col] = old_val
            else:
                err = new_err
        elif mat == 1:
            row = rng.randint(0, m * p)
            col = rng.randint(0, rank)
            old_val = V[row, col]
            V[row, col] = (old_val + 1) % FIELD
            j, k = divmod(row, p)
            new_err = 0
            for i in range(n):
                s = 0
                for c in range(rank):
                    s = (s + U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]) % FIELD
                if s != 1:
                    new_err += 1
            if new_err >= err:
                V[row, col] = old_val
            else:
                err = new_err
        else:
            row = rng.randint(0, p * n)
            col = rng.randint(0, rank)
            old_val = W[row, col]
            W[row, col] = (old_val + 1) % FIELD
            k, i = divmod(row, n)
            new_err = 0
            for j in range(m):
                s = 0
                for c in range(rank):
                    s = (s + U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]) % FIELD
                if s != 1:
                    new_err += 1
            if new_err >= err:
                W[row, col] = old_val
            else:
                err = new_err
    
    if best_err == 0:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _lift_from_field(U_mod, V_mod, W_mod, n=3, m=3, p=4):
    """
    Convert a solution over F_13 to integer/half-integer entries.
    Map values in [0, 12] to integers in [-6, 6] (centered representation).
    """
    rank = len(U_mod[0])
    U = [[(x - FIELD // 2) for x in row] for row in U_mod]
    V = [[(x - FIELD // 2) for x in row] for row in V_mod]
    W = [[(x - FIELD // 2) for x in row] for row in W_mod]
    return U, V, W

def _polish_from_field(U_int, V_int, W_int, n=3, m=3, p=4, timeout=15.0):
    """
    Take an integer solution from finite field and polish it with gradient descent
    to find a nearby exact real solution.
    """
    rank = len(U_int[0])
    U = np.array(U_int, dtype=float)
    V = np.array(V_int, dtype=float)
    W = np.array(W_int, dtype=float)
    
    # Use gradient descent starting from this initialization
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_lr = 0.01
    start_time = time.time()
    
    for iteration in range(5000):
        if time.time() - start_time > timeout:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        U -= current_lr * gU
        V -= current_lr * gV
        W -= current_lr * gW
        
        new_loss, _, _, _ = _loss_and_grad(U, V, W, n, m, p)
        if new_loss > loss * 1.1:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            current_lr *= 0.5
            if current_lr < 1e-8:
                break
        else:
            if new_loss < loss * 0.9:
                current_lr = min(current_lr * 1.05, 0.05)
    
    return best_U, best_V, best_W, best_loss

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find lower-rank decompositions using combined finite-field + gradient descent
    for rank in range(best_rank - 1, 22, -1):
        if time.time() > deadline - 25:
            break
        
        # Strategy 1: Finite field search
        ff_result = _finite_field_search(n, m, p, rank, timeout=20.0, seed=rank)
        if ff_result is not None:
            U_int, V_int, W_int = ff_result
            U_lift, V_lift, W_lift = _lift_from_field(U_int, V_int, W_int, n, m, p)
            
            # Verify the lifted solution
            if _verify_decomposition(U_lift, V_lift, W_lift, n, m, p):
                best = {"U": U_lift, "V": V_lift, "W": W_lift}
                best_rank = rank
                continue
            
            # Try to polish the lifted solution
            U_pol, V_pol, W_pol, loss = _polish_from_field(U_lift, V_lift, W_lift, n, m, p, timeout=10.0)
            if loss < 1e-8 and _verify_decomposition(U_pol.tolist(), V_pol.tolist(), W_pol.tolist(), n, m, p):
                best = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}
                best_rank = rank
                continue
        
        # Strategy 2: Standard gradient descent
        if time.time() > deadline - 25:
            break
        
        result = _try_rank(n, m, p, rank, timeout=20.0, num_seeds=3)
        if result is not None:
            best = result
            best_rank = rank
    
    return best
