"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 145  # safety margin under the 150s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _to_halfint(Mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * Mat) / 2.0

def _compute_error(U, V, W, n=N, m=M, p=P):
    """Compute max error of decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))

def _structured_init(rank, seed, n=N, m=M, p=P):
    """Create structured initial guess with symmetry."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        # Random but structured entries
        u_vals = rng.randn(n, m) * 0.5
        v_vals = rng.randn(m, p) * 0.5
        w_vals = rng.randn(p, n) * 0.5
        
        U[:, r] = u_vals.flatten()
        V[:, r] = v_vals.flatten()
        W[:, r] = w_vals.flatten()
    
    return U, V, W

def _gradient_step(U, V, W, lr, n=N, m=M, p=P):
    """One gradient descent step on the squared error objective."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    err = T - 1.0  # gradient w.r.t. T
    
    # dL/dU3[i,j,r] = err[i,j,k] * V3[j,k,r] * W3[k,i,r] summed over k
    dU = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
    dV = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
    dW = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
    
    U -= lr * dU.flatten()
    V -= lr * dV.flatten()
    W -= lr * dW.flatten()
    
    return U, V, W

def _local_search_halfint(U, V, W, n=N, m=M, p=P, max_iter=2000, time_budget=10.0):
    """Local search on half-integer lattice to find exact solution."""
    best_err = np.inf
    best = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        Ur = _to_halfint(U)
        Vr = _to_halfint(V)
        Wr = _to_halfint(W)
        
        err = _compute_error(Ur, Vr, Wr, n, m, p)
        if err < 1e-9:
            return Ur, Vr, Wr
        
        if err < best_err:
            best_err = err
            best = (Ur.copy(), Vr.copy(), Wr.copy())
        
        # Perturb a random entry
        choice = np.random.randint(3)
        if choice == 0:
            idx = (np.random.randint(n * m), np.random.randint(U.shape[1]))
            U[idx] += np.random.choice([-0.5, 0.5])
        elif choice == 1:
            idx = (np.random.randint(m * p), np.random.randint(V.shape[1]))
            V[idx] += np.random.choice([-0.5, 0.5])
        else:
            idx = (np.random.randint(p * n), np.random.randint(W.shape[1]))
            W[idx] += np.random.choice([-0.5, 0.5])
    
    if best_err < 1e-9:
        return best
    return None

def _try_rank(rank, seed, time_budget, n=N, m=M, p=P):
    """Try to find an exact half-integer decomposition of given rank."""
    if time.time() > time_budget:
        return None
    
    np.random.seed(seed)
    
    # Structured initialization
    U, V, W = _structured_init(rank, seed, n, m, p)
    
    # Normalize columns
    for r in range(rank):
        nU = np.linalg.norm(U[:, r])
        nV = np.linalg.norm(V[:, r])
        nW = np.linalg.norm(W[:, r])
        if nU < 1e-12 or nV < 1e-12 or nW < 1e-12:
            continue
        scale = (nU * nV * nW) ** (1.0 / 3.0)
        U[:, r] = U[:, r] / nU * scale
        V[:, r] = V[:, r] / nV * scale
        W[:, r] = W[:, r] / nW * scale
    
    # Gradient descent phase
    lr = 0.01
    min_lr = 1e-6
    for it in range(500):
        if time.time() > time_budget:
            break
        
        err = _compute_error(U, V, W, n, m, p)
        if err < 1e-12:
            break
        
        U, V, W = _gradient_step(U, V, W, lr, n, m, p)
        
        # Adaptive learning rate
        if it % 50 == 49:
            err_new = _compute_error(U, V, W, n, m, p)
            if err_new > err * 1.1:
                lr *= 0.5
            else:
                lr = min(lr * 1.1, 0.05)
    
    # Half-integer local search
    result = _local_search_halfint(U, V, W, n, m, p, max_iter=3000, time_budget=time_budget)
    if result is not None:
        Ur, Vr, Wr = result
        if _verify(Ur, Vr, Wr, n, m, p):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    
    return None

def _try_alternating_least_squares(rank, seed, time_budget, n=N, m=M, p=P):
    """Alternating least squares approach."""
    if time.time() > time_budget:
        return None
    
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.3
    V = np.random.randn(m * p, rank) * 0.3
    W = np.random.randn(p * n, rank) * 0.3
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for outer in range(100):
        if time.time() > time_budget:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed i,j: sum_k [sum_r U[i,j,r] * (V[j,k,r]*W[k,i,r])] = 1 for all k
                # Let A[k,r] = V[j,k,r]*W[k,i,r], then sum_r U[i,j,r]*A[k,r] = 1
                A = np.einsum('kr,ir->kr', V3[j, :, :], W3[:, i, :])  # (p, rank)
                # Solve: A.T @ u = ones(p)
                try:
                    u = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    U3[i, j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed j,k: sum_i [sum_r V[j,k,r] * (U[i,j,r]*W[k,i,r])] = 1
                B = np.einsum('ir,ki->ir', U3[:, j, :], W3[k, :, :])  # (n, rank)
                try:
                    v = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    V3[j, k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed k,i: sum_j [sum_r W[k,i,r] * (U[i,j,r]*V[j,k,r])] = 1
                C = np.einsum('jr,jk->jr', U3[i, :, :], V3[:, k, :])  # (m, rank)
                try:
                    w = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    W3[k, i, :] = w
                except:
                    pass
        
        U = U3.flatten()
        V = V3.flatten()
        W = W3.flatten()
        
        err = _compute_error(U, V, W, n, m, p)
        if err < 1e-12:
            break
    
    # Half-integer local search
    result = _local_search_halfint(U, V, W, n, m, p, max_iter=3000, time_budget=time_budget)
    if result is not None:
        Ur, Vr, Wr = result
        if _verify(Ur, Vr, Wr, n, m, p):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    
    return None

def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    
    # Schoolbook fallback (always valid)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = 20
    # Try ranks from 20 upward
    for rank in range(20, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break
        
        # Multiple seeds per rank
        for seed in range(30):
            if time.time() > DEADLINE - 5:
                break
            
            # Try gradient-based approach
            result = _try_rank(rank, seed, DEADLINE - 3, n, m, p)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
            
            # Try ALS approach
            if time.time() < DEADLINE - 5:
                result = _try_alternating_least_squares(rank, seed, DEADLINE - 3, n, m, p)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
    
    return best_result

# EVOLVE-BLOCK-END
