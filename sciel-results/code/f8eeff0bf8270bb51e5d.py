import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using Fraction arithmetic."""
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                s = Fraction(0)
                for r in range(rank):
                    # Convert to Fraction for exact check
                    u_val = U[a][r] if isinstance(U[a][r], (int, float, Fraction)) else Fraction(U[a][r])
                    v_val = V[b][r] if isinstance(V[b][r], (int, float, Fraction)) else Fraction(V[b][r])
                    w_val = W[c][r] if isinstance(W[c][r], (int, float, Fraction)) else Fraction(W[c][r])
                    
                    # If they are floats, try to snap to fractions
                    if isinstance(u_val, float):
                        u_frac = Fraction(u_val).limit_denominator(1000000)
                    else:
                        u_frac = u_val if isinstance(u_val, Fraction) else Fraction(u_val)
                    if isinstance(v_val, float):
                        v_frac = Fraction(v_val).limit_denominator(1000000)
                    else:
                        v_frac = v_val if isinstance(v_val, Fraction) else Fraction(v_val)
                    if isinstance(w_val, float):
                        w_frac = Fraction(w_val).limit_denominator(1000000)
                    else:
                        w_frac = w_val if isinstance(w_val, Fraction) else Fraction(w_val)
                    
                    s += u_frac * v_frac * w_frac
                
                if s != target.get((a, b, c), 0):
                    return False
    return True

def _verify_numeric(U, V, W, n, m, p, rank, tol=1e-9):
    """Fast numeric verification."""
    target = _target_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(s - target.get((a, b, c), 0)) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _try_find_decomposition(target_rank, n, m, p, deadline, seed=0):
    """
    Try to find a rank-r decomposition using numerical optimization with
    multiple strategies: random init, structured init, and local refinement.
    """
    np.random.seed(seed)
    
    nm = n*m
    mp = m*p
    pn = p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T[a, b, c] = 1.0
    
    def compute_T_rec(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def loss(U, V, W):
        err = compute_T_rec(U, V, W) - T
        return np.sum(err**2)
    
    # Multiple initialization strategies
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    t0 = time.time()
    
    # Strategy 1: Random initialization with Adam
    for attempt in range(5):
        if time.time() - t0 > deadline - 10:
            break
        
        # Different scales for initialization
        scale = [0.5, 1.0, 0.1, 2.0, 0.01][attempt % 5]
        U = np.random.randn(nm, target_rank) * scale
        V = np.random.randn(mp, target_rank) * scale
        W = np.random.randn(pn, target_rank) * scale
        
        # Adam optimizer
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        max_iters = 3000
        cur_loss = loss(U, V, W)
        
        while t < max_iters and (time.time() - t0) < deadline - 5:
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            err = compute_T_rec(U, V, W) - T
            
            gU = np.einsum('abc,br,cr->ar', err, V, W)
            gV = np.einsum('abc,ar,cr->br', err, U, W)
            gW = np.einsum('abc,ar,br->cr', err, U, V)
            
            m_U = beta1*m_U + (1-beta1)*gU
            v_U = beta2*v_U + (1-beta2)*gU**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U -= lr_t * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1*m_V + (1-beta1)*gV
            v_V = beta2*v_V + (1-beta2)*gV**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V -= lr_t * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1*m_W + (1-beta1)*gW
            v_W = beta2*v_W + (1-beta2)*gW**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W -= lr_t * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            cur_loss = loss(U, V, W)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if cur_loss < 1e-12:
                break
        
        if best_loss < 1e-8:
            break
    
    # Strategy 2: Structured initialization (block-diagonal inspired)
    if best_loss > 1e-8 and (time.time() - t0) < deadline - 10:
        # Initialize with blocks that partially satisfy the tensor
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Try to place ones in structured positions
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if idx < target_rank:
                        U[i*m+j, idx] = 1.0
                        V[j*p+k, idx] = 1.0
                        W[k*n+i, idx] = 1.0
                        idx += 1
        
        # Add random perturbation
        U += np.random.randn(nm, target_rank) * 0.1
        V += np.random.randn(mp, target_rank) * 0.1
        W += np.random.randn(pn, target_rank) * 0.1
        
        # Refine with Adam
        lr = 0.02
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0
        max_iters = 2000
        cur_loss = loss(U, V, W)
        
        while t < max_iters and (time.time() - t0) < deadline - 5:
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            err = compute_T_rec(U, V, W) - T
            gU = np.einsum('abc,br,cr->ar', err, V, W)
            gV = np.einsum('abc,ar,cr->br', err, U, W)
            gW = np.einsum('abc,ar,br->cr', err, U, V)
            
            m_U = beta1*m_U + (1-beta1)*gU
            v_U = beta2*v_U + (1-beta2)*gU**2
            U -= lr_t * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)
            
            m_V = beta1*m_V + (1-beta1)*gV
            v_V = beta2*v_V + (1-beta2)*gV**2
            V -= lr_t * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
            
            m_W = beta1*m_W + (1-beta1)*gW
            v_W = beta2*v_W + (1-beta2)*gW**2
            W -= lr_t * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
            
            cur_loss = loss(U, V, W)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if cur_loss < 1e-12:
                break
    
    if best_U is None or best_loss > 1e-6:
        return None, None, None, best_loss
    
    # Try to snap to rational values
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Attempt to find exact rational representation
    def try_snap(arr, denom_candidates=[1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20]):
        snapped = []
        for row in arr:
            snapped_row = []
            for val in row:
                best_frac = Fraction(val).limit_denominator(100000)
                snapped_row.append(best_frac)
            snapped.append(snapped_row)
        return snapped
    
    # Check if the current solution is already close to rational
    U_frac = try_snap(best_U)
    V_frac = try_snap(best_V)
    W_frac = try_snap(best_W)
    
    # Verify with fractions
    if _verify(U_frac, V_frac, W_frac, n, m, p, target_rank):
        return U_frac, V_frac, W_frac, 0.0
    
    # If not exact, try a few more refinement steps
    # Convert back to float for more refinement
    U_f = np.array(U_frac, dtype=float)
    V_f = np.array(V_frac, dtype=float)
    W_f = np.array(W_frac, dtype=float)
    
    # Additional refinement
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-10
    m_U = np.zeros_like(U_f)
    v_U = np.zeros_like(U_f)
    m_V = np.zeros_like(V_f)
    v_V = np.zeros_like(V_f)
    m_W = np.zeros_like(W_f)
    v_W = np.zeros_like(W_f)
    t = 0
    max_iters = 1000
    
    while t < max_iters and (time.time() - t0) < deadline - 2:
        t += 1
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        err = compute_T_rec(U_f, V_f, W_f) - T
        gU = np.einsum('abc,br,cr->ar', err, V_f, W_f)
        gV = np.einsum('abc,ar,cr->br', err, U_f, W_f)
        gW = np.einsum('abc,ar,br->cr', err, U_f, V_f)
        
        m_U = beta1*m_U + (1-beta1)*gU
        v_U = beta2*v_U + (1-beta2)*gU**2
        U_f -= lr_t * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)
        
        m_V = beta1*m_V + (1-beta1)*gV
        v_V = beta2*v_V + (1-beta2)*gV**2
        V_f -= lr_t * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
        
        m_W = beta1*m_W + (1-beta1)*gW
        v_W = beta2*v_W + (1-beta2)*gW**2
        W_f -= lr_t * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
        
        cur_loss = loss(U_f, V_f, W_f)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
        
        if cur_loss < 1e-14:
            break
    
    # Final snap attempt
    U_frac2 = try_snap(best_U)
    V_frac2 = try_snap(best_V)
    W_frac2 = try_snap(best_W)
    
    if _verify(U_frac2, V_frac2, W_frac2, n, m, p, target_rank):
        return U_frac2, V_frac2, W_frac2, 0.0
    
    # Return the best numeric solution
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_loss

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 140.0  # Safety margin from 150s limit
    
    # Start with schoolbook (rank 36)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = (U_school, V_school, W_school, n*m*p)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound for 3x3x4: max(3*3, 3*4, 3*4) = 12
    # But practical bounds suggest higher. Try ranks from 24 down.
    
    for target_rank in [24, 22, 20, 18, 16, 14, 12]:
        if time.time() - t0 > DEADLINE - 15:
            break
        
        found = False
        for seed in range(4):
            if time.time() - t0 > DEADLINE - 10:
                break
            
            U, V, W, loss_val = _try_find_decomposition(
                target_rank, n, m, p,
                deadline=time.time() - t0 + (DEADLINE - 8),
                seed=seed
            )
            
            if U is not None and loss_val < 1e-6:
                # Verify numerically first (fast)
                if _verify_numeric(U, V, W, n, m, p, target_rank, tol=1e-8):
                    # Then verify exactly
                    if _verify(U, V, W, n, m, p, target_rank):
                        best_result = (U, V, W, target_rank)
                        found = True
                        break
        
        if found:
            continue  # Try lower rank
        else:
            # If we couldn't find at this rank, try next lower
            continue
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
