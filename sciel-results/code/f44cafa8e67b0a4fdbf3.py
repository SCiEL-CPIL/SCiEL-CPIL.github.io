import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin for safety
    
    # Known bounds for 3x5x6: lower ~24, try from there
    # Theoretical lower bound is max(n, m, p) * min(n, m, p) roughly
    # For 3x5x6, let's try ranks from 20 to 40 first, then expand
    
    def try_rank(rank, num_trials=15, max_steps=2000):
        """Try to find a decomposition at given rank using gradient descent."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(rank * 7919 + trial * 104729)
            
            # Smart initialization: start with structured random
            U = rng.randn(nm, rank) * 0.3
            V = rng.randn(mp, rank) * 0.3
            W = rng.randn(pn, rank) * 0.3
            
            # Add some structure: ensure each row of T is somewhat represented
            for t in range(min(10, rank)):
                idx_a = rng.randint(nm)
                idx_b = rng.randint(mp)
                idx_c = rng.randint(pn)
                U[idx_a, t] += 1.0
                V[idx_b, t] += 1.0
                W[idx_c, t] += 1.0
            
            lr = 0.01
            momentum_U = np.zeros_like(U)
            momentum_V = np.zeros_like(V)
            momentum_W = np.zeros_like(W)
            beta = 0.9
            
            prev_loss = np.inf
            no_improve_count = 0
            
            for step in range(max_steps):
                if time.time() > deadline:
                    return None
                
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = T - recon
                loss = np.sum(err**2)
                
                if loss < 1e-12:
                    return (U, V, W)
                
                if loss < prev_loss * 0.999:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count > 200:
                    break
                
                prev_loss = loss
                
                grad_U = np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = np.einsum('abc,ar,br->cr', err, U, V)
                
                momentum_U = beta * momentum_U + (1 - beta) * grad_U
                momentum_V = beta * momentum_V + (1 - beta) * grad_V
                momentum_W = beta * momentum_W + (1 - beta) * grad_W
                
                U -= lr * momentum_U
                V -= lr * momentum_V
                W -= lr * momentum_W
                
                # Adaptive learning rate
                if step % 50 == 0:
                    if loss < prev_loss * 0.5:
                        lr *= 1.1
                    else:
                        lr *= 0.9
                    lr = min(lr, 0.05)
                    lr = max(lr, 0.001)
            
            # Check final result
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T - recon)**2)
            if loss < 1e-8:
                return (U, V, W)
        
        return None
    
    def polish(U, V, W, max_steps=500):
        """High-precision polish pass."""
        lr = 0.005
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = T - recon
            loss = np.sum(err**2)
            
            if loss < 1e-14:
                break
            
            grad_U = np.einsum('abc,br,cr->ar', err, V, W)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if step % 100 == 0:
                lr *= 0.8
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T - recon)) < 1e-9:
            return True
        return False
    
    def verify_and_round(U, V, W):
        """Round to half-integers and verify."""
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        recon_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
        if np.max(np.abs(T - recon_round)) < 1e-9:
            return (U_round, V_round, W_round)
        
        # Try unrounded if close enough
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T - recon)) < 1e-9:
            return (U, V, W)
        
        return None
    
    # Try ranks in a strategic order
    # For 3x5x6, known results suggest rank around 24-30
    rank_candidates = list(range(20, 45))
    rank_candidates.sort(key=lambda r: abs(r - 27))  # Start near expected rank
    
    for rank in rank_candidates:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
        
        result = try_rank(rank, num_trials=12, max_steps=1500)
        
        if result is not None:
            U, V, W = result
            
            # Polish
            if polish(U, V, W, max_steps=300):
                verified = verify_and_round(U, V, W)
                if verified is not None:
                    best_U, best_V, best_W = verified
                    best_rank = rank
                    # Continue searching for better ranks
                    continue
            
            # Even if polish didn't fully converge, check if it's close
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(T - recon)) < 1e-7:
                verified = verify_and_round(U, V, W)
                if verified is not None:
                    best_U, best_V, best_W = verified
                    best_rank = rank
    
    # If we haven't found anything better, try a few more ranks
    if best_rank == rank_school:
        for rank in range(45, min(60, rank_school + 1)):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, num_trials=8, max_steps=1000)
            if result is not None:
                U, V, W = result
                if polish(U, V, W, max_steps=200):
                    verified = verify_and_round(U, V, W)
                    if verified is not None:
                        best_U, best_V, best_W = verified
                        best_rank = rank
                        break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
