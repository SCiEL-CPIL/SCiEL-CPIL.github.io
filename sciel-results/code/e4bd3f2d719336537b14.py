import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    # Precompute index arrays for fast evaluation
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need this to equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    def make_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_numpy(U, V, W):
        """Fast verification using numpy. Returns True if exact decomposition."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def sse_numpy(U, V, W):
        """Fast SSE computation."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        sse = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    sse += (s - 1.0) ** 2
        return sse
    
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Strategy: Try structured decompositions based on algebraic patterns
    # For 4x4 * 4x5, we know rank 20 is achievable via block methods.
    # Let's try to find rank < 20.
    
    vals = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    def local_search_numpy(rank, U, V, W, seed, max_steps=3000):
        """Fast local search using numpy for SSE evaluation."""
        rng = random.Random(seed)
        U_ = np.array(U, dtype=float)
        V_ = np.array(V, dtype=float)
        W_ = np.array(W, dtype=float)
        
        cur_sse = sse_numpy(U_, V_, W_)
        if cur_sse < 1e-9:
            return {"U": U_.tolist(), "V": V_.tolist(), "W": W_.tolist()}
        
        best_sse = cur_sse
        no_improve = 0
        step = 0
        
        while step < max_steps:
            if time.time() > deadline - 5:
                break
            if cur_sse < 1e-9:
                break
            step += 1
            
            # Pick a random factor and entry
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_[idx, c]
                best_val = old
                best_move_sse = cur_sse
                for v in vals:
                    if v == old: continue
                    U_[idx, c] = v
                    new_sse = sse_numpy(U_, V_, W_)
                    if new_sse < best_move_sse:
                        best_move_sse = new_sse
                        best_val = v
                U_[idx, c] = best_val
                if best_val != old:
                    cur_sse = best_move_sse
                    if cur_sse < best_sse - 1e-12:
                        best_sse = cur_sse
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_[idx, c]
                best_val = old
                best_move_sse = cur_sse
                for v in vals:
                    if v == old: continue
                    V_[idx, c] = v
                    new_sse = sse_numpy(U_, V_, W_)
                    if new_sse < best_move_sse:
                        best_move_sse = new_sse
                        best_val = v
                V_[idx, c] = best_val
                if best_val != old:
                    cur_sse = best_move_sse
                    if cur_sse < best_sse - 1e-12:
                        best_sse = cur_sse
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_[idx, c]
                best_val = old
                best_move_sse = cur_sse
                for v in vals:
                    if v == old: continue
                    W_[idx, c] = v
                    new_sse = sse_numpy(U_, V_, W_)
                    if new_sse < best_move_sse:
                        best_move_sse = new_sse
                        best_val = v
                W_[idx, c] = best_val
                if best_val != old:
                    cur_sse = best_move_sse
                    if cur_sse < best_sse - 1e-12:
                        best_sse = cur_sse
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 300:
                # Random jump
                for _ in range(5):
                    f = rng.randrange(3)
                    if f == 0:
                        idx = rng.randrange(n*m)
                        c = rng.randrange(rank)
                        U_[idx, c] = rng.choice(vals)
                    elif f == 1:
                        idx = rng.randrange(m*p)
                        c = rng.randrange(rank)
                        V_[idx, c] = rng.choice(vals)
                    else:
                        idx = rng.randrange(p*n)
                        c = rng.randrange(rank)
                        W_[idx, c] = rng.choice(vals)
                cur_sse = sse_numpy(U_, V_, W_)
                no_improve = 0
        
        if cur_sse < 1e-9:
            return {"U": U_.tolist(), "V": V_.tolist(), "W": W_.tolist()}
        return None
    
    # Try ranks from 16 upward
    for rank in range(16, min(n*m*p, 40)):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        num_restarts = 15
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)
            
            # Different initialization strategies
            strategy = restart % 3
            
            if strategy == 0:
                # Sparse random
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for idx in range(n*m):
                    for c in range(rank):
                        if rng.random() < 0.3:
                            U[idx, c] = rng.choice(vals)
                for idx in range(m*p):
                    for c in range(rank):
                        if rng.random() < 0.3:
                            V[idx, c] = rng.choice(vals)
                for idx in range(p*n):
                    for c in range(rank):
                        if rng.random() < 0.3:
                            W[idx, c] = rng.choice(vals)
                
            elif strategy == 1:
                # Block-structured: assign each (i,j,k) to a random rank slot
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = rng.randrange(rank)
                            U[i*m+j, c] = 1
                            V[j*p+k, c] = 1
                            W[k*n+i, c] = 1
                # Add some noise
                for _ in range(20):
                    f = rng.randrange(3)
                    if f == 0:
                        idx = rng.randrange(n*m)
                        c = rng.randrange(rank)
                        U[idx, c] += rng.choice([-1, 1]) * 0.5
                    elif f == 1:
                        idx = rng.randrange(m*p)
                        c = rng.randrange(rank)
                        V[idx, c] += rng.choice([-1, 1]) * 0.5
                    else:
                        idx = rng.randrange(p*n)
                        c = rng.randrange(rank)
                        W[idx, c] += rng.choice([-1, 1]) * 0.5
                
            else:
                # Dense random with small values
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for idx in range(n*m):
                    for c in range(rank):
                        if rng.random() < 0.5:
                            U[idx, c] = rng.choice([0.5, -0.5, 1, -1])
                for idx in range(m*p):
                    for c in range(rank):
                        if rng.random() < 0.5:
                            V[idx, c] = rng.choice([0.5, -0.5, 1, -1])
                for idx in range(p*n):
                    for c in range(rank):
                        if rng.random() < 0.5:
                            W[idx, c] = rng.choice([0.5, -0.5, 1, -1])
            
            result = local_search_numpy(rank, U.tolist(), V.tolist(), W.tolist(), seed + 7777, max_steps=1500)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break
            
    return best
