"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix mult."""
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        """Vectorized verification."""
        rank = U.shape[1]
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        # For each (i,j,k): sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook (rank 36) decomposition."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def try_canonical_W(rank, seed, max_iters=8000):
        """
        Exploit invariance: fix W to a canonical sparse form.
        W has shape (pn, rank) = (12, rank).
        We fix W to have a block structure where each column has at most a few nonzeros.
        
        Strategy: Fix W such that for each output index (k,i), W[k*n+i, c] is 1 for 
        exactly one c (the "owner" of that output), and 0 otherwise. This makes the
        system: for each (i,j,k), U[i*m+j, owner(k,i)] * V[j*p+k, owner(k,i)] = 1.
        
        But this only gives rank >= pn = 12 if each (k,i) has a unique owner, which is 
        the schoolbook. Instead, allow some sharing: multiple (k,i) can share an owner c.
        
        Better approach: Fix W to be a fixed sparse pattern, then solve for U and V.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Fix W to a canonical form: each column of W has exactly one 1, rest 0.
        # This means W is a (pn x rank) matrix where each column has one 1.
        # The position of the 1 in column c determines which (k,i) "uses" that rank term.
        
        # We need: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # With W[k*n+i,c] = 1 if owner(k,i) == c, else 0:
        #   U[i*m+j, owner(k,i)] * V[j*p+k, owner(k,i)] = 1 for all (i,j,k)
        
        # This means: for each c, let S_c = {(i,k) : owner(k,i) == c}
        # Then for all (i,j,k) with owner(k,i) == c: U[i*m+j,c] * V[j*p+k,c] = 1
        # This requires U[i*m+j,c] = 1/V[j*p+k,c] for all j, which is only possible
        # if all V[j*p+k,c] are equal for fixed k, and all U[i*m+j,c] are equal for fixed j.
        # This is very restrictive and only works for schoolbook.
        
        # Better: Don't fix W to be 0/1. Instead, fix W to have a specific sparse pattern
        # with values from a small set, then solve for U and V via least squares.
        
        # Alternative: Fix U to canonical form and solve for V, W.
        # Or: Use the fact that we can apply invertible transformations.
        
        # Let's try: fix W to be the first pn columns of identity (if rank >= pn),
        # then the remaining columns are free. But this is just schoolbook + extras.
        
        # Most promising: random sparse W with small integer entries, then solve for U, V.
        
        # Initialize W with random sparse pattern
        W = np.zeros((pn, rank))
        for c in range(rank):
            # Each column of W has 1-3 nonzeros
            nnz = np.random.randint(1, 4)
            positions = np.random.choice(pn, nnz, replace=False)
            for pos in positions:
                W[pos, c] = np.random.choice([-1, 1])
        
        # Now solve for U and V: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is still non-linear in U and V. But if we fix one, it becomes linear.
        
        # Fix U randomly, solve for V linearly, then fix V, solve for U, iterate.
        
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        
        def loss_and_grad(U, V, W):
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_val = W[k*n+i]
                        val = np.dot(U[i*m+j] * w_val, V[j*p+k])
                        err = val - 1.0
                        loss += err * err
                        dU[i*m+j] += 2 * err * V[j*p+k] * w_val
                        dV[j*p+k] += 2 * err * U[i*m+j] * w_val
            return loss, dU, dV
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V = U.copy(), V.copy()
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            loss, dU, dV = loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V = U.copy(), V.copy()
            
            if best_loss < 1e-14:
                break
            
            if it % 200 == 0 and it > 0:
                if loss > best_loss * 2:
                    lr *= 0.5
                elif loss < best_loss * 0.3:
                    lr = min(lr * 1.5, 0.1)
            
            U -= lr * dU
            V -= lr * dV
        
        return best_loss, best_U, best_V, W
    
    def try_joint_optimization(rank, seed, max_iters=10000):
        """Joint optimization of U, V, W with better initialization."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with structure: start near schoolbook and perturb
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Place schoolbook terms in first nm*mp*p... no, just random init
        U = np.random.randn(nm, rank) * 0.3
        V = np.random.randn(mp, rank) * 0.3
        W = np.random.randn(pn, rank) * 0.3
        
        def loss_and_grad(U, V, W):
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i*m+j
                        idx_v = j*p+k
                        idx_w = k*n+i
                        val = 0.0
                        for c in range(rank):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        err = val - 1.0
                        loss += err * err
                        for c in range(rank):
                            uw = U[idx_u, c] * W[idx_w, c]
                            vw = V[idx_v, c] * W[idx_w, c]
                            uv = U[idx_u, c] * V[idx_v, c]
                            dU[idx_u, c] += 2 * err * vw
                            dV[idx_v, c] += 2 * err * uw
                            dW[idx_w, c] += 2 * err * uv
            return loss, dU, dV, dW
        
        lr = 0.005
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            loss, dU, dV, dW = loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-14:
                break
            
            if it % 300 == 0 and it > 0:
                if loss > best_loss * 2:
                    lr *= 0.5
                elif loss < best_loss * 0.3:
                    lr = min(lr * 1.3, 0.05)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return best_loss, best_U, best_V, best_W
    
    def try_strassen_based(rank, seed):
        """
        Try to construct using Strassen-like blocks.
        For 3x3*3x4, we can think of it as 3x3*3x3 + 3x3*3x1.
        Use known rank-23 for 3x3*3x3 (Lader) and add 3 for the extra column.
        But R(3,3,4) = 22, so there's a more efficient construction.
        
        Try: fix the structure based on Strassen's 7-term 2x2 and extend.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with a structured pattern
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Random sparse initialization with values from {-1, 0, 1}
        for c in range(rank):
            for r in range(nm):
                if np.random.random() < 0.3:
                    U[r, c] = np.random.choice([-1, 1])
            for r in range(mp):
                if np.random.random() < 0.3:
                    V[r, c] = np.random.choice([-1, 1])
            for r in range(pn):
                if np.random.random() < 0.3:
                    W[r, c] = np.random.choice([-1, 1])
        
        def loss_and_grad(U, V, W):
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i*m+j
                        idx_v = j*p+k
                        idx_w = k*n+i
                        val = 0.0
                        for c in range(rank):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        err = val - 1.0
                        loss += err * err
                        for c in range(rank):
                            dU[idx_u, c] += 2 * err * V[idx_v, c] * W[idx_w, c]
                            dV[idx_v, c] += 2 * err * U[idx_u, c] * W[idx_w, c]
                            dW[idx_w, c] += 2 * err * U[idx_u, c] * V[idx_v, c]
            return loss, dU, dV, dW
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            loss, dU, dV, dW = loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-14:
                break
            
            if it % 200 == 0 and it > 0:
                if loss > best_loss * 2:
                    lr *= 0.5
                elif loss < best_loss * 0.3:
                    lr = min(lr * 1.5, 0.1)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return best_loss, best_U, best_V, best_W
    
    # Try to find minimal rank decomposition
    best_result = None
    best_rank = n * m * p  # schoolbook = 36
    
    # Try ranks from 22 upward
    for rank in range(22, min(37, n*m*p + 1)):
        if time.time() > DEADLINE - 30:
            break
        
        # Try multiple strategies for this rank
        for strategy_name, strategy_fn, n_attempts in [
            ("canonical_W", try_canonical_W, 15),
            ("joint", try_joint_optimization, 10),
            ("strassen", try_strassen_based, 10),
        ]:
            for attempt in range(n_attempts):
                if time.time() > DEADLINE - 10:
                    break
                
                seed = rank * 1000 + attempt * 7 + 13
                loss, U, V, W = strategy_fn(rank, seed)
                
                if loss < 1e-10:
                    if verify_numpy(U, V, W):
                        # Try to round to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if verify_numpy(U_r, V_r, W_r):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        # Try integer rounding
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        if verify_numpy(U_i, V_i, W_i):
                            return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                if best_result is None or loss < best_result[0]:
                    best_result = (loss, U, V, W, rank)
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
