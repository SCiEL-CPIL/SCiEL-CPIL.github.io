import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T[a, b, c] where a=k*n+i, b=i*m+j, c=j*p+k
    # Note: The task contract says output factor W[k*n+i, r].
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The tensor T_{b, c, a} = sum_r U_{b, r} V_{c, r} W_{a, r}
    # Let's verify indices:
    # U index: i*m+j (b)
    # V index: j*p+k (c)
    # W index: k*n+i (a)
    # Target T[b, c, a] = 1 if valid (i,j,k) else 0.
    # Actually, the standard tensor T_{i,j,k} = 1.
    # T_decomp[b, c, a] = sum U[b,r] V[c,r] W[a,r]
    
    # Pre-compute the non-zero entries of the target tensor for faster verification
    # and potentially for structured initialization or constraint enforcement.
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b = i * m + j
                c = j * p + k
                a = k * n + i
                target_entries.append((b, c, a))
                
    T_target = np.zeros((nm, mp, pn))
    for b, c, a in target_entries:
        T_target[b, c, a] = 1.0
        
    start_time = time.time()
    deadline = start_time + 115  # 15s safety margin
    
    # Schoolbook decomposition (Rank 105)
    def schoolbook():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_r = U_sb, V_sb, W_sb, r_sb
    
    def verify(U, V, W):
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        # Check exactness using the pre-computed entries for speed and precision
        # T_decomp[b, c, a] = sum_r U[b,r] V[c,r] W[a,r]
        for b, c, a in target_entries:
            val = np.dot(U[b, :], V[c, :] * W[a, :])
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def try_rank(r, time_limit):
        if time.time() > time_limit:
            return None
        
        # We will use a hybrid approach:
        # 1. Random initialization
        # 2. Alternating Least Squares (ALS) or Gradient Descent
        # 3. Since the target is sparse and binary, we can try to snap to nearest half-integers
        #    and verify.
        
        # Number of restarts depends on time available
        # Let's do a few high-quality restarts with more iterations
        num_restarts = 20
        max_iter = 300
        
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            np.random.seed(restart * 1000 + r)
            
            # Initialize with small random values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Gradient Descent with adaptive learning rate
            lr = 0.05
            loss_history = []
            
            for it in range(max_iter):
                if time.time() > time_limit:
                    break
                
                # Forward pass
                # Efficient calculation of loss and gradients
                # E[b, c, a] = sum_r U[b,r] V[c,r] W[a,r] - T_target[b, c, a]
                
                # Instead of full tensor multiplication which is O(nm*mp*pn*r),
                # we can exploit sparsity.
                # But for small dimensions (15, 35, 21), full tensor is fine.
                # nm=15, mp=35, pn=21. Total size 11025.
                
                T_decomp = np.einsum('br,cr,ar->bca', U, V, W)
                E = T_decomp - T_target
                loss = np.sum(E**2)
                
                if loss < 1e-10:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    break
                
                # Gradients
                # dL/dU[b,r] = 2 * sum_{c,a} E[b,c,a] V[c,r] W[a,r]
                # dL/dV[c,r] = 2 * sum_{b,a} E[b,c,a] U[b,r] W[a,r]
                # dL/dW[a,r] = 2 * sum_{b,c} E[b,c,a] U[b,r] V[c,r]
                
                grad_U = 2 * np.einsum('bca,cr,ar->br', E, V, W)
                grad_V = 2 * np.einsum('bca,br,ar->cr', E, U, W)
                grad_W = 2 * np.einsum('bca,br,cr->ar', E, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if len(loss_history) > 1:
                    if loss > loss_history[-1] * 1.1:
                        lr *= 0.5
                    elif it % 20 == 0:
                        lr *= 0.95
                loss_history.append(loss)
                
                # Early exit if loss is stuck
                if it > 50 and abs(loss_history[-1] - loss_history[-2]) < 1e-12:
                    break
        
        return None

    # Search for lower ranks
    # Start from r_sb - 1 down to a lower bound.
    # Lower bound for matrix multiplication rank is roughly max(nm, mp, pn) = 35.
    # But for 3x5x7, known bounds might be higher.
    # Let's try down to 80 to be safe with time.
    
    current_deadline = start_time + 110
    
    for r in range(r_sb - 1, 79, -1):
        if time.time() > current_deadline - 10:
            break
        
        # Allocate time based on rank
        # Higher ranks are easier to find if they exist, lower are harder.
        # But we are going down.
        # Give 2 seconds per rank initially, increasing for lower ranks?
        # Let's just use a fixed small budget per rank to ensure we scan many.
        time_per_rank = 2.0
        if r < 90:
            time_per_rank = 3.0
            
        res = try_rank(r, time.time() + time_per_rank)
        if res is not None:
            best_U, best_V, best_W, best_r = res
            # If we find one, do we continue to find lower?
            # The task asks for rank-minimal. So yes, continue.
            # But if we find one, it's a big win.
            # Let's continue searching for lower ranks.
            continue
            
    # If we found a better rank, return it.
    # Note: The loop above updates best_U, best_V, best_W if found.
    # If multiple are found, the last one (lowest r) is kept because we go down.
    # Wait, if r=104 fails, r=103 fails... r=90 succeeds. best_r=90.
    # Then r=89... if succeeds, best_r=89.
    # So the final best is the lowest found.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
