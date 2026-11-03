"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for evaluator
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor for 2x3x6
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def optimize_233(max_steps=500, tol=1e-8):
        """Find rank 10 decomposition for 2x3x3."""
        n2, m2, p2 = 2, 3, 3
        nm2, mp2, pn2 = 6, 9, 6
        rank = 10
        T233 = np.zeros((nm2, mp2, pn2), dtype=np.float64)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i * m2 + j
                    b = j * p2 + k
                    c = k * n2 + i
                    T233[a, b, c] = 1.0
        
        best_loss = np.inf
        best_U, best_V, best_W = None, None, None
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(nm2, rank) * 0.5
            V = np.random.randn(mp2, rank) * 0.5
            W = np.random.randn(pn2, rank) * 0.5
            
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    break
                
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_rec - T233
                loss = np.sum(diff**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if np.max(np.abs(diff)) < tol:
                    break
                
                grad_U = np.einsum('abc,br,cr->ar', diff, V, W) * 2
                grad_V = np.einsum('abc,ar,cr->br', diff, U, W) * 2
                grad_W = np.einsum('abc,ar,br->cr', diff, U, V) * 2
                
                vU = momentum * vU + (1 - momentum) * grad_U
                vV = momentum * vV + (1 - momentum) * grad_V
                vW = momentum * vW + (1 - momentum) * grad_W
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
                # Simple adaptive LR
                if step % 50 == 0 and step > 0:
                    if loss < best_loss * 0.9:
                        lr *= 0.8
                    elif loss > best_loss * 1.1:
                        lr *= 1.5
                        lr = min(lr, 0.1)
            
            if best_loss < 1e-4:
                break
        
        if best_U is not None and best_loss < 1e-4:
            return best_U, best_V, best_W
        return None

    # Try to find rank 10 for 2x3x3
    U233, V233, W233 = optimize_233()
    
    if U233 is not None:
        # Construct rank 20 solution for 2x3x6
        rank_20 = 20
        U = np.zeros((nm, rank_20), dtype=np.float64)
        V = np.zeros((mp, rank_20), dtype=np.float64)
        W = np.zeros((pn, rank_20), dtype=np.float64)
        
        # Block 1: k in 0..2
        # U part is same for both blocks? 
        # In the decomposition sum_r U V W, if we use the same U for both, 
        # we are effectively doing sum (U V1 W1 + U V2 W2).
        # This is valid if the decomposition is linear.
        # We set U[:, 0:10] = U233
        U[:, 0:10] = U233
        
        # V part for block 1 (k=0,1,2)
        # V233 has indices j*3+k. We need V[j*6+k, r] for k=0,1,2.
        for j in range(3):
            for k in range(3):
                V[j*6 + k, 0:10] = V233[j*3 + k, :]
        
        # W part for block 1 (k=0,1,2)
        # W233 has indices k*2+i. We need W[k*2+i, r] for k=0,1,2.
        for k in range(3):
            for i in range(2):
                W[k*2 + i, 0:10] = W233[k*2 + i, :]
                
        # Block 2: k in 3..5
        # Use same U233? Or different?
        # If we use the same U233, we are reusing the A coefficients.
        # This is valid because A is the same.
        U[:, 10:20] = U233
        
        # V part for block 2 (k=3,4,5)
        # Map k' = k-3 in 0..2 to V233 index j*3+k'
        for j in range(3):
            for k_prime in range(3):
                k = k_prime + 3
                V[j*6 + k, 10:20] = V233[j*3 + k_prime, :]
        
        # W part for block 2 (k=3,4,5)
        for k_prime in range(3):
            k = k_prime + 3
            for i in range(2):
                W[k*2 + i, 10:20] = W233[k_prime*2 + i, :]
        
        # Verify
        if verify(U, V, W, T):
            # Return as list of lists
            U_list = [[float(U[i,j]) for j in range(rank_20)] for i in range(nm)]
            V_list = [[float(V[i,j]) for j in range(rank_20)] for i in range(mp)]
            W_list = [[float(W[i,j]) for j in range(rank_20)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
