"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _try_decompose_233(rank):
    """
    Try to find a rank-`rank` exact tensor decomposition for 2x3 by 3x3.
    Returns (U, V, W) as lists or None.
    """
    n, m, p = 2, 3, 3
    # Variables:
    # U: n*m x rank = 6 x rank
    # V: m*p x rank = 9 x rank
    # W: p*n x rank = 6 x rank
    
    # Total parameters: (6+9+6)*rank = 21*rank
    # For small ranks, we can try random initialization + local search.
    
    best_rank = None
    start_time = time.time()
    timeout = 30  # seconds per rank attempt
    
    # Try random restarts
    for restart in range(50):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with small random values
        np.random.seed(restart)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Gradient-based optimization using numpy
        # Objective: minimize sum of (tensor_component - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        # for all i,j,k in the support, and sum of (sum_c ...)^2 for non-support entries
        
        # For 2x3x3: support is all (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1,2}
        # That's all 18 entries. So we want all 18 entries to be 1.
        # Wait, the tensor for matrix multiplication T_{(i,j),(j,k),(k,i)} = delta_{ii'}delta_{jj'}delta_{kk'}
        # Actually the multiplication tensor has entries:
        # T[(i,j), (j',k'), (k,i')] = 1 if i=i' and j=j' and k=k', else 0
        # For 2x3*3x3: 
        #   First index: (i,j) with i in 0..1, j in 0..2 -> 6 entries
        #   Second index: (j',k') with j' in 0..2, k' in 0..2 -> 9 entries
        #   Third index: (k,i') with k in 0..2, i' in 0..1 -> 6 entries
        #   T[(i,j),(j',k'),(k,i')] = 1 if i=i', j=j', k=k' else 0
        
        # So there are 2*3*3 = 18 ones and the rest zeros.
        # Total tensor size: 6*9*6 = 324 entries
        
        lr = 0.01
        for it in range(200):
            # Compute current tensor estimate
            # T_est[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # We only need to check the 18 support entries (should be 1) 
            # and optionally the zeros (should be 0)
            
            # For speed, only compute support entries
            errors = []
            grads_U = np.zeros_like(U)
            grads_V = np.zeros_like(V)
            grads_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        # T[a,b,c] should be 1
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err = 1.0 - val
                        errors.append(err)
                        # d/dU[a,r] = -err * V[b,r] * W[c,r]
                        grads_U[a, :] -= err * V[b, :] * W[c, :]
                        grads_V[b, :] -= err * U[a, :] * W[c, :]
                        grads_W[c, :] -= err * U[a, :] * V[b, :]
            
            # Also add penalty for zero entries (sample a few)
            # For efficiency, check a subset of zero entries
            for a in range(n*m):
                for b in range(m*p):
                    i = a // m
                    j = a % m
                    jj = b // p
                    kk = b % p
                    if jj != j:
                        continue
                    for c in range(p*n):
                        ii = c % n
                        kk2 = c // n
                        if ii != i or kk2 != kk:
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            # Penalty: val should be 0
                            errors.append(val)
                            grads_U[a, :] += val * V[b, :] * W[c, :]
                            grads_V[b, :] += val * U[a, :] * W[c, :]
                            grads_W[c, :] += val * U[a, :] * V[b, :]
            
            loss = np.sum(np.array(errors) ** 2)
            if loss < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
            
            U -= lr * grads_U
            V -= lr * grads_V
            W -= lr * grads_W
            
            if it % 20 == 0 and loss < 1e-6:
                # Try to polish
                lr = 0.001
                for it2 in range(50):
                    errors2 = []
                    grads_U2 = np.zeros_like(U)
                    grads_V2 = np.zeros_like(V)
                    grads_W2 = np.zeros_like(W)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                err = 1.0 - val
                                errors2.append(err)
                                grads_U2[a, :] -= err * V[b, :] * W[c, :]
                                grads_V2[b, :] -= err * U[a, :] * W[c, :]
                                grads_W2[c, :] -= err * U[a, :] * V[b, :]
                    loss2 = np.sum(np.array(errors2) ** 2)
                    if loss2 < 1e-14:
                        return U.tolist(), V.tolist(), W.tolist()
                    U -= lr * grads_U2
                    V -= lr * grads_V2
                    W -= lr * grads_W2
                lr = 0.01
        if time.time() - start_time > timeout:
            break
    
    return None

def _verify_233(U, V, W):
    """Verify that U,V,W give an exact decomposition for 2x3*3x3."""
    n, m, p = 2, 3, 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check all support entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check zero entries
    for a in range(n*m):
        for b in range(m*p):
            i = a // m
            j = a % m
            jj = b // p
            kk = b % p
            for c in range(p*n):
                ii = c % n
                kk2 = c // n
                if not (ii == i and jj == j and kk2 == kk):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Try ranks from lower bound up
    # Lower bound for 2x3*3x3: at least max(n*m, m*p, p*n) = max(6,9,6) = 9
    # But known results suggest rank 18 might be minimal, or maybe lower
    # Let's try 9, 10, 11, ... up to 17
    
    for rank in range(9, 18):
        if time.time() - start > 100:
            break
        result = _try_decompose_233(rank)
        if result is not None:
            U, V, W = result
            if _verify_233(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
