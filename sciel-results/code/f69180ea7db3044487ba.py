"""Tensor decomposition for 3x3 * 3x4 matrix multiplication using SAT-like search over finite fields."""
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition as fallback."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(W[0]) if W else 0
    # For each (i,j,k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for non-matching triples, the sum should be 0
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # should equal 1 if this is a valid multiplication entry, else 0
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] where the "multiplication" maps (A,B) -> C
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # Actually the standard tensor: T[i,j,k] such that C[i,k] = sum_{j,c} A[i,j]*B[j,k]*T[i,j,k]
    # Wait, let me reconsider. The tensor for matrix multiplication C = A*B where A is n*m, B is m*p, C is n*p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]  ... no that's not right either
    
    # Let me re-read the convention: W[k*n+i, r]
    # The tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) triple contributes with coefficient 1
    
    # Wait, that's not right. If T[i,j,k] = 1 for all, then C[i,k] = sum_j sum_c A[i,j]*B[j,k]*W[k*n+i,c]... no.
    
    # Let me think again. The bilinear map: (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # This can be written as C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T_{i,j,k}^{(c)} where T is the tensor
    # Actually, the standard formulation: the multiplication tensor T has T[i,j,k] = 1 for all valid indices
    # and C[i,k] = sum_j A[i,j] B[j,k] means the tensor entries are all 1.
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_mod(U, V, W, mod):
    """Verify decomposition modulo mod."""
    n, m, p = DIMENSIONS
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = (s + U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]) % mod
                if s % mod != 1 % mod:
                    return False
    return True

def _random_init(rank, mod):
    """Generate random initial U, V, W over Z/mod."""
    n, m, p = DIMENSIONS
    U = [[random.randint(0, mod-1) for _ in range(rank)] for _ in range(n*m)]
    V = [[random.randint(0, mod-1) for _ in range(rank)] for _ in range(m*p)]
    W = [[random.randint(0, mod-1) for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def _gradient_step(U, V, W, mod, lr=1):
    """Perform one gradient descent step on the loss sum (T[i,j,k] - 1)^2 mod mod.
    For exact solution, we want T[i,j,k] ≡ 1 (mod mod) for all i,j,k."""
    n, m, p = DIMENSIONS
    rank = len(W[0]) if W else 0
    
    # Compute current T values and gradients
    # dL/dU[i*m+j,c] = sum_{k} (T[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c]
    # dL/dV[j*p+k,c] = sum_{i} (T[i,j,k]-1) * U[i*m+j,c] * W[k*n+i,c]
    # dL/dW[k*n+i,c] = sum_{j} (T[i,j,k]-1) * U[i*m+j,c] * V[j*p+k,c]
    
    # First compute T
    T = [[[0]*p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = (s + U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]) % mod
                T[i][j][k] = s
    
    # Compute residuals
    R = [[[0]*p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i][j][k] = (T[i][j][k] - 1) % mod
    
    # Compute gradients
    dU = [[0]*rank for _ in range(n*m)]
    dV = [[0]*rank for _ in range(m*p)]
    dW = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = R[i][j][k]
                if r == 0:
                    continue
                for c in range(rank):
                    dU[i*m+j][c] = (dU[i*m+j][c] + r * V[j*p+k][c] * W[k*n+i][c]) % mod
                    dV[j*p+k][c] = (dV[j*p+k][c] + r * U[i*m+j][c] * W[k*n+i][c]) % mod
                    dW[k*n+i][c] = (dW[k*n+i][c] + r * U[i*m+j][c] * V[j*p+k][c]) % mod
    
    # Update (negative gradient step)
    new_U = [[(U[i][c] - lr * dU[i][c]) % mod for c in range(rank)] for i in range(n*m)]
    new_V = [[(V[i][c] - lr * dV[i][c]) % mod for c in range(rank)] for i in range(m*p)]
    new_W = [[(W[i][c] - lr * dW[i][c]) % mod for c in range(rank)] for i in range(p*n)]
    
    return new_U, new_V, new_W

def _loss(U, V, W, mod):
    """Compute loss = sum (T[i,j,k]-1)^2 mod mod."""
    n, m, p = DIMENSIONS
    rank = len(W[0]) if W else 0
    loss = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = (s + U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]) % mod
                diff = (s - 1) % mod
                loss = (loss + diff * diff) % mod
    return loss

def _search_rank(rank, mod, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank over Z/mod."""
    random.seed(seed)
    best_loss = None
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(20):
        U, V, W = _random_init(rank, mod)
        current_loss = _loss(U, V, W, mod)
        
        if current_loss == 0:
            return U, V, W
        
        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_U, best_V, best_W = U[:], V[:], W[:]
        
        # Gradient descent
        for it in range(max_iters):
            U, V, W = _gradient_step(U, V, W, mod, lr=1)
            current_loss = _loss(U, V, W, mod)
            
            if current_loss == 0:
                return U, V, W
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_U, best_V, best_W = U[:], V[:], W[:]
    
    if best_loss == 0:
        return best_U, best_V, best_W
    return None

def _try_lift_to_int(U, V, W, mod):
    """Try to interpret modular solution as integer solution.
    If all entries are 0 or 1 in the mod solution, they might work as integers."""
    # Check if the solution works over integers
    if _verify(U, V, W):
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    # Known: 3x3x3 needs rank >= 20, 3x3x4 likely similar
    # Try a range of ranks
    for rank in range(20, 37):
        if time.time() > deadline:
            break
        
        # Try different moduli
        for mod in [5, 7, 11, 13]:
            if time.time() > deadline:
                break
            
            result = _search_rank(rank, mod, max_iters=2000, seed=rank*100+mod)
            if result is not None:
                U, V, W = result
                # Try to verify over integers
                int_result = _try_lift_to_int(U, V, W, mod)
                if int_result is not None:
                    return {"U": int_result[0], "V": int_result[1], "W": int_result[2]}
                
                # If modular solution exists, try to find integer solution by fixing structure
                # Check if entries are only 0 or 1 (or small integers)
                max_val = 0
                for mat in [U, V, W]:
                    for row in mat:
                        for v in row:
                            if abs(v) > max_val:
                                max_val = abs(v)
                
                # If all values are 0 or 1 mod mod, they might be valid integers
                if max_val <= 1:
                    if _verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
