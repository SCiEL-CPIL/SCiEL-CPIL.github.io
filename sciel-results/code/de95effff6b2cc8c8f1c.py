"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import itertools
import time
from collections import deque

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    result = np.zeros((n*m, m*p, p*n))
    for r_idx in range(r):
        result += np.einsum('a,b,c->abc', U[:,r_idx], V[:,r_idx], W[:,r_idx])
    return np.allclose(result, T, atol=tol)

def _als_polish(U, V, W, T, n, m, p, iterations=50, tol=1e-12):
    """Alternating Least Squares polishing step."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    for _ in range(iterations):
        # Update U: for each a, solve min_Ua ||sum_r Ua_r V[b,r] W[c,r] - T[a,:,:]||^2
        # This is a linear system: (V^T W elementwise)^T ... 
        # Actually, fixing V and W, the contribution to entry (a,b,c) is sum_r U[a,r]*V[b,r]*W[c,r]
        # Let M[b,c] = sum_r V[b,r]*W[c,r] (mp x pn matrix)
        # Then T[a,:,:] = U[a,:] @ M  (where U[a,:] is 1xr, M is rxmp*pn? No.)
        # U[a,r] * (V[b,r]*W[c,r]) summed over r.
        # So T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
        # Let's define B_r = V[:,r] * W[:,r] (elementwise product, size mp*pn? No, V is mp x r, W is pn x r)
        # V[b,r] * W[c,r] is not a matrix product directly.
        # Let's reshape.
        # For fixed V, W, the tensor T is approximated by U @ (V ⊙ W) where ⊙ is some operation.
        # Let's just use least squares on the flattened tensor.
        # Flatten T into a matrix T_mat of shape (nm, mp*pn)
        # Then T_mat[a, b*pn+c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let M be a matrix of shape (mp*pn, r) where M[b*pn+c, r] = V[b,r]*W[c,r]
        # Then T_mat = U @ M.T
        # So U = T_mat @ pinv(M.T)
        
        M = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                M[b*pn+c, :] = V[b, :] * W[c, :]
        
        T_mat = T.reshape(nm, mp*pn)
        U_new = T_mat @ np.linalg.pinv(M.T)
        
        # Update V
        # T_mat2 of shape (mp, nm*pn) where T_mat2[b, a*pn+c] = T[a,b,c]
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let N be matrix of shape (nm*pn, r) where N[a*pn+c, r] = U[a,r]*W[c,r]
        # Then T_mat2 = V @ N.T
        N = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                N[a*pn+c, :] = U[a, :] * W[c, :]
        T_mat2 = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm*pn)
        V_new = T_mat2 @ np.linalg.pinv(N.T)
        
        # Update W
        # T_mat3 of shape (pn, nm*mp) where T_mat3[c, a*mp+b] = T[a,b,c]
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let P be matrix of shape (nm*mp, r) where P[a*mp+b, r] = U[a,r]*V[b,r]
        # Then T_mat3 = W @ P.T
        P = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                P[a*mp+b, :] = U[a, :] * V[b, :]
        T_mat3 = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm*mp)
        W_new = T_mat3 @ np.linalg.pinv(P.T)
        
        # Check convergence
        U_old, V_old, W_old = U.copy(), V.copy(), W.copy()
        U, V, W = U_new, V_new, W_new
        
        if np.linalg.norm(U - U_old) + np.linalg.norm(V - V_old) + np.linalg.norm(W - W_old) < tol:
            break
            
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification checking only active and a sample of inactive entries."""
    r = U.shape[1]
    # Check all active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of inactive entries
    checked = 0
    for _ in range(200):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for cc in range(r):
            val += U[a, cc] * V[b, cc] * W[c, cc]
        if abs(val) > tol:
            return False
        checked += 1
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_rank = n * m * p  # schoolbook = 60
    best_result = None
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    np.random.seed(42)
    
    # Lower bound is max(nm, mp, pn) = 30.
    # Try ranks from 30 up to 59.
    # Use a combination of random initialization, gradient descent, and ALS polishing.
    
    for rank in range(30, best_rank):
        if time.time() > deadline - 15:
            break
            
        found = False
        max_restarts = 3 if rank < 45 else 2
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random values
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # Gradient Descent Phase
            lr = 0.02
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            best_loss = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Run a short GD phase
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                    
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - recon
                loss = np.sum(R**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                # Gradients
                grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
                
                # Momentum update
                vU = momentum * vU + (1 - momentum) * grad_U
                vV = momentum * vV + (1 - momentum) * grad_V
                vW = momentum * vW + (1 - momentum) * grad_W
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
                # Adaptive lr
                if loss > best_loss * 1.5:
                    lr *= 0.7
                    vU = 0
                    vV = 0
                    vW = 0

            # If GD found a reasonably good solution, polish with ALS
            if best_loss < 1e-1:
                U, V, W = _als_polish(best_U, best_V, best_W, T, n, m, p, iterations=50)
                
                # Final verification
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    found = True
                    best_rank = rank
                    best_result = (U, V, W)
                    break
        
        if found:
            break

    # Convert to lists
    U_final, V_final, W_final = best_result
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
