import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Precompute the non-zero indices of the tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k*n+i)
    # Note: The task says W[k*n+i, r]. So the output index is c = k*n + i.
    # A index a = i*m + j
    # B index b = j*p + k
    # C index c = k*n + i
    
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_indices.append((a, b, c))
                
    def check_decomposition(U, V, W, tol=1e-9):
        """
        Check if U, V, W exactly decompose the tensor.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        R = U.shape[1]
        # We need sum_r U[a,r] V[b,r] W[c,r] == 1 for (a,b,c) in tensor_indices
        # and == 0 for all other (a,b,c).
        # Checking all (a,b,c) is too slow (5184 * R).
        # But we can check the tensor product explicitly?
        # T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a tensor of size 12x24x18 = 5184 elements.
        # Then compare with the target tensor.
        
        # Construct target tensor T_target
        T_target = np.zeros((nm, mp, pn), dtype=float)
        for a, b, c in tensor_indices:
            T_target[a, b, c] = 1.0
            
        # Compute reconstructed tensor
        # Using einsum is efficient enough for small dims
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check exactness (integer/half-integer inputs should lead to exact float results if small)
        if not np.allclose(T_rec, T_target, atol=tol):
            return False
        return True

    def als_search(rank, max_iters=50, time_limit=5.0):
        """
        Attempt to find a decomposition of given rank using ALS.
        """
        start_time = time.time()
        R = rank
        # Initialize randomly
        # We want integer/half-integer solutions. 
        # Let's initialize with random integers in {-1, 0, 1}
        U = np.random.randint(-1, 2, (nm, R))
        V = np.random.randint(-1, 2, (mp, R))
        W = np.random.randint(-1, 2, (pn, R))
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Fix V, W, solve for U
            # We want U such that sum_r U[a,r] (V[b,r] W[c,r]) approx T[a,b,c]
            # This is not a simple linear LS because the target depends on a,b,c.
            # Let's flatten the tensor equation.
            # For each (b,c), we have a vector V_b_c of length R: V_b_c[r] = V[b,r] W[c,r]
            # Then we want U[a, :] dot V_b_c ~= T[a,b,c]
            # This is a linear system for each (b,c) pair? No, U is shared.
            # We can form a large linear system.
            # Variables: U_flat (nm * R)
            # Equations: for each (a,b,c), sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # This is linear in U if V,W are fixed.
            
            # Build the matrix A_U for the linear system A_U * U_flat = T_flat
            # A_U has size (nm*mp*pn, nm*R). This is huge (5184, 12*R).
            # But it's sparse.
            # Row index idx = a * (mp*pn) + b * pn + c
            # Col index for U[a,r] is a * R + r.
            # Value is V[b,r] * W[c,r].
            
            # Instead of building the full matrix, we can use normal equations or iterative methods.
            # Given the small size, let's just use numpy's lstsq on a sparse-like structure or dense if R is small.
            # Actually, we can optimize U by minimizing || T - U V W^T ||_F^2.
            # Gradient descent might be simpler and faster for small R.
            
            # Let's use a simple gradient step on the flattened U, V, W.
            # Error E = || T - sum u_r v_r w_r ||^2
            # dE/du_r = -2 * (T - Rec) * v_r * w_r (contracted over b,c)
            
            # Compute Rec
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            Error = T_target - T_rec
            
            # Compute gradients
            # dU[a,r] = -2 * sum_{b,c} Error[a,b,c] V[b,r] W[c,r]
            # dV[b,r] = -2 * sum_{a,c} Error[a,b,c] U[a,r] W[c,r]
            # dW[c,r] = -2 * sum_{a,b} Error[a,b,c] U[a,r] V[b,r]
            
            # We can compute these using einsum
            dU = -2 * np.einsum('abc,br,cr->ar', Error, V, W)
            dV = -2 * np.einsum('abc,ar,cr->br', Error, U, W)
            dW = -2 * np.einsum('abc,ar,br->cr', Error, U, V)
            
            # Step size
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Project to integers (or half-integers)
            # For now, project to integers
            U = np.round(U)
            V = np.round(V)
            W = np.round(W)
            
            # Check if we found a solution
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_rec, T_target, atol=1e-9):
                return U, V, W, True
                
            # Track best
            err = np.linalg.norm(Error)
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        return None, None, None, False

    # Try to find a lower rank
    # Start from 71 down to 60
    # If found, return it
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 12, 0), -1):
        U, V, W, success = als_search(rank)
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
