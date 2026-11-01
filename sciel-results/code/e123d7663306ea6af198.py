"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                target_indices.append((a, b, c))
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)
    
    def compute_gradients(U, V, W):
        """Compute gradients of tensor_error w.r.t. U, V, W efficiently."""
        r = U.shape[1]
        R = compute_R(U, V, W)
        E = R - target  # error tensor
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            # grad_U[:, c] = 2 * einsum('abc,bc2->ac', E, V[:, c] * W[:, c]) 
            # But need to be careful: for fixed c, we differentiate w.r.t. U[:, c]
            # d/dU[a,c'] E[a,b,c2]^2 = 2*E[a,b,c2]*dR[a,b,c2]/dU[a',c']
            # = 2*E[a,b,c2]*V[b,c']*W[c2,c']*delta_{a,a'}
            # So grad_U[a',c'] = 2*sum_{b,c2} E[a',b,c2]*V[b,c']*W[c2,c']
            
            # Efficient computation:
            # temp1[b, c2] = E[:, b, c2] weighted... let's do it step by step
            # For each c', we need: sum_{b,c2} E[a,b,c2] * V[b,c'] * W[c2,c']
            
            # Let's compute: for fixed c', 
            # grad_U[:, c'] = 2 * (E * W[:, :, c'] expanded) contracted with V[:, c']
            
            # E has shape (nm, mp, pn)
            # We want: for each a, sum_{b,c2} E[a,b,c2] * V[b,c'] * W[c2,c']
            
            # = sum_{b} V[b,c'] * (sum_{c2} E[a,b,c2] * W[c2,c'])
            # = sum_{b} V[b,c'] * (E[:, b, :] dot W[:, c'])
            
            # Let's compute M[a, b] = sum_{c2} E[a,b,c2] * W[c2,c']
            # Then grad_U[a, c'] = 2 * sum_b M[a,b] * V[b,c']
            
            # M = np.einsum('abc,c->ab', E, W[:, c'])
            # grad_U[:, c'] = 2 * M @ V[:, c']
            
            M_U = np.einsum('abc,c->ab', E, W[:, c])
            grad_U[:, c] = 2.0 * (M_U @ V[:, c])
            
            # Similarly for V:
            # grad_V[b, c'] = 2 * sum_{a,c2} E[a,b,c2] * U[a,c'] * W[c2,c']
            # Let N[b, c2] = sum_a E[a,b,c2] * U[a,c']
            # grad_V[b, c'] = 2 * sum_{c2} N[b,c2] * W[c2,c']
            
            N_V = np.einsum('abc,a->bc', E, U[:, c])
            grad_V[:, c] = 2.0 * (N_V @ W[:, c])
            
            # For W:
            # grad_W[c2, c'] = 2 * sum_{a,b} E[a,b,c2] * U[a,c'] * V[b,c']
            # Let P[a,b] = E[a,b,:] * ... 
            # grad_W[:, c'] = 2 * einsum('abc,a,b->c', E, U[:, c'], V[:, c'])
            
            # = 2 * sum_{a,b} E[a,b,c2] * U[a,c'] * V[b,c']
            # Let Q[c2] = sum_{a,b} E[a,b,c2] * U[a,c'] * V[b,c']
            # = sum_{a,b} E[a,b,c2] * (U[:, c'] * V[:, c'])_{a,b}... no
            # = einsum('abc,ab->c', E, U[:, c'][:, None] * V[:, c'][None, :])
            
            UV_outer = np.outer(U[:, c], V[:, c])  # shape (nm, mp)
            grad_W[:, c] = 2.0 * np.einsum('abc,ab->c', E, UV_outer)
        
        return grad_U, grad_V, grad_W
    
    def optimize(U, V, W, max_iters=500, lr_init=0.1):
        """Optimize U, V, W to minimize tensor error using gradient descent with momentum."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        # Momentum
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.9
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            # Update velocities
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            # Update
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                # Decay learning rate
                lr *= 0.995
            else:
                # Roll back and reduce learning rate
                lr *= 0.5
                if lr < 1e-10:
                    break
        
        return U, V, W, prev_error
    
    best_result = None
    best_rank = 42
    best_error = float('inf')
    
    # Try ranks from low to high
    for rank in range(25, 42):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(hash((rank, attempt)) % 2**32)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            # Optimize
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=300, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                best_result = (U_rounded, V_rounded, W_rounded)
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    # Convert to lists as required
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
