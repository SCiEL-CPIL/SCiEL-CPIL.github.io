import numpy as np
import time
import itertools
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # Leave 15s safety margin

    def _verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition: sum_r U[:,r] outer V[:,r] outer W[:,r] matches matmul tensor."""
        # Build target tensor T of shape (n, m, p)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Actually: the multiplication tensor has T[i,j,k]=1 iff the term A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid i,j,k (since every such triple is a valid term)
        target = np.ones((n, m, p))
        
        # Reconstruct from decomposition
        # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        
        # Reshape and compute
        U_r = U.T  # (r, n*m)
        V_r = V.T  # (r, m*p)
        W_r = W.T  # (r, p*n)
        
        # The decomposition gives: for each r, U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The tensor reconstructed is: T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_recon = np.zeros((n, m, p))
        for r_idx in range(r):
            U_vec = U_r[r_idx].reshape(n, m)
            V_vec = V_r[r_idx].reshape(m, p)
            W_vec = W_r[r_idx].reshape(p, n)
            # For each (i,j,k): contribution = U[i,j] * V[j,k] * W[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_vec[i, j] * V_vec[j, k] * W_vec[k, i]
        
        return np.allclose(T_recon, target, atol=tol)

    def _verify_fast(U, V, W, tol=1e-9):
        """Faster verification using einsum."""
        r = U.shape[1]
        # Target tensor
        target = np.ones((n, m, p))
        
        # Reshape
        U_3d = U.T.reshape(r, n, m)  # (r, i, j)
        V_3d = V.T.reshape(r, m, p)  # (r, j, k)
        W_3d = W.T.reshape(r, p, n)  # (r, k, i)
        
        # T_recon[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
        # einsum: 'rij,rjk,rki->ijk'
        T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T_recon, target, atol=tol)

    def _schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63

    def _strassen_3x3x3():
        """Known rank-7 Strassen decomposition for 3x3x3 matrix multiplication."""
        # Strassen's 7-multiplication algorithm
        # Split matrices into 2x2 blocks
        # For 3x3, we need to embed into 4x4 or use specific constructions
        # 
        # Actually, let's use a known explicit rank-7 decomposition.
        # The standard Strassen algorithm has 7 multiplications.
        # For 3x3, one approach: use the 7 multiplications with appropriate projections.
        #
        # Let's use a concrete known rank-7 decomposition.
        # From literature: the multiplication tensor of (3,3,3) has rank 7.
        # Here's one explicit decomposition:
        
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # Strassen's 7 products for splitting into blocks.
        # For 3x3, we can think of it as:
        # Let A be 3x3, B be 3x3, C = A*B be 3x3.
        # 
        # One known rank-7 decomposition (adapted from Strassen):
        # The 7 tensors correspond to the 7 multiplications.
        
        # Actually, let me use a verified explicit form.
        # Indices: i,j in {0,1,2} for 3x3
        # 
        # Let me just hardcode a known valid decomposition if I can verify it quickly,
        # otherwise fall back to schoolbook.
        
        # For safety, let's construct a decomposition that we know works.
        # The simplest verified one: use the 9-term schoolbook for 3x3x3.
        # But we want rank 7. Let me try to find one.
        
        # Known: Strassen decomposition can be written as follows for (4,4,4).
        # For (3,3,3), there's a specific rank-7 form.
        
        # Let me use this explicit rank-7 decomposition from the literature:
        # (This is a standard one)
        
        # I'll define it using the structure.
        # The 7 basis elements:
        # P1 = M11*M11
        # P2 = (M21+M11)*M12  -> for 3x3 this gets complex
        # 
        # Actually, for 3x3 specifically, let me just try a few known patterns
        # and verify. If none work quickly, fall back.
        
        # For now, return None to indicate we don't have a verified rank-7.
        return None

    # Try structured split: 3x3x7 = 3x3x3 + 3x3x4
    def _structured_split():
        """Split 3x3x7 into 3x3x3 and 3x3x4 parts."""
        n, m, p = DIMENSIONS
        # First 3 columns: 3x3x3 multiplication
        # Last 4 columns: 3x3x4 multiplication
        
        U1, V1, W1 = _schoolbook(n, m, 3)
        U2, V2, W2 = _schoolbook(n, m, 4)
        
        r1 = U1.shape[1]  # 27
        r2 = U2.shape[1]  # 36
        r_total = r1 + r2  # 63
        
        # This is same as schoolbook, no improvement. Skip if not helpful.
        return None

    # Try random search with gradient descent on the tensor matching loss
    def _random_gradient_search(r_target, max_iters=500, lr=0.01, n_restarts=5):
        """Try to find a rank-r decomposition via random init + gradient descent."""
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r_target) * 0.5
            V = np.random.randn(m * p, r_target) * 0.5
            W = np.random.randn(p * n, r_target) * 0.5
            
            U_3d = U.T.reshape(r_target, n, m)
            V_3d = V.T.reshape(r_target, m, p)
            W_3d = W.T.reshape(r_target, p, n)
            target = np.ones((n, m, p))
            
            def loss_and_grad(params):
                U_flat = params[0: r_target * n * m].reshape(r_target, n, m)
                V_flat = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                W_flat = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                
                T_recon = np.einsum('rij,rjk,rki->ijk', U_flat, V_flat, W_flat)
                diff = T_recon - target
                loss = np.sum(diff ** 2)
                
                # Gradients
                # dL/dU_3d = diff * V_3d * W_3d (with appropriate index swapping)
                # dT_recon/dU_3d[i,j,r] = V_3d[r,j,k] * W_3d[r,k,i] (summed over k)
                # dT_recon/dV_3d[j,k,r] = U_3d[r,i,j] * W_3d[r,k,i] (summed over i)
                # dT_recon/dW_3d[k,i,r] = U_3d[r,i,j] * V_3d[r,j,k] (summed over j)
                
                dU = np.einsum('ijk,rjk,rki->rij', diff, V_flat, W_flat)
                dV = np.einsum('ijk,rij,rki->rjk', diff, U_flat, W_flat)
                dW = np.einsum('ijk,rij,rjk->rki', diff, U_flat, V_flat)
                
                grad = np.concatenate([
                    dU.reshape(-1),
                    dV.reshape(-1),
                    dW.reshape(-1)
                ])
                return loss, grad
            
            # Use gradient descent manually (avoid scipy overhead)
            params = np.concatenate([
                U_3d.reshape(-1),
                V_3d.reshape(-1),
                W_3d.reshape(-1)
            ])
            
            loss = np.inf
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                l, g = loss_and_grad(params)
                loss = l
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                if loss < 1e-6:
                    break
                params -= lr * g
                # Optional: adaptive lr
                lr = max(0.001, lr * 0.999)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
        
        if best_params is not None and best_loss < 1e-5:
            U_3d = best_params[0: r_target * n * m].reshape(r_target, n, m)
            V_3d = best_params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
            W_3d = best_params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
            
            U_out = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
            V_out = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
            W_out = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
            
            return U_out, V_out, W_out, best_loss
        
        return None

    # Try decreasing ranks from below schoolbook
    # Use a smarter approach: try ranks that are known to be achievable
    # For 3x3x7, the schoolbook is 63. We can try to find lower ranks.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also rank >= ceil(n*m*p / max(n,m,p)) = ceil(63/7) = 9 (weak)
    # Try ranks from 63 down to 25 in steps of 1 for better chances
    
    # First, try to find any decomposition with rank < 63 using gradient descent
    # Focus on ranks that are more likely to be achievable
    candidate_ranks = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    
    for r_target in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        # Use multiple restarts with different strategies
        found = False
        for strategy in range(3):
            if time.time() > deadline - 30:
                break
            
            # Different initialization strategies
            if strategy == 0:
                # Random init
                result = _random_gradient_search(r_target, max_iters=200, lr=0.05, n_restarts=5)
            elif strategy == 1:
                # Sparse init (many zeros)
                U_init = np.zeros((n * m, r_target))
                V_init = np.zeros((m * p, r_target))
                W_init = np.zeros((p * n, r_target))
                # Add some random non-zero entries
                for _ in range(r_target * 3):
                    ri = np.random.randint(r_target)
                    U_init[np.random.randint(n * m), ri] = np.random.randn()
                    V_init[np.random.randint(m * p), ri] = np.random.randn()
                    W_init[np.random.randint(p * n), ri] = np.random.randn()
                
                # Try gradient descent from this init
                U_3d = U_init.T.reshape(r_target, n, m)
                V_3d = V_init.T.reshape(r_target, m, p)
                W_3d = W_init.T.reshape(r_target, p, n)
                target = np.ones((n, m, p))
                
                def loss_and_grad(params):
                    U_flat = params[0: r_target * n * m].reshape(r_target, n, m)
                    V_flat = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                    W_flat = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                    
                    T_recon = np.einsum('rij,rjk,rki->ijk', U_flat, V_flat, W_flat)
                    diff = T_recon - target
                    loss = np.sum(diff ** 2)
                    
                    dU = np.einsum('ijk,rjk,rki->rij', diff, V_flat, W_flat)
                    dV = np.einsum('ijk,rij,rki->rjk', diff, U_flat, W_flat)
                    dW = np.einsum('ijk,rij,rjk->rki', diff, U_flat, V_flat)
                    
                    grad = np.concatenate([
                        dU.reshape(-1),
                        dV.reshape(-1),
                        dW.reshape(-1)
                    ])
                    return loss, grad
                
                params = np.concatenate([
                    U_3d.reshape(-1),
                    V_3d.reshape(-1),
                    W_3d.reshape(-1)
                ])
                
                loss = np.inf
                lr = 0.05
                for it in range(200):
                    if time.time() > deadline - 30:
                        break
                    l, g = loss_and_grad(params)
                    loss = l
                    if loss < 1e-6:
                        break
                    params -= lr * g
                    lr = max(0.001, lr * 0.999)
                
                if loss < 1e-5:
                    U_3d = params[0: r_target * n * m].reshape(r_target, n, m)
                    V_3d = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                    W_3d = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                    U_c = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
                    V_c = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
                    W_c = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
                    
                    if _verify_fast(U_c, V_c, W_c, tol=1e-5):
                        U_best, V_best, W_best = U_c, V_c, W_c
                        best_rank = r_target
                        found = True
                        break
            else:
                # Use schoolbook as starting point and try to reduce rank
                # This is complex, so skip for now
                pass
        
        if found:
            break
        
        # If we found a valid decomposition at this rank, break
        if best_rank == r_target:
            break

    # Final check and return
    if time.time() > deadline - 5:
        # Return best found
        pass

    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
