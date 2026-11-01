"""Rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    lb = max(nm, mp, pn)  # 18

    # Precompute indices for tensor T
    # T is a tensor of size (n, m, p) with all entries 1.
    # We want U (n*m, r), V (m*p, r), W (p*n, r)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U, V, W, r):
        """Check exact decomposition using NumPy."""
        # Reshape U, V, W to tensor forms
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Compute sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a batched dot product or einsum
        # Result shape: (n, m, p)
        # np.einsum('ijc,jkc,kic->ijp', U_t, V_t, W_t) 
        # Wait, W is indexed by k, i. So W_t is (p, n, r).
        # We need to align indices.
        # U[i, j, c]
        # V[j, k, c]
        # W[k, i, c] -> W_t[k, i, c]
        
        result = np.einsum('ijc,jkc,kic->ijp', U_t, V_t, W_t)
        
        # Check if all entries are close to 1.0
        if np.allclose(result, 1.0, atol=1e-7):
            return True
        return False

    def optimize(U, V, W, r, steps=200, lr=0.01):
        """Run gradient descent to minimize error."""
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Precompute partial sums for gradient
        # grad_U[i, j, c] = 2 * sum_k (sum_c' U[i,j,c']V[j,k,c']W[k,i,c'] - 1) * V[j,k,c] * W[k,i,c]
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Compute current tensor product
            prod = np.einsum('ijc,jkc,kic->ijp', U_t, V_t, W_t)
            error = prod - 1.0
            
            # Compute gradients
            # grad_U[i, j, c] = 2 * sum_k error[i, j, k] * V[j, k, c] * W[k, i, c]
            # This can be computed as:
            # temp1[i, j, k, c] = error[i, j, k] * V[j, k, c]
            # grad_U[i, j, c] = 2 * sum_k temp1[i, j, k, c] * W[k, i, c]
            
            # Using einsum for efficiency
            # error: (n, m, p)
            # V_t: (m, p, r)
            # W_t: (p, n, r)
            
            # grad_U: (n, m, r)
            # grad_U[i, j, c] = 2 * sum_k error[i, j, k] * V_t[j, k, c] * W_t[k, i, c]
            # Let's use einsum: 'ijk,jkc,kic->ijc'
            grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', error, V_t, W_t)
            
            # grad_V[j, k, c] = 2 * sum_i error[i, j, k] * U_t[i, j, c] * W_t[k, i, c]
            # 'ijk,ijc,kic->jkc'
            grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', error, U_t, W_t)
            
            # grad_W[k, i, c] = 2 * sum_j error[i, j, k] * U_t[i, j, c] * V_t[j, k, c]
            # 'ijk,ijc,jkc->kic'
            grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', error, U_t, V_t)
            
            # Update
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
            
            # Check convergence
            if np.max(np.abs(error)) < 1e-8:
                break

        U = U_t.reshape(nm, r)
        V = V_t.reshape(mp, r)
        W = W_t.reshape(pn, r)
        return U, V, W

    def clean(mat):
        """Round to nearest integer or half-integer."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Try integer
                iv = round(val)
                if abs(val - iv) < 1e-6:
                    new_row.append(int(iv))
                else:
                    # Try half-integer
                    hv = round(val * 2) / 2.0
                    if abs(val - hv) < 1e-6:
                        new_row.append(hv)
                    else:
                        new_row.append(val)
            result.append(new_row)
        return result

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None

    # Schoolbook fallback
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

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try ranks from lower bound upwards
    for rank in range(lb, n * m * p + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        for seed in range(100):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(seed)
            # Initialize with small random values
            U = rng.randn(nm, rank) * 0.1
            V = rng.randn(mp, rank) * 0.1
            W = rng.randn(pn, rank) * 0.1
            
            # Optimize
            U, V, W = optimize(U, V, W, rank, steps=500, lr=0.05)
            
            # Verify
            if verify(U, V, W, rank):
                # Clean
                U_clean = clean(U.tolist())
                V_clean = clean(V.tolist())
                W_clean = clean(W.tolist())
                
                # Final verification of cleaned version
                if verify(np.array(U_clean), np.array(V_clean), np.array(W_clean), rank):
                    best_rank = rank
                    best_U, best_V, best_W = U_clean, V_clean, W_clean
                    found = True
                    break
        
        if found:
            break

    # If best_U is still None (should not happen), use schoolbook
    if best_U is None:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    return {"U": best_U, "V": best_V, "W": best_W}
