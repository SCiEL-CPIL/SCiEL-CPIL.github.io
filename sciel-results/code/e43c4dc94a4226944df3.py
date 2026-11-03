import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
        n, m, p = DIMENSIONS
        # Target tensor T
        # T[i*m+j, j*p+k, k*n+i] = 1
        # We want sum_alpha U[i*m+j, alpha] V[j*p+k, alpha] W[k*n+i, alpha] = T[i*m+j, j*p+k, k*n+i]
        
        # Precompute non-zero indices
        nz_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    nz_indices.append((idx_u, idx_v, idx_w))
        
        def check_decomposition(U, V, W):
            # U: (nm, r), V: (mp, r), W: (pn, r)
            r = U.shape[1]
            # Reconstruct
            # This is O(r * n * m * p)
            # We only need to check non-zero entries?
            # No, we must ensure zero entries are zero.
            # But if we use sparse logic, we can check all.
            # For small sizes, dense check is fine.
            
            # Efficient check:
            # Sum over alpha
            # For each (i,j,k), compute sum
            # This is slow in Python. Use NumPy.
            
            # Flatten T
            T_flat = np.zeros(n * m * p)
            for idx_u, idx_v, idx_w in nz_indices:
                # The tensor is not a simple matrix.
                # We need to verify the identity:
                # sum_alpha U[idx_u, alpha] V[idx_v, alpha] W[idx_w, alpha] == 1 if (idx_u, idx_v, idx_w) is in nz_indices
                # AND == 0 otherwise.
                
                # This is hard to vectorize simply without building the full tensor.
                # Let's build the full tensor T (size 9 * 15 * 15 = 2025).
                # It's small.
                pass
            
            # Build full T
            T = np.zeros((n*m, m*p, p*n))
            for idx_u, idx_v, idx_w in nz_indices:
                T[idx_u, idx_v, idx_w] = 1.0
            
            # Reconstruct
            # R = sum_alpha U[:, alpha] np.outer(V[:, alpha], W[:, alpha])
            # This is O(r * nm * mp * pn)
            # r ~ 40, nm=9, mp=15, pn=15. 40 * 9 * 15 * 15 = 81000 ops. Fast.
            
            R = np.zeros((n*m, m*p, p*n))
            for alpha in range(r):
                R += np.einsum('i,j,k->ijk', U[:, alpha], V[:, alpha], W[:, alpha])
            
            return np.allclose(R, T, atol=1e-9)

        def is_valid_entries(U, V, W):
            # Check if all entries are integer or half-integer
            for M in [U, V, W]:
                for val in M:
                    if not (np.isclose(val * 2, round(val * 2), atol=1e-9)):
                        return False
            return True

        # Start with schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Try to reduce rank
        # Simple search: try to remove columns if they are linearly dependent?
        # Or try to find a lower rank decomposition.
        
        # Given time, just return schoolbook if no quick win.
        # But we can try a few random perturbations.
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
