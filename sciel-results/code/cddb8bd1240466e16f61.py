import numpy as np

def construct():
    n, m, p = 2, 4, 6
    # Target tensor T[i, j, k] for A[i, j] * B[j, k] = C[i, k]
    # A is 2x4, B is 4x6, C is 2x6
    # T[i, j, k] = 1 if A[i,j] and B[j,k] contribute to C[i,k]
    # Actually, the tensor is defined by T[i, j, k] = 1 for all i,j,k?
    # No, the multiplication tensor has T[i, j, k] = 1 if the term A[i,j]B[j,k] appears in C[i,k].
    # In standard matrix multiplication, C[i,k] = sum_j A[i,j]B[j,k].
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, the shape is (n, m, p) = (2, 4, 6).
    # T[i, j, k] = 1.
    
    T = np.ones((n, m, p))
    
    # Reshape T to a matrix for linear solving
    # Equation: sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # This is not linear in U, V, W simultaneously.
    # But if we fix U and V, it is linear in W.
    # Let M be a matrix where rows correspond to (i, j) and cols to (k, r)?
    # T[i, j, k] = sum_r (U[i, r] V[j, r]) W[k, r]
    # Let X[i, j, r] = U[i, r] V[j, r].
    # Then T[i, j, k] = sum_r X[i, j, r] W[k, r].
    # We can vectorize this.
    # For each r, we have a term U[:, r] (outer) V[:, r] (outer) ... no.
    # Term r is U[:, r] (n x 1) * V[:, r] (m x 1) * W[:, r] (p x 1).
    # The contribution to T[i, j, k] is U[i, r] * V[j, r] * W[k, r].
    
    # If U and V are fixed, we can form a matrix A_sys of size (n*m*p) x (p*R)
    # Actually, for each (i, j), we have a vector of length p.
    # T[i, j, :] = sum_r (U[i, r] * V[j, r]) * W[:, r]
    # Let coeff[i, j, r] = U[i, r] * V[j, r].
    # Then T[i, j, k] = sum_r coeff[i, j, r] * W[k, r].
    # This is a linear system for W.
    # We can stack all (i, j) equations.
    # Total equations: n * m = 8.
    # Each equation is a vector of length p=6.
    # So we have 8 vectors of length 6.
    # Unknowns: W is (p x R) = 6 x R.
    # Total unknowns: 6 * R.
    # Total equations: 8 * 6 = 48.
    
    # Matrix form:
    # Let W_flat be length 6*R.
    # Let T_flat be length 48.
    # We need a matrix M_sys of size 48 x (6*R).
    # M_sys[ (i*m+j)*p + k, k*R + r ] = U[i, r] * V[j, r]
    
    # Try ranks 40, 41, 42
    for R in [40, 41, 42]:
        # Try a few random integer initializations
        for _ in range(100):
            # Initialize U and V with small integers
            U = np.random.randint(-1, 2, size=(n, R))
            V = np.random.randint(-1, 2, size=(m, R))
            
            # Build system matrix
            M_sys = np.zeros((n * m * p, p * R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = (i * m + j) * p + k
                        for r in range(R):
                            col_idx = k * R + r
                            M_sys[row_idx, col_idx] = U[i, r] * V[j, r]
            
            T_flat = T.flatten()
            
            # Solve for W_flat
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(M_sys, T_flat, rcond=None)
                W = W_flat.reshape(p, R)
                
                # Check if W is close to half-integers
                W_half = np.round(W * 2) / 2.0
                if np.allclose(W, W_half, atol=1e-6):
                    # Verify the decomposition exactly
                    # Reconstruct T
                    T_check = np.zeros((n, m, p))
                    for r in range(R):
                        T_check += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W_half[:, r])
                    if np.allclose(T_check, T, atol=1e-9):
                        # Found a valid decomposition
                        # Convert to lists
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W_half.tolist()
                        }
            except Exception:
                pass
        
        # If not found, fall back to next rank
    # Fallback to known rank 42 decomposition
    # ... (copy the current logic) ...
