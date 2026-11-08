import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Schoolbook fallback
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find lower rank
    # Target ranks: lower bound is max(n*m, m*p, p*n) = 12
    for R in range(12, n * m * p):
        if time.time() > deadline - 10:
            break
            
        # Run a few random restarts
        found = False
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                    
                # Solve for W given U, V
                # Equation: sum_r U_{r, i, j} V_{r, j, k} W_{r, i, k} = 1
                # For fixed i, k: sum_j (sum_r U_{r, i, j} V_{r, j, k}) W_{r, i, k} = 1?
                # No, the sum is over r.
                # Let A_{r, i, j} = U_{r, i, j}, B_{r, j, k} = V_{r, j, k}, C_{r, i, k} = W_{r, i, k}.
                # We want sum_r A_{r, i, j} B_{r, j, k} C_{r, i, k} = 1 for all i, j, k.
                
                # This is not a simple linear system for W because W is multiplied by U*V which depends on j.
                # Wait, for a fixed r, the term is U_{r, i, j} V_{r, j, k} W_{r, i, k}.
                # This is not separable in a way that allows simple ALS for the "all-ones" target unless we reshape.
                
                # Actually, we can view this as:
                # Let M_{i, k} be the matrix we want to reconstruct? No.
                # The condition is element-wise equality of the tensor T and the sum of rank-1 tensors.
                
                # Let's check if we can linearize.
                # If we fix U and V, the equation for W_{r, i, k} is:
                # sum_r' W_{r', i, k} * (U_{r', i, j} V_{r', j, k}) = 1?
                # No, the index r is the same for U, V, W.
                # So for a specific i, j, k:
                # sum_{r=1}^R (U_{r, i, j} V_{r, j, k}) W_{r, i, k} = 1.
                # This is a linear equation in W_{r, i, k} for all r?
                # Yes! For fixed i, j, k, let $A_{r} = U_{r, i, j} V_{r, j, k}$.
                # Then $\sum_r A_r W_{r, i, k} = 1$.
                # This is one equation.
                # We have $n \cdot m \cdot p$ equations.
                # Unknowns are $W_{r, i, k}$ for $r=1..R, i=1..n, k=1..p$. Total $R \cdot n \cdot p$.
                # If $R \cdot n \cdot p \ge n \cdot m \cdot p$, i.e., $R \ge m$, we might have a solution.
                # Here $m=3$. So if $R \ge 3$, we have enough degrees of freedom?
                # But the equations are coupled.
                
                # We can set up a linear system for W.
                # Variables: $W_{r, i, k}$.
                # Equations: For each $(i, j, k)$, $\sum_r (U_{r, i, j} V_{r, j, k}) W_{r, i, k} = 1$.
                
                # Let's build the matrix for this system.
                # Number of equations: $N_{eq} = n \cdot m \cdot p = 24$.
                # Number of variables: $N_{var} = R \cdot n \cdot p = R \cdot 8$.
                # If $R=12$, $N_{var} = 96$. $N_{eq} = 24$. Underdetermined.
                # We can use least squares.
                
                # Construct matrix A_sys of shape (24, 96)
                # Row index: flat index of (i, j, k)
                # Col index: flat index of (r, i, k)
                
                A_sys = np.zeros((n * m * p, R * n * p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for r in range(R):
                                col = r * n * p + k * n + i
                                A_sys[row, col] = U[i * m + j, r] * V[j * p + k, r]
                
                b_vec = np.ones(n * m * p)
                
                # Solve A_sys @ W_flat = b_vec
                W_flat, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                W = W_flat.reshape(R, n, p).T # W[r, i, k] -> W[i, k, r]?
                # Our W matrix is (p*n, R). Index k*n+i.
                # So W_mat[k*n+i, r] = W[r, i, k].
                W_mat = np.zeros((p * n, R))
                for r in range(R):
                    for i in range(n):
                        for k in range(p):
                            W_mat[k * n + i, r] = W[r, i, k]
                
                # Check error
                # Compute T_est_{i, j, k} = sum_r U_{r, i, j} V_{r, j, k} W_{r, i, k}
                # We can check this by reconstructing.
                # Error = sum (T_est - 1)^2
                # If error < 1e-9, we found a solution.
                
                # To save time, only check error every few iterations or at the end.
                if it % 5 == 0:
                    T_est = np.zeros((n, m, p))
                    for r in range(R):
                        # U_r is (n, m), V_r is (m, p), W_r is (p, n) -> W_{r, i, k}
                        # Term is U_{r, i, j} * V_{r, j, k} * W_{r, i, k}
                        # This is element-wise product.
                        # We can compute this efficiently?
                        # For each r, we have 3 matrices.
                        # We need to broadcast them.
                        # U_r[i, j], V_r[j, k], W_r[i, k].
                        # This is a 3-way contraction.
                        # We can do:
                        # Temp[i, j, k] = U_r[i, j] * V_r[j, k] * W_r[i, k]
                        # This is not a standard matmul.
                        # But for small sizes, loop is fine.
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_est[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[r, i, k]
                    
                    err = np.sum((T_est - 1) ** 2)
                    if err < 1e-9:
                        # Found valid decomposition
                        # Convert to list of lists
                        return {"U": U.tolist(), "V": V.tolist(), "W": W_mat.tolist()}
            
            # End restart loop
        # End R loop

    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
