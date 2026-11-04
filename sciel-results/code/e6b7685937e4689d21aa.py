import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # The tensor T has entries T[i, j, k] = 1 if i, j, k are the indices for C[i,k] = sum_j A[i,j]B[j,k]
    # Wait, the standard matrix multiplication tensor T_{ijk} corresponds to A_{ij} B_{jk} C_{ik}.
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the third index of the tensor is mapped to $W$ row index $k*n+i$.
    # So the tensor entry $T_{i,j,k}$ (where $i$ is row of A, $j$ is col of A/row of B, $k$ is col of B)
    # is associated with $U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
    # The sum over $r$ of $U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$ should equal 1 if the indices match the multiplication pattern?
    # Actually, the tensor of matrix multiplication is defined such that
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The tensor $T$ is a multilinear map.
    # The decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # The entry $T_{i,j,k}$ in the standard basis is 1 if the term $A_{ij}B_{jk}$ contributes to $C_{ik}$?
    # No, the tensor $T$ is defined by $T(A, B) = A B$.
    # In terms of basis vectors $e_i \otimes e_j \otimes e_k$, the coefficient is 1 if $k$ is the column index of $C$ and $i$ is the row index of $C$ and $j$ is the intermediate index.
    # Specifically, $T(e_i, e_j, e_k) = 1$ if we are computing the $(i,k)$ entry of the product, which involves summing over $j$.
    # So for a fixed $i, k$, there are $m$ terms (one for each $j$).
    # So $T_{i,j,k} = 1$ for all $i \in [0, n), j \in [0, m), k \in [0, p)$.
    # Wait, if $T_{i,j,k} = 1$ for all $i,j,k$, then the tensor is the all-ones tensor.
    # That would mean $C_{ik} = \sum_j A_{ij} B_{jk} \times 1$?
    # Yes, the structure tensor of matrix multiplication $M_{n,m,p}$ has entries 1 at positions $(i, j, k)$ corresponding to the term $A_{ij} B_{jk}$ contributing to $C_{ik}$.
    # So for every $i, j, k$, the entry is 1.
    # So we need $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all $i, j, k$.
    
    # Let's verify the indices.
    # U shape: (n*m, rank). Index: i*m + j.
    # V shape: (m*p, rank). Index: j*p + k.
    # W shape: (p*n, rank). Index: k*n + i.
    
    # The equation is:
    # For all i in [0, n), j in [0, m), k in [0, p):
    #   sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # This is a system of $n*m*p$ equations.
    # For 3x3x3, 27 equations.
    
    # We want to find U, V, W with minimal rank.
    
    def get_residuals(params, rank):
        # params is flattened U, V, W
        # U: n*m*rank
        # V: m*p*rank
        # W: p*n*rank
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank : ]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :]) * 0 # Wrong, it's element-wise product sum
                    # val = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    val = np.einsum('r,r,r->', U[u_idx, :], V[v_idx, :], W[w_idx, :])
                    residuals.append(val - 1.0)
        return np.array(residuals)

    def try_rank(rank, max_restarts=20):
        start_time = time.time()
        time_limit = 110.0 # Safety margin
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            # Scale matters. If we use uniform [0,1], the product might be small.
            # Let's use normal distribution.
            np.random.seed(restart * 123 + rank)
            U = np.random.normal(0, 1, (n*m, rank))
            V = np.random.normal(0, 1, (m*p, rank))
            W = np.random.normal(0, 1, (p*n, rank))
            
            # We can fix one column to remove scaling symmetry?
            # Or just let the solver handle it.
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use least squares
            try:
                res = least_squares(get_residuals, params, args=(rank,), method='lm', max_nfev=1000)
                if res.success and np.max(np.abs(res.fun)) < 1e-8:
                    U_sol = res.x[:n*m*rank].reshape((n*m, rank))
                    V_sol = res.x[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
                    W_sol = res.x[n*m*rank + m*p*rank : ].reshape((p*n, rank))
                    
                    # Check for integer/half-integer
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                val = np.einsum('r,r,r->', U_snap[u_idx, :], V_snap[v_idx, :], W_snap[w_idx, :])
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                            
                    if max_err < 1e-9:
                        return U_snap, V_snap, W_snap
                    else:
                        # Return float solution if it's very good
                        if np.max(np.abs(res.fun)) < 1e-10:
                            return U_sol, V_sol, W_sol
            except Exception:
                pass
        return None

    # Try ranks from 19 to 27
    for rank in range(19, 28):
        sol = try_rank(rank)
        if sol is not None:
            U, V, W = sol
            # Convert to lists of lists for the output format
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
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
