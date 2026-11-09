"""Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U': (6, rank), 'V': (12, rank), 'W': (8, rank)} with integer/half-integer entries.
    The output follows the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build the target tensor T[i, j, k] = 1
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_integer_matrix(M):
        """Round matrix to nearest half-integer."""
        return np.round(2.0 * np.array(M, dtype=float)) / 2.0
    
    # Try rank 21 first (known optimal)
    best_result = None
    
    for rank in [21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        n_trials = 50 if rank == 21 else 20
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                    break
                    
            rng = np.random.RandomState(trial * 1000 + rank)
            
            # Initialize with small random values
            U = rng.randn(nm, rank) * 0.5
            V = rng.randn(mp, rank) * 0.5
            W = rng.randn(pn, rank) * 0.5
            
            # Alternating least squares
            # For fixed V, W: solve for U
            # For fixed U, W: solve for V
            # For fixed U, V: solve for W
            
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: for each (i,j), solve
                # U[i*m+j, :] * sum_r V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p+0, :].*W[0*n+i, :] + V[j*p+1, :].*W[1*n+i, :] + ...) = 1
                # Actually, for fixed V and W, the equation for U[i*m+j, :] is:
                # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # Let M_u[i*m+j, k] = sum_r V[j*p+k, r] * W[k*n+i, r]
                # Then U[i*m+j, :] @ M_u[i*m+j, :]^T = 1
                # So U[i*m+j, :] = ones / M_u[i*m+j, :]^T (if invertible)
                
                for i in range(n):
                    for j in range(m):
                        # Build the matrix for U[i*m+j, :]
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve: U[i*m+j, :] @ M^T = ones(p)
                        # M @ U[i*m+j, :]^T = ones(p)
                        try:
                            U[i*m+j, :] = np.linalg.solve(M, np.ones(p))
                        except np.linalg.LinAlgError:
                            U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                
                # Update V: for each (j,k), solve
                # V[j*p+k, :] @ (U[i*m+j, :].*W[k*n+i, :])^T = 1 for all i
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.solve(M, np.ones(n))
                        except np.linalg.LinAlgError:
                            V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                
                # Update W: for each (k,i), solve
                # W[k*n+i, :] @ (U[i*m+j, :].*V[j*p+k, :])^T = 1 for all j
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.solve(M, np.ones(m))
                        except np.linalg.LinAlgError:
                            W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                
                # Check if converged
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += abs(s - 1.0)
                if err < 1e-10:
                    break
            
            # Round to half-integers
            U_h = to_half_integer_matrix(U)
            V_h = to_half_integer_matrix(V)
            W_h = to_half_integer_matrix(W)
            
            # Verify the rounded solution
            if verify(U_h, V_h, W_h):
                # Found a valid half-integer decomposition!
                best_result = (rank, U_h.tolist(), V_h.tolist(), W_h.tolist())
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
