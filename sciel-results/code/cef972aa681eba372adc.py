"""
Optimized search for rank-minimal tensor decomposition of 2x3 x 3x9 matrix multiplication.
Uses the Kronecker product structure T_{2,3,9} = T_{2,3,3} x T_{1,1,3}.
Searches for low-rank decompositions of T_{2,3,3} using randomized linear algebra.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    def verify_subtensor(U, V, W, n_sub, m_sub, p_sub):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = U_arr[i * m_sub + j, :]
                    v = V_arr[j * p_sub + k, :]
                    w = W_arr[k * n_sub + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def find_subtensor_decomposition(n_sub, m_sub, p_sub, target_rank, max_trials=5000):
        """
        Search for a decomposition of T_{n_sub, m_sub, p_sub} with rank target_rank.
        Uses random U, V and solves for W.
        """
        num_eqs = n_sub * m_sub * p_sub
        num_unk_W = p_sub * n_sub * target_rank
        
        # Pre-compute indices for the linear system A W = b
        # Equation for (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # We fix U and V, so we solve for W.
        # The system is: M * W_flat = 1
        # Where M is a matrix of size (num_eqs, num_unk_W)
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(12345 + trial)
            
            # Initialize U and V with small integers
            U = np.random.choice([-1, 0, 1], size=(n_sub * m_sub, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m_sub * p_sub, target_rank))
            
            # Build the linear system matrix A for W
            # A[row, col] where row corresponds to (i,j,k) and col to (w_idx, r)
            # w_idx = k * n_sub + i
            # col = r * (p_sub * n_sub) + w_idx ? No, W is (p*n, rank).
            # W_flat index = w_idx * rank + r
            
            A = np.zeros((num_eqs, num_unk_W))
            row = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        u_idx = i * m_sub + j
                        v_idx = j * p_sub + k
                        w_idx = k * n_sub + i
                        
                        # For each r, the coefficient of W[w_idx, r] is U[u_idx, r] * V[v_idx, r]
                        for r in range(target_rank):
                            coeff = U[u_idx, r] * V[v_idx, r]
                            if coeff != 0:
                                A[row, w_idx * target_rank + r] = coeff
                        row += 1
            
            # Solve A W_flat = 1
            try:
                # Use least squares
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(num_eqs), rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Reshape W_flat to (p_sub * n_sub, target_rank)
            W = W_flat.reshape((p_sub * n_sub, target_rank))
            
            # Check if W is close to half-integers
            W_snapped = snap_to_half_int(W)
            
            # Verify the snapped solution
            if verify_subtensor(U.tolist(), V.tolist(), W_snapped.tolist(), n_sub, m_sub, p_sub):
                return U.tolist(), V.tolist(), W_snapped.tolist()
            
            # Optional: If not snapped, try a local polish on U, V, W?
            # For now, just restart.
        
        return None

    # Search for T_{2,3,3} decomposition
    n2, m2, p2 = 2, 3, 3
    
    # Try ranks 15, 16, 17 (18 is schoolbook)
    for r in [15, 16, 17]:
        if time.time() > deadline - 10:
            break
            
        res = find_subtensor_decomposition(n2, m2, p2, r)
        if res is not None:
            U2, V2, W2 = res
            # Construct full T_{2,3,9}
            r_total = 3 * r
            U = [[0] * r_total for _ in range(n * m)]
            V = [[0] * r_total for _ in range(m * p)]
            W = [[0] * r_total for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        k1 = k % 3
                        k2 = k // 3
                        for c in range(r):
                            col = c * 3 + k2
                            U[i * m + j][col] = U2[i * m2 + j][c]
                            V[j * p + k][col] = V2[j * p2 + k1][c]
                            W[k * n + i][col] = W2[k1 * n2 + i][c]
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        # If r=15 failed, maybe try a more aggressive search for r=16?
        # The loop continues to next r.

    return schoolbook()
