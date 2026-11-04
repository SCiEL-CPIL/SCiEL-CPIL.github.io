"""Combined program: structured Strassen tiling (rank 77) + fast rank reduction with incremental tracking."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 115.0  # safety margin before 120s evaluator limit

def _build_structured_rank77():
    """Build the proven rank-77 nested Strassen tiling decomposition."""
    n, m, p = 4, 4, 6
    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=6, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    return np.max(np.abs(T - T_expected)) < tol


def _build_expected_tensor(n=4, m=4, p=6):
    """Build the expected matrix multiplication tensor."""
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    return T_expected


def _compute_tensor(U, V, W, n=4, m=4, p=6):
    """Compute the tensor from factor matrices."""
    R = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return T


def _rank_reduce_fast(U, V, W, tol=1e-8, deadline=None):
    """Fast rank reduction: greedily try to zero columns and check feasibility."""
    n, m, p = 4, 4, 6
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    R = U.shape[1]
    
    T_expected = _build_expected_tensor(n, m, p)
    
    # Precompute column contributions
    def get_tensor_subset(indices):
        """Compute tensor from a subset of columns."""
        T = np.zeros((n*m, m*p, p*n))
        for r in indices:
            T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return T
    
    active = list(range(R))
    
    # Greedy removal: try removing each column, check if remaining still works
    # This is O(R^2 * cost_of_tensor_check) which is expensive but let's try
    # with early termination
    
    for iteration in range(R):
        if deadline and time.time() > deadline:
            break
        if len(active) <= 1:
            break
        
        removed = False
        for idx in range(len(active) - 1, -1, -1):
            if deadline and time.time() > deadline:
                break
            
            r = active[idx]
            # Try removing column r
            trial_active = active[:idx] + active[idx+1:]
            T_trial = get_tensor_subset(trial_active)
            err = np.max(np.abs(T_trial - T_expected))
            
            if err < tol:
                active = trial_active
                removed = True
                break  # restart from beginning
    
        if not removed:
            break
    
    current_rank = len(active)
    U_red = U[:, active]
    V_red = V[:, active]
    W_red = W[:, active]
    
    return U_red, V_red, W_red, current_rank


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Step 1: Build structured rank-77 decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook if something went wrong
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 2: Try fast rank reduction
    deadline = t_start + _DEADLINE
    try:
        U_red, V_red, W_red, new_rank = _rank_reduce_fast(U, V, W, deadline=deadline)
        
        # Verify reduced decomposition
        if _verify_decomposition(U_red, V_red, W_red):
            U, V, W = U_red, V_red, W_red
    except Exception:
        pass  # Keep original if reduction fails
    
    # Convert to list with integer entries where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
