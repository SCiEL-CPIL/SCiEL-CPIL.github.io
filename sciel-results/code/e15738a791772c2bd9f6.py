import numpy as np
import time

def _build_77():
    """Build the rank-77 decomposition using nested Strassen + schoolbook."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)
    
    # Strassen coefficients for 2x2 x 2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                ia = alpha[si].get(a, 0)
                if ia != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        sa = alpha[s].get(r, 0)
                        if sa != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += ia * sa
            for b in range(4):
                ib = beta[si].get(b, 0)
                if ib != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        sb = beta[s].get(c, 0)
                        if sb != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += ib * sb
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    sa = alpha[s].get(r, 0)
                    if sa != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += sa
                for c in range(4):
                    sb = beta[s].get(c, 0)
                    if sb != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += sb
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=6):
    """Verify that U @ diag(1) @ V^T reconstructs the multiplication tensor via W."""
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i is in [0, n*m), j is in [0, m*p), k is in [0, p*n)
    # Output index k = col*4 + row (W[k*n+i,r] convention means k indexes output)
    # Actually the convention is W[k*n+i, r] where output C[i,j] uses
    # sum over intermediate: C[i,j] = sum_k A[i,k]*B[k,j]
    # Tensor form: T[A_idx, B_idx, C_idx] = sum_r U[A_idx,r]*V[B_idx,r]*W[C_idx,r]
    # A_idx = i*m + k, B_idx = k*p + j, C_idx = j*n + i
    # But the problem says W[k*n+i,r] which is output index = j*n + i where j is col, i is row
    
    # Let's verify with random matrices
    A = np.random.randint(-5, 6, (n, m))
    B = np.random.randint(-5, 6, (m, p))
    C_expected = A @ B
    
    # Compute via decomposition
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        # U row for A[i,k] is at index i*m + k
        # V row for B[k,j] is at index k*p + j
        # W row for C[i,j] is at index j*n + i
        for i in range(n):
            for k in range(m):
                u_val = U[i*m + k, r]
                if u_val == 0:
                    continue
                for j in range(p):
                    v_val = V[k*p + j, r]
                    if v_val == 0:
                        continue
                    w_val = W[j*n + i, r]
                    C_computed[i, j] += u_val * v_val * w_val
    
    return np.array_equal(C_expected, C_computed)

def _try_rank_reduction(U, V, W, target_rank, max_time=60):
    """Try to reduce rank by identifying and removing dependent columns."""
    start = time.time()
    R = U.shape[1]
    
    # Build the full tensor for each candidate column
    n, m, p = 4, 4, 6
    # For each rank-1 component, the contribution is outer(u_col, v_col, w_col)
    # Flatten: contribution[r] = U[:,r] * V[:,r] * W[:,r] reshaped
    
    # We want to find the smallest subset of columns that spans the tensor
    # Use a greedy approach: start with all, try removing one at a time
    
    def tensor_from_cols(U_s, V_s, W_s, cols):
        """Reconstruct tensor from selected columns."""
        # Full tensor is (n*m) x (m*p) x (p*n)
        # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
        # Instead of full tensor, verify with random probes
        T = np.zeros((n*m, m*p, p*n))
        for r in cols:
            T += np.einsum('i,j,k->ijk', U_s[:,r], V_s[:,r], W_s[:,r])
        return T
    
    # Full tensor
    all_cols = list(range(R))
    T_full = tensor_from_cols(U, V, W, all_cols)
    
    # Try removing columns one by one
    current_cols = list(all_cols)
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for idx in range(len(current_cols)):
            test_cols = current_cols[:idx] + current_cols[idx+1:]
            if len(test_cols) >= target_rank:
                T_test = tensor_from_cols(U, V, W, test_cols)
                if np.array_equal(T_full, T_test):
                    current_cols = test_cols
                    improved = True
                    break
    
    return current_cols

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = 110.0  # safety margin
    
    U, V, W = _build_77()
    
    # Verify correctness
    if not _verify(U, V, W):
        raise ValueError("Decomposition failed verification")
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        cols = _try_rank_reduction(U, V, W, 70, max_time=30)
        if len(cols) < U.shape[1]:
            U = U[:, cols]
            V = V[:, cols]
            W = W[:, cols]
            # Re-verify
            if not _verify(U, V, W):
                # Revert
                U, V, W = _build_77()
    
    # Convert to lists with integer entries
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                # Check for half-integers
                xf = float(x)
                if xf == int(xf):
                    new_row.append(int(xf))
                else:
                    # Check if it's a half-integer
                    xf2 = xf * 2
                    if xf2 == int(xf2):
                        new_row.append(xf2 / 2)
                    else:
                        new_row.append(xf)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
