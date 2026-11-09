import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication.
    Uses structured search exploiting the bilinear form of matrix multiplication.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 115

    # Build target tensor: T[a,b,c] = 1 if (a,b,c) corresponds to a valid multiplication term
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0

    # Try ranks from lower bound upward
    # Lower bound: need at least n*p = 10 terms (one per output element)
    # But structural constraints may require more
    for rank in range(10, 41):
        if time.time() > deadline:
            break
        result = _try_rank_structured(n, m, p, nm, mp, pn, rank, target, deadline)
        if result is not None:
            return result

    # Fallback: schoolbook
    return _schoolbook(n, m, p, nm, mp, pn)

def _schoolbook(n, m, p, nm, mp, pn):
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_structured(n, m, p, nm, mp, pn, rank, target, deadline):
    """
    Find rank-r decomposition using the structure of matrix multiplication.
    Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    Each rank-1 term contributes u_a * v_b * w_c to T[a,b,c].
    
    Strategy: 
    1. Fix W randomly, solve for U and V via least squares
    2. Iterate: update each factor given the others
    3. Use the constraint that non-target entries must be zero
    """
    np.random.seed(42 + rank * 7)
    
    # Precompute target array for faster access
    target_arr = np.zeros((nm, mp, pn), dtype=np.float64)
    for (a, b, c), tval in target.items():
        target_arr[a, b, c] = tval
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Initialize W with structured pattern
        W = np.zeros((pn, rank), dtype=np.float64)
        # Try different initialization strategies
        if trial < 5:
            # Random sparse
            for r in range(rank):
                W[np.random.choice(pn, size=min(3, pn), replace=False), r] = 1.0
        elif trial < 10:
            # Structured: group by output index
            for r in range(rank):
                c = r % pn
                W[c, r] = 1.0
        elif trial < 15:
            # Random dense
            W = np.random.randn(pn, rank) * 0.5
        else:
            # Mixed
            W = np.random.randn(pn, rank) * 0.3
            W[np.random.choice(pn, size=rank, replace=False), np.arange(rank)] = 1.0
        
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        
        # Alternating minimization
        for it in range(300):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_r U[a,r] * (V[b,:] * W[c,:])_r = target_arr[a,b,c] for all b,c
            # This is a linear system: M_a * U[a,:] = rhs_a
            # where M_a[b,c,r] = V[b,r] * W[c,r] reshaped
            
            # Efficient approach: for fixed V, W, the constraint on U[a,:] is:
            # sum_{b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2 = 0
            # Gradient: for each r, d/dU[a,r] = sum_{b,c} 2*(resid[a,b,c]) * V[b,r] * W[c,r]
            
            # Use normal equations approach
            # For U: A_U * U[a,:] = b_U where A_U[r1,r2] = sum_{b,c} V[b,r1]*W[c,r1]*V[b,r2]*W[c,r2]
            # b_U[r1] = sum_{b,c} T[a,b,c] * V[b,r1] * W[c,r1]
            
            # Precompute for efficiency
            # M_VW = outer product structure
            # For each pair (b,c), the vector is V[b,:] * W[c,:]
            
            # Build A_U and b_U
            A_U = np.zeros((rank, rank), dtype=np.float64)
            b_U = np.zeros((nm, rank), dtype=np.float64)
            
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        tval = target_arr[a, b, c]
                        vw = V[b, :] * W[c, :]
                        A_U += np.outer(vw, vw)
                        b_U[a] += tval * vw
            
            # Solve A_U * U[a,:] = b_U[a,:] for each a
            try:
                # Add small regularization for numerical stability
                A_U_reg = A_U + 1e-10 * np.eye(rank)
                U = np.linalg.solve(A_U_reg, b_U.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(A_U, b_U.T, rcond=None)[0].T
            
            # Update V similarly
            A_V = np.zeros((rank, rank), dtype=np.float64)
            b_V = np.zeros((mp, rank), dtype=np.float64)
            
            for b in range(mp):
                for a in range(nm):
                    for c in range(pn):
                        tval = target_arr[a, b, c]
                        uw = U[a, :] * W[c, :]
                        A_V += np.outer(uw, uw)
                        b_V[b] += tval * uw
            
            try:
                A_V_reg = A_V + 1e-10 * np.eye(rank)
                V = np.linalg.solve(A_V_reg, b_V.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(A_V, b_V.T, rcond=None)[0].T
            
            # Update W similarly
            A_W = np.zeros((rank, rank), dtype=np.float64)
            b_W = np.zeros((pn, rank), dtype=np.float64)
            
            for c in range(pn):
                for a in range(nm):
                    for b in range(mp):
                        tval = target_arr[a, b, c]
                        uv = U[a, :] * V[b, :]
                        A_W += np.outer(uv, uv)
                        b_W[c] += tval * uv
            
            try:
                A_W_reg = A_W + 1e-10 * np.eye(rank)
                W = np.linalg.solve(A_W_reg, b_W.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A_W, b_W.T, rcond=None)[0].T
            
            # Check convergence
            err = _compute_error(U, V, W, target_arr, nm, mp, pn)
            if err < best_err:
                best_err = err
                if err < 1e-8:
                    # Try to round and verify
                    U_r, V_r, W_r = _round_to_half_int(U, V, W)
                    if _verify_exact(U_r, V_r, W_r, nm, mp, pn, target):
                        return {"U": U_r, "V": V_r, "W": W_r}
                    best_result = (U_r, V_r, W_r, err)
        
        # After each trial, check if we got close enough
        if best_err < 1e-6 and best_result is not None:
            U_r, V_r, W_r, _ = best_result
            if _verify_exact(U_r, V_r, W_r, nm, mp, pn, target):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _compute_error(U, V, W, target_arr, nm, mp, pn):
    """Compute the error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                err += (s - target_arr[a, b, c]) ** 2
    return err

def _round_to_half_int(U, V, W, max_denom=4):
    """Round entries to multiples of 1/max_denom."""
    def round_mat(M):
        M_r = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                best = 0
                best_err = abs(val)
                for den in range(1, max_denom + 1):
                    for num in range(-50, 51):
                        cand = num / den
                        err = abs(val - cand)
                        if err < best_err:
                            best_err = err
                            best = cand
                M_r[i, j] = best
        return M_r.tolist()
    
    U_r = round_mat(U)
    V_r = round_mat(V)
    W_r = round_mat(W)
    return U_r, V_r, W_r

def _verify_exact(U, V, W, nm, mp, pn, target, tol=1e-9):
    rank = len(U[0])
    target_set = set(target.keys())
    
    # Check target entries
    for (a, b, c), tval in target.items():
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s - tval) > tol:
            return False
    
    # Check non-target entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in target_set:
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
    return True
# EVOLVE-BLOCK-END
