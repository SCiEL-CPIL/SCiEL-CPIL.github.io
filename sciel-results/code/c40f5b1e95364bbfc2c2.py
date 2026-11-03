"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul. Verified numerically."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def _verify_222():
    U, V, W = _strassen_222()
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(7):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _build_2x5x6():
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    sU, sV, sW = _strassen_222()
    col = 0
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            U[0, col] = sU[0, r]; U[1, col] = sU[1, r]
            U[5, col] = sU[2, r]; U[6, col] = sU[3, r]
            V[k_offset, col] = sV[0, r]; V[k_offset+1, col] = sV[1, r]
            V[6+k_offset, col] = sV[2, r]; V[7+k_offset, col] = sV[3, r]
            W[k_offset*2, col] = sW[0, r]; W[k_offset*2+1, col] = sW[1, r]
            W[k_offset*2+2, col] = sW[2, r]; W[k_offset*2+3, col] = sW[3, r]
            col += 1
    for i in range(2):
        for k in range(6):
            U[i*5+2, col] = 1
            V[2*6+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            U[3, col] = sU[0, r]; U[4, col] = sU[1, r]
            U[8, col] = sU[2, r]; U[9, col] = sU[3, r]
            V[3*6+k_offset, col] = sV[0, r]; V[3*6+k_offset+1, col] = sV[1, r]
            V[4*6+k_offset, col] = sV[2, r]; V[4*6+k_offset+1, col] = sV[3, r]
            W[k_offset*2, col] = sW[0, r]; W[k_offset*2+1, col] = sW[1, r]
            W[k_offset*2+2, col] = sW[2, r]; W[k_offset*2+3, col] = sW[3, r]
            col += 1
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_full(U, V, W, n=2, m=5, p=6):
    rank = len(U[0])
    Ua = np.array(U); Va = np.array(V); Wa = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err += (total - 1.0) ** 2
    return err

def _als_decompose(n, m, p, target_rank, U0=None, V0=None, W0=None, max_iters=200, deadline=None, tol=1e-10):
    """Alternating least squares for tensor decomposition."""
    if U0 is not None:
        U = U0.copy(); V = V0.copy(); W = W0.copy()
    else:
        rng = np.random.default_rng(42)
        U = rng.standard_normal((n*m, target_rank)) * 0.3
        V = rng.standard_normal((m*p, target_rank)) * 0.3
        W = rng.standard_normal((p*n, target_rank)) * 0.3
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for row in range(n*m):
            i = row // m; j = row % m
            A_mat = np.zeros((p, target_rank))
            for k in range(p):
                A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
            except:
                pass
        
        # Update V
        for row in range(m*p):
            j = row // p; k = row % p
            A_mat = np.zeros((n, target_rank))
            for i in range(n):
                A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
            except:
                pass
        
        # Update W
        for row in range(p*n):
            k = row // n; i = row % n
            A_mat = np.zeros((m, target_rank))
            for j in range(m):
                A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W[row, :], _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
            except:
                pass
        
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Snap to half-integers and verify."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    err = _compute_error(U_s, V_s, W_s, n, m, p)
    if err < tol:
        return U_s, V_s, W_s, True
    return None, None, None, False

def _try_reduce_from_strassen(n, m, p, start_rank=54, deadline=None):
    """Try to reduce rank starting from the Strassen decomposition."""
    # Get the rank-54 solution
    base = _build_2x5x6()
    U54 = np.array(base['U'])
    V54 = np.array(base['V'])
    W54 = np.array(base['W'])
    
    for target_rank in range(start_rank - 1, max(start_rank - 10, 40), -1):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Strategy 1: Remove columns and re-optimize
        # Try removing each possible column
        found = False
        for remove_col in range(start_rank):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            # Remove this column
            keep = [c for c in range(start_rank) if c != remove_col]
            U_init = U54[:, keep]
            V_init = V54[:, keep]
            W_init = W54[:, keep]
            
            # Add a small random perturbation to help escape local minima
            rng = np.random.default_rng(remove_col)
            U_init = U_init + rng.standard_normal((n*m, target_rank)) * 0.05
            V_init = V_init + rng.standard_normal((m*p, target_rank)) * 0.05
            W_init = W_init + rng.standard_normal((p*n, target_rank)) * 0.05
            
            U_opt, V_opt, W_opt, err = _als_decompose(
                n, m, p, target_rank, U_init, V_init, W_init,
                max_iters=100, deadline=deadline - 2
            )
            
            if err < 1e-8:
                U_s, V_s, W_s, valid = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
                if valid:
                    return U_s, V_s, W_s, target_rank
        
        # Strategy 2: Full random restart with target_rank
        if not found:
            for attempt in range(3):
                if deadline is not None and time.time() > deadline - 3:
                    break
                U_opt, V_opt, W_opt, err = _als_decompose(
                    n, m, p, target_rank,
                    max_iters=150, deadline=deadline - 2
                )
                if err < 1e-8:
                    U_s, V_s, W_s, valid = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
                    if valid:
                        return U_s, V_s, W_s, target_rank
    
    return None, None, None, start_rank

def _try_border_rank_approach(n, m, p, deadline=None):
    """
    Border rank technique: perturb the tensor slightly, find a lower-rank 
    decomposition for the perturbed tensor, then snap back.
    
    The idea: the matrix multiplication tensor might have border rank < rank.
    We add small random perturbations to the target values and try to find
    a decomposition that works for the perturbed problem, then snap to 
    half-integers.
    """
    # Start from rank-54 and try to compress
    base = _build_2x5x6()
    U54 = np.array(base['U'])
    V54 = np.array(base['V'])
    W54 = np.array(base['W'])
    
    best_result = None
    best_rank = 54
    
    for target_rank in range(53, 45, -1):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Multiple restarts with different perturbations of the Strassen solution
        for seed in range(5):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            rng = np.random.default_rng(seed * 1000 + target_rank)
            
            # Initialize by taking first target_rank columns of the rank-54 solution
            # with random perturbations
            cols = rng.choice(54, size=target_rank, replace=False)
            U_init = U54[:, cols].copy()
            V_init = V54[:, cols].copy()
            W_init = W54[:, cols].copy()
            
            # Add structured perturbation
            scale = 0.1
            U_init += rng.standard_normal((n*m, target_rank)) * scale
            V_init += rng.standard_normal((m*p, target_rank)) * scale
            W_init += rng.standard_normal((p*n, target_rank)) * scale
            
            U_opt, V_opt, W_opt, err = _als_decompose(
                n, m, p, target_rank, U_init, V_init, W_init,
                max_iters=200, deadline=deadline - 2
            )
            
            if err < 1e-7:
                U_s, V_s, W_s, valid = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
                if valid:
                    best_result = (U_s, V_s, W_s)
                    best_rank = target_rank
                    break
        else:
            continue
        break
    
    if best_result is not None:
        return best_result[0], best_result[1], best_result[2], best_rank
    return None, None, None, 54

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First verify and return the known rank-54 decomposition
    try:
        if _verify_222():
            result = _build_2x5x6()
            if _verify_full(result['U'], result['V'], result['W']):
                # Check half-integer constraint
                ok = True
                for mat_name in ['U', 'V', 'W']:
                    for row in result[mat_name]:
                        for val in row:
                            if abs(val * 2 - round(val * 2)) > 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    best_result = result
                    best_rank = 54
                    
                    # Try to improve using border rank / compression approach
                    if time.time() < deadline - 10:
                        U_new, V_new, W_new, new_rank = _try_border_rank_approach(
                            n, m, p, deadline=deadline - 5
                        )
                        if U_new is not None and new_rank < best_rank:
                            best_result = {
                                "U": U_new.tolist(),
                                "V": V_new.tolist(),
                                "W": W_new.tolist()
                            }
                            best_rank = new_rank
                    
                    # Try further reduction from the best found
                    if best_rank > 48 and time.time() < deadline - 10:
                        U_new, V_new, W_new, new_rank = _try_reduce_from_strassen(
                            n, m, p, start_rank=best_rank, deadline=deadline - 5
                        )
                        if U_new is not None and new_rank < best_rank:
                            best_result = {
                                "U": U_new.tolist(),
                                "V": V_new.tolist(),
                                "W": W_new.tolist()
                            }
                            best_rank = new_rank
                    
                    return best_result
    except Exception:
        pass
    
    # Fall back to schoolbook rank 60
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
# EVOLVE-BLOCK-END
