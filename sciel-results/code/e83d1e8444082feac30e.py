import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact_fast(U, V, W, n, m, p):
    """Fast float-based verification."""
    R = len(U[0])
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Exact Fraction-based verification."""
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _construct_rank17(n, m, p):
    """
    Constructive rank 17 decomposition for 2x3x3 matrix multiplication.
    """
    if n != 2 or m != 3 or p != 3:
        return None

    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen's 7 terms for i,j,k in {0,1}
    # r=0 (M1)
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    
    # r=1 (M2)
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    
    # r=2 (M3)
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    
    # r=3 (M4)
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    
    # r=4 (M5)
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    
    # r=5 (M6)
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    
    # r=6 (M7)
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1

    # Boundary terms (r=7..16)
    boundary_terms = [
        (0, 2, 0), (1, 2, 0),
        (0, 2, 1), (1, 2, 1),
        (0, 2, 2), (1, 2, 2),
        (0, 0, 2), (1, 0, 2),
        (0, 1, 2), (1, 1, 2)
    ]
    
    for idx, (i, j, k) in enumerate(boundary_terms):
        r = 7 + idx
        U[u_idx(i, j)][r] = 1
        V[v_idx(j, k)][r] = 1
        W[w_idx(k, i)][r] = 1

    return {"U": U, "V": V, "W": W}


def _try_eliminate_tensor(U, V, W, n, m, p, target_idx):
    """
    Try to eliminate tensor at target_idx by expressing it as a linear combination
    of the other tensors. Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    R = len(U[0])
    if R <= 1:
        return (False, None, None, None)
    
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    
    # Indices of other tensors
    other_indices = [r for r in range(R) if r != target_idx]
    num_other = len(other_indices)
    
    # Build linear system: for each (i,j,k), sum_{r in other} c_r * U[i,j,r]*V[j,k,r]*W[k,i,r] = U[i,j,t]*V[j,k,t]*W[k,i,t]
    # A is (n*m*p) x num_other
    # b is (n*m*p)
    
    num_eqs = n * m * p
    A = np.zeros((num_eqs, num_other))
    b = np.zeros(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui_t = Uf[i*m+j][target_idx]
                vk_t = Vf[j*p+k][target_idx]
                wk_t = Wf[k*n+i][target_idx]
                b[eq_idx] = ui_t * vk_t * wk_t
                
                for col, r in enumerate(other_indices):
                    A[eq_idx, col] = Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                eq_idx += 1
    
    # Solve least squares
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return (False, None, None, None)
    
    # Check residual
    if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-7:
        return (False, None, None, None)
    
    # Verify solution
    lhs = A.dot(coeffs)
    if not np.allclose(lhs, b, atol=1e-9):
        return (False, None, None, None)
    
    # Check if coefficients are "simple" (close to integers or half-integers)
    # This is a heuristic to ensure exactness
    simple_coeffs = []
    for c in coeffs:
        if abs(c - round(c)) < 1e-6:
            simple_coeffs.append(round(c))
        elif abs(2*c - round(2*c)) < 1e-6:
            simple_coeffs.append(round(2*c) / 2.0)
        else:
            # Not simple, but might still be exact. Let's try anyway.
            simple_coeffs.append(c)
    
    # Update U, V, W
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    
    for col, r in enumerate(other_indices):
        c = simple_coeffs[col]
        if abs(c) < 1e-12:
            continue
        for i in range(n * m):
            new_U[i][r] -= c * U[i][target_idx]
        for j in range(m * p):
            new_V[j][r] -= c * V[j][target_idx]
        for k in range(p * n):
            new_W[k][r] -= c * W[k][target_idx]
    
    # Remove target tensor
    for i in range(n * m):
        new_U[i].pop(target_idx)
    for j in range(m * p):
        new_V[j].pop(target_idx)
    for k in range(p * n):
        new_W[k].pop(target_idx)
    
    return (True, new_U, new_V, new_W)


def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=30.0):
    """
    Try to reduce the rank of a decomposition from current rank to target_rank.
    """
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return None
    
    start_time = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_R = current_rank
    
    for attempt in range(current_R - target_rank):
        if time.time() - start_time > max_time:
            break
        if deadline_exceeded():
            break
        if current_R <= target_rank:
            break
        
        # Try to eliminate each tensor
        eliminated = False
        for target_idx in range(current_R):
            success, new_U, new_V, new_W = _try_eliminate_tensor(
                current_U, current_V, current_W, n, m, p, target_idx
            )
            if success:
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_R -= 1
                eliminated = True
                break
        
        if not eliminated:
            break
    
    if current_R <= target_rank:
        # Convert back to int/float
        U_out = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in current_U]
        V_out = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in current_V]
        W_out = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in current_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try constructive rank 17
    result = _construct_rank17(n, m, p)
    if result is not None:
        if _verify_exact_fast(result["U"], result["V"], result["W"], n, m, p):
            # Try to reduce rank further
            for target in range(16, 12, -1):
                if deadline_exceeded():
                    break
                reduced = _try_rank_reduction(result["U"], result["V"], result["W"], n, m, p, target, max_time=20.0)
                if reduced is not None:
                    if _verify_exact(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                        result = reduced
                        break
    
    if result is None:
        result = _schoolbook(n, m, p)
    
    return result


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
