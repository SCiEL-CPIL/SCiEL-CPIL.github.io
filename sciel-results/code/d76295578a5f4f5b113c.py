"""Structural change: Use alternating linear solves to find a low-rank decomposition.
For each candidate rank R, initialize randomly and iterate: fix two matrices, solve
for the third via linear least squares. Check validity at each step."""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify_numpy(U, V, W, n, m, p):
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _solve_for_U(V, W, n, m, p, R):
    """Given V and W, solve for U such that for all i,k,j,j':
    sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    
    For fixed (i,k), we need: sum_c U[i*m+j,c] * (W[k*n+i,c] * V[j'*p+k,c]) = delta_{j,j'}
    This is: U_i_row @ M_ik = I_m where U_i_row is the row of U for fixed i (m entries per j),
    and M_ik[j,c] = W[k*n+i,c] * V[j*p+k,c]
    
    So for each (i,k): U[i*m:(i+1)*m, :] @ M_ik = I_m
    where U[i*m:(i+1)*m, :] is an m x R matrix.
    """
    U = [[0] * R for _ in range(n * m)]
    for i in range(n):
        for k in range(p):
            # M_ik is m x R: M_ik[j, c] = W[k*n+i, c] * V[j*p+k, c]
            M_ik = np.zeros((m, R), dtype=np.float64)
            for j in range(m):
                for c in range(R):
                    M_ik[j, c] = W[k * n + i][c] * V[j * p + k][c]
            # Solve U_block @ M_ik = I_m, where U_block is m x R
            # This is: for each j, sum_c U_block[j,c] * M_ik[j2,c] = delta_{j,j2}
            # So U_block @ M_ik = I
            # U_block = I @ pinv(M_ik) = pinv(M_ik)
            # But we need U_block[j,c] such that sum_c U_block[j,c] * M_ik[j2,c] = delta_{j,j2}
            # This means U_block @ M_ik^T = I? Let me re-check.
            # sum_c U[i*m+j, c] * W[k*n+i, c] * V[j'*p+k, c] = delta_{j,j'}
            # Let A[j,c] = W[k*n+i,c] * V[j*p+k,c] (m x R)
            # Then sum_c U[i*m+j,c] * A[j',c] = delta_{j,j'}
            # This is: U_block @ A^T = I_m? No:
            # (U_block @ A^T)[j, j'] = sum_c U_block[j,c] * A^T[c,j'] = sum_c U_block[j,c] * A[j',c]
            # So yes: U_block @ A^T = I_m
            # U_block = I_m @ pinv(A^T) = pinv(A^T)
            # But pinv(A^T) has shape R x m, and U_block should be m x R.
            # Actually: U_block @ A^T = I_m => U_block = I_m @ (A^T)^+ where (A^T)^+ is the Moore-Penrose pseudoinverse
            # (A^T)^+ has shape R x m, so I_m @ (A^T)^+ is m x R. Yes!
            try:
                U_block = np.linalg.pinv(A.T)  # Wait, A is m x R, A.T is R x m, pinv(A.T) is m x R
                # Actually pinv(A.T) gives the pseudoinverse of A.T which is m x R
                # But we need U_block such that U_block @ A.T = I_m
                # If A.T is R x m, its pseudoinverse is m x R, and (A.T)^+ @ A.T = I_m if rank(A.T)=m
                # So U_block = pinv(A.T) works if A.T has full row rank m
                U_block = np.linalg.pinv(A.T)
            except Exception:
                U_block = np.zeros((m, R), dtype=np.float64)
            for j in range(m):
                for c in range(R):
                    U[i * m + j][c] = int(round(U_block[j, c]))
    return U


def _solve_for_V(U, W, n, m, p, R):
    """Given U and W, solve for V.
    sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    For fixed (i,k): let B[j,c] = W[k*n+i,c] * U[i*m+j,c] (m x R)
    Then sum_c B[j,c] * V[j'*p+k,c] = delta_{j,j'}
    This is: B @ V_block^T = I_m where V_block is m x R (rows indexed by j')
    (B @ V_block^T)[j, j'] = sum_c B[j,c] * V_block[j',c]
    So V_block^T = pinv(B) => V_block = pinv(B).T
    """
    V = [[0] * R for _ in range(m * p)]
    for k in range(p):
        for i in range(n):
            B = np.zeros((m, R), dtype=np.float64)
            for j in range(m):
                for c in range(R):
                    B[j, c] = W[k * n + i][c] * U[i * m + j][c]
            try:
                # V_block @ B^T = I_m? No:
                # sum_c B[j,c] * V_block[j',c] = delta_{j,j'}
                # (B @ V_block^T)[j,j'] = sum_c B[j,c] * V_block^T[c,j'] = sum_c B[j,c] * V_block[j',c]
                # So B @ V_block^T = I_m => V_block^T = pinv(B) => V_block = pinv(B).T
                V_block = np.linalg.pinv(B).T  # m x R
            except Exception:
                V_block = np.zeros((m, R), dtype=np.float64)
            for jp in range(m):
                for c in range(R):
                    V[jp * p + k][c] = int(round(V_block[jp, c]))
    return V


def _solve_for_W(U, V, n, m, p, R):
    """Given U and V, solve for W.
    sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    For fixed (i,k): let C[j,c] = U[i*m+j,c] * V[j*p+k,c] (m x R)
    Then sum_c W[k*n+i,c] * C[j,c] = delta_{j,j'} for all j,j'
    This is: W_row @ C = I_m where W_row is 1 x R? No:
    W[k*n+i,c] is a scalar for each c. So sum_c W[k*n+i,c] * C[j,c] = delta_{j,j'}
    This means: for each j, sum_c W[k*n+i,c] * C[j,c] = delta_{j,j'}
    So W_row @ C = e_j^T? No, it's W_row @ C = I_m? 
    W_row is 1 x R, C is m x R. W_row @ C^T is 1 x m.
    (W_row @ C^T)[j] = sum_c W_row[c] * C[j,c]
    We need this to equal delta_{j,j'} for a fixed j'. But we need it for all j,j' simultaneously.
    This means: for each j', we need sum_c W[k*n+i,c] * C[j,c] = delta_{j,j'}
    So the vector [sum_c W[c]*C[0,c], sum_c W[c]*C[1,c], ...] should be e_{j'}
    But this must hold for ALL j', which means W_row @ C^T must be the identity? That's m equations for R unknowns.
    Actually: we need sum_c W[k*n+i,c] * C[j,c] = delta_{j,j'} for ALL j and j'.
    This is m^2 equations. Let me re-index: for fixed (i,k), we have m^2 constraints on W[k*n+i, 0..R-1].
    Let D[j,j',c] = C[j,c] * delta... no.
    Actually: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    Let E[j,j',c] = U[i*m+j,c] * V[j'*p+k,c]
    Then sum_c W[k*n+i,c] * E[j,j',c] = delta_{j,j'}
    This is a linear system in W[k*n+i, :]: for each (j,j'), sum_c W[c] * E[j,j',c] = delta_{j,j'}
    So we have m^2 equations, R unknowns. Stack them:
    A is m^2 x R where A[(j,j'),c] = E[j,j',c] = U[i*m+j,c] * V[j'*p+k,c]
    b is m^2 vector where b[(j,j')] = delta_{j,j'}
    Solve A @ W_row = b via least squares.
    """
    W = [[0] * R for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            # Build A (m^2 x R) and b (m^2)
            A = np.zeros((m * m, R), dtype=np.float64)
            b = np.zeros(m * m, dtype=np.float64)
            idx = 0
            for j in range(m):
                for jp in range(m):
                    for c in range(R):
                        A[idx, c] = U[i * m + j][c] * V[jp * p + k][c]
                    b[idx] = 1.0 if j == jp else 0.0
                    idx += 1
            try:
                W_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                W_row = np.zeros(R, dtype=np.float64)
            for c in range(R):
                W[k * n + i][c] = int(round(W_row[c]))
    return W


def _random_init(n, m, p, R, rng):
    U = [[rng.choice([0, 1, -1]) for _ in range(R)] for _ in range(n * m)]
    V = [[rng.choice([0, 1, -1]) for _ in range(R)] for _ in range(m * p)]
    W = [[rng.choice([0, 1, -1]) for _ in range(R)] for _ in range(p * n)]
    return U, V, W


def _try_rank(R, n, m, p, deadline, rng):
    """Try to find a valid rank-R decomposition via alternating solves."""
    max_iters = 50
    for init in range(3):
        if time.time() > deadline - 20.0:
            return None
        U, V, W = _random_init(n, m, p, R, rng)
        for it in range(max_iters):
            if time.time() > deadline - 20.0:
                return None
            U = _solve_for_U(V, W, n, m, p, R)
            V = _solve_for_V(U, W, n, m, p, R)
            W = _solve_for_W(U, V, n, m, p, R)
            # Check validity
            if _verify_numpy(U, V, W, n, m, p):
                # Round to integers and re-verify
                U = [[int(round(x)) for x in row] for row in U]
                V = [[int(round(x)) for x in row] for row in V]
                W = [[int(round(x)) for x in row] for row in W]
                if _verify_numpy(U, V, W, n, m, p):
                    return U, V, W
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    rng = random.Random(42)

    # Try ranks from low to high
    for R in range(20, 121):
        if time.time() > deadline - 30.0:
            break
        result = _try_rank(R, n, m, p, deadline, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
