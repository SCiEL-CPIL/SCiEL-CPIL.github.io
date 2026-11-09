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

def _verify_exact(U, V, W, n, m, p):
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
    if n != 2 or m != 3 or p != 3:
        return None
    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    # Strassen 7 terms
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1
    # Boundary 10 terms
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

def _search_rank16(n, m, p, time_budget=100.0):
    # Dimensions
    n_u = n * m # 6
    n_v = m * p # 9
    n_w = p * n # 6
    R = 16
    
    # Target tensor values
    # T[i, j, k] = 1 if i,j,k are valid, else 0?
    # Actually T[i, j, k] is 1 for all i,j,k in range.
    # The equation is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # We fix U randomly and solve for V, W using ALS with rank-1 constraint
    # Variables: X_r is n_v x n_w matrix.
    # Equation: sum_r U[i*m+j, r] X_r[j*p+k, k*n+i] = 1
    
    # Build linear system matrix A and vector b
    # Unknowns: x_{r, beta, gamma}
    # Index mapping for unknowns:
    # idx = r * n_v * n_w + beta * n_w + gamma
    
    num_unknowns = R * n_v * n_w
    A = np.zeros((n * m * p, num_unknowns))
    b = np.ones(n * m * p)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Equation: sum_r U[i*m+j, r] X_r[j*p+k, k*n+i] = 1
                for r in range(R):
                    beta = j * p + k
                    gamma = k * n + i
                    unk_idx = r * n_v * n_w + beta * n_w + gamma
                    A[eq_idx, unk_idx] = 1.0 # Coefficient is U[i*m+j, r] which we will multiply later?
                    # Wait, U is fixed. So A should contain U values.
                    # Let's re-construct A after U is fixed.
                eq_idx += 1
                
    # Since A depends on U, we can't build it once.
    # We will build A inside the loop or pre-compute the structure.
    # Structure: For each eq (i,j,k), and each r, the variable is X_r[j*p+k, k*n+i].
    # The coefficient is U[i*m+j, r].
    
    def solve_als(U_np, max_iter=100):
        # U_np is n_u x R
        # Build A
        A = np.zeros((n * m * p, num_unknowns))
        b = np.ones(n * m * p)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        beta = j * p + k
                        gamma = k * n + i
                        unk_idx = r * n_v * n_w + beta * n_w + gamma
                        A[eq_idx, unk_idx] = U_np[i * m + j, r]
                    eq_idx += 1
        
        V_np = np.zeros((n_v, R))
        W_np = np.zeros((n_w, R))
        
        for it in range(max_iter):
            if deadline_exceeded():
                break
            # 1. Solve for X
            # X is R x n_v x n_w
            # Minimize || A x - b ||^2
            # x is vectorized X
            try:
                x_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                break
            
            X = x_sol.reshape(R, n_v, n_w)
            
            # 2. Update V, W
            new_V = np.zeros((n_v, R))
            new_W = np.zeros((n_w, R))
            
            for r in range(R):
                # SVD of X[r]
                U_s, s, Vt = np.linalg.svd(X[r], full_matrices=False)
                # Rank 1 approximation
                if len(s) == 0:
                    continue
                sigma = s[0]
                if sigma < 1e-12:
                    continue
                # V[:, r] = sqrt(sigma) * U_s[:, 0]
                # W[:, r] = sqrt(sigma) * Vt[0, :]
                new_V[:, r] = np.sqrt(sigma) * U_s[:, 0]
                new_W[:, r] = np.sqrt(sigma) * Vt[0, :]
            
            V_np = new_V
            W_np = new_W
            
            # Check error
            # Reconstruct T
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                        err += (val - 1.0)**2
            if err < 1e-6:
                return U_np, V_np, W_np
                
        return None

    # Random search
    best_res = None
    start_time = time.time()
    np.random.seed(42)
    
    while time.time() - start_time < time_budget and not deadline_exceeded():
        # Generate random sparse U
        # Each column of U has 2-4 non-zeros
        U_np = np.zeros((n_u, R))
        for r in range(R):
            num_nz = np.random.choice([2, 3, 4])
            indices = np.random.choice(n_u, num_nz, replace=False)
            for idx in indices:
                U_np[idx, r] = np.random.choice([-1, 1])
        
        res = solve_als(U_np, max_iter=50)
        if res:
            U_np, V_np, W_np = res
            # Try to round to integers/half-integers
            # Scale to make largest entry 1?
            # Or just check if entries are close to half-integers
            U_rounded = np.round(U_np * 2) / 2.0
            V_rounded = np.round(V_np * 2) / 2.0
            W_rounded = np.round(W_np * 2) / 2.0
            
            # Verify exact
            if _verify_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
    return None

def construct():
    n, m, p = DIMENSIONS
    # First try rank 16
    res16 = _search_rank16(n, m, p, time_budget=100.0)
    if res16:
        return res16
        
    # Fallback to rank 17
    res17 = _construct_rank17(n, m, p)
    if res17:
        return res17
        
    return _schoolbook(n, m, p)

def main():
    return construct()

if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
