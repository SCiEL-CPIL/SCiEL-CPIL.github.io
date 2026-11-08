import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4
    
    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off + 1 * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def solve_milp_for_rank(R, max_cols_to_try=5, timeout=30):
        """
        Try to find a rank-R decomposition by fixing V to a sparse canonical form
        and solving a linear system for U and W.
        
        Strategy: Fix V to have exactly one non-zero (1 or -1) per row, with each column
        of V having a specific sparse pattern. Then solve for U and W.
        """
        if time.time() > deadline - timeout:
            return None
        
        # Try different sparse patterns for V
        # Pattern 1: V is a permutation-like matrix with some entries
        # We'll try to fix V to have rank R columns, each with limited support
        
        best_result = None
        
        for attempt in range(max_cols_to_try):
            if time.time() > deadline - timeout:
                break
            
            np.random.seed(42 + attempt * 100)
            
            # Create a random sparse V matrix with R columns
            # Each row of V should have at most 1 non-zero
            V = np.zeros((V_dim, R))
            # Assign each row to a random column
            row_assignments = np.random.randint(0, R, size=V_dim)
            for r in range(V_dim):
                col = row_assignments[r]
                V[r, col] = 1.0  # or -1.0 with some probability
            
            # Now we need to find U (U_dim x R) and W (W_dim x R) such that
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            
            # For each (a,b,c) with T[a,b,c] = 1:
            #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # For each (a,b,c) with T[a,b,c] = 0:
            #   sum_r U[a,r] * V[b,r] * W[c,r] = 0
            
            # This is still non-linear because of the product U*V*W.
            # But if V is fixed, we can think of it as: for each r, the contribution
            # is U[:,r] * V[:,r] * W[:,r] (outer product scaled by V[:,r])
            
            # Let's define X_r = U[:,r] * W[:,r] (element-wise product), then
            # T[a,b,c] = sum_r V[b,r] * X_r[a] * ... wait, this doesn't work directly.
            
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # If V is fixed, for each r, let u_r = U[:,r], w_r = W[:,r]
            # Then T = sum_r V[:,r] * (u_r * w_r) as a tensor... no.
            # T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            
            # This is linear in U and W separately if we fix one.
            # Let's fix V, then for each b, we have:
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # If we define M_b = diag(V[:,b]) ... no, V is (V_dim, R), so V[b,r] is the b-th row.
            
            # Let's restructure: For fixed V, define for each b:
            # T[:,b,:] = sum_r U[:,r] * V[b,r] * W[:,r].T  (outer product)
            
            # This is a sum of outer products. If we stack all equations:
            # We have U_dim * V_dim * W_dim equations, with U_dim*R + W_dim*R unknowns.
            
            # For U_dim=10, W_dim=10, V_dim=25, R=45:
            # Unknowns: 10*45 + 10*45 = 900
            # Equations: 10*25*10 = 2500
            # This is over-determined, so we need to check if a solution exists.
            
            # Let's use least squares:
            # Reshape the problem: For each b, T[:,b,:] is a 10x10 matrix.
            # T[:,b,:] = sum_r V[b,r] * U[:,r] * W[:,r].T
            
            # Let's create a linear system:
            # Unknowns: U (10xR) and W (10xR), total 20R variables.
            # For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # This is still bilinear in U and W. To linearize, we can try:
            # Fix U first, solve for W, then iterate. Or use alternating least squares.
            
            # Let's use ALS:
            U = np.random.randn(U_dim, R) * 0.1
            W = np.random.randn(W_dim, R) * 0.1
            
            for iteration in range(100):
                if time.time() > deadline - timeout:
                    break
                
                # Solve for U given W
                # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # For fixed a: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let B[a,r] = V[:,r] * W[:,r] ... no, B is (V_dim, W_dim, R)
                
                # For each a, we have a linear system in U[a,:]:
                # For each (b,c): T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This is: T[a,:,:] = U[a,:] @ (V.T * W) ... element-wise?
                
                # Let's define M[a] as a (V_dim*W_dim) x R matrix where
                # M[a][b*W_dim+c, r] = V[b,r] * W[c,r]
                # Then T[a,:,:].flatten() = M[a] @ U[a,:]
                
                U_new = np.zeros((U_dim, R))
                for a in range(U_dim):
                    M_a = np.zeros((V_dim * W_dim, R))
                    for b in range(V_dim):
                        for c in range(W_dim):
                            M_a[b * W_dim + c, :] = V[b, :] * W[c, :]
                    # Solve M_a @ U[a,:] = T[a,:,:].flatten()
                    try:
                        U_new[a, :] = np.linalg.lstsq(M_a, T[a, :, :].flatten(), rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                
                # Solve for W given U
                W_new = np.zeros((W_dim, R))
                for c in range(W_dim):
                    M_c = np.zeros((U_dim * V_dim, R))
                    for a in range(U_dim):
                        for b in range(V_dim):
                            M_c[a * V_dim + b, :] = U[a, :] * V[b, :]
                    try:
                        W_new[c, :] = np.linalg.lstsq(M_c, T[:, :, c].flatten(), rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                
                # Check convergence
                diff_U = np.linalg.norm(U_new - U)
                diff_W = np.linalg.norm(W_new - W)
                U = U_new
                W = W_new
                
                if diff_U + diff_W < 1e-8:
                    break
            
            # Check if the solution is valid
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            loss = np.sum((prod - T) ** 2)
            
            if loss < 1e-6:
                # Try to snap to half-integers
                valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
                if valid:
                    best_result = (U_s, V_s, W_s)
                    break
                else:
                    # Check if it's close enough to integer/half-integer
                    U_round = snap_to_half(U)
                    V_round = snap_to_half(V)
                    W_round = snap_to_half(W)
                    prod2 = np.einsum('ar,br,cr->abc', U_round, V_round, W_round, optimize=True)
                    if np.allclose(prod2, T, atol=1e-5):
                        best_result = (U_round, V_round, W_round)
                        break
        
        return best_result

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    
    # Try to find lower rank using MILP approach
    for target_rank in [45, 44, 43]:
        if time.time() > deadline - 20:
            break
        
        result = solve_milp_for_rank(target_rank, max_cols_to_try=5, timeout=15)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
