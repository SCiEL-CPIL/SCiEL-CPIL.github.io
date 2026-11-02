"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # wall-clock limit with safety margin
    SAFETY_MARGIN = 5.0  # seconds to spare

    n, m, p = DIMENSIONS

    # Schoolbook decomposition as fallback
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

    def check_valid(U, V, W):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    import random
    import numpy as np

    def random_decomposition(rank, max_val=2):
        """Generate a random decomposition with integer entries."""
        U = np.array([[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(n * m)], dtype=np.float64)
        V = np.array([[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(m * p)], dtype=np.float64)
        W = np.array([[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(p * n)], dtype=np.float64)
        return U, V, W

    def compute_tensor(U, V, W):
        """Compute the full tensor T[i,j,k]."""
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            # U[:, c] is length n*m
            # V[:, c] is length m*p
            # W[:, c] is length p*n
            
            # Reshape to matrices
            # U_c[i, j] = U[i*m+j, c]
            # V_c[j, k] = V[j*p+k, c]
            # W_c[k, i] = W[k*n+i, c]
            
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            
            # T += Uc @ Vc @ Wc^T ? 
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # Let M = Uc @ Vc. M[i,k] = sum_j Uc[i,j] Vc[j,k].
            # Then T[i,j,k] needs Vc[j,k] and Wc[k,i].
            # This is not a simple matrix multiplication chain for the whole tensor at once.
            # We need the outer product structure.
            
            # T_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This can be computed as:
            # Temp[i,j] = Uc[i,j]
            # We want to add to T[i,j,k] the term Uc[i,j] * Vc[j,k] * Wc[k,i]
            
            # Efficient way:
            # For each i, j:
            #   T[i,j,:] += Uc[i,j] * (Vc[j,:] * Wc[:,i])
            
            # Vectorized:
            # Vc is (m, p). Wc is (p, n).
            # We need for each j, the vector Vc[j,:] * Wc[:,j]? No, Wc[k,i].
            
            # Let's iterate over j (inner dim of U, V).
            # For a fixed j:
            #   U_col = Uc[:, j] (length n)
            #   V_row = Vc[j, :] (length p)
            #   We need to add U_col[i] * V_row[k] * Wc[k, i] to T[i, j, k]
            #   This is outer(U_col, V_row) * Wc.T?
            #   outer(U_col, V_row) is (n, p).
            #   Wc.T is (n, p). Wc.T[i, k] = Wc[k, i].
            #   So T[:, j, :] += outer(U_col, V_row) * Wc.T
            
            for j in range(m):
                U_col = Uc[:, j]
                V_row = Vc[j, :]
                T[:, j, :] += np.outer(U_col, V_row) * Wc.T

        return T

    def local_search_fast(U, V, W, max_iters=10000):
        """Fast local search using incremental tensor updates."""
        rank = U.shape[1]
        T = compute_tensor(U, V, W)
        error = np.sum((T - 1.0) ** 2)
        
        # Pre-compute indices for slices
        # U index u -> i, j
        # V index v -> j, k
        # W index w -> k, i
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Pick a random matrix and entry
            mat_id = random.randint(0, 2) # 0:U, 1:V, 2:W
            delta = random.choice([0.5, -0.5, 1.0, -1.0])
            
            if mat_id == 0: # U
                u = random.randint(0, n * m - 1)
                c = random.randint(0, rank - 1)
                i = u // m
                j = u % m
                
                # Affected T slice: T[i, j, :]
                # Change: delta * (V[j*p + k, c] * W[k*n + i, c]) for k in 0..p-1
                v_vec = V[j*p : (j+1)*p, c] # length p
                w_vec = W[i :: n, c]       # length p (indices i, i+n, ..., i+(p-1)n)
                
                change = delta * (v_vec * w_vec)
                
                old_T_slice = T[i, j, :].copy()
                new_T_slice = old_T_slice + change
                
                old_err_slice = np.sum((old_T_slice - 1.0) ** 2)
                new_err_slice = np.sum((new_T_slice - 1.0) ** 2)
                
                if new_err_slice < error - old_err_slice + 1e-12:
                    T[i, j, :] = new_T_slice
                    U[u, c] += delta
                    error += new_err_slice - old_err_slice
                    if error < 1e-12:
                        return U, V, W, True
            elif mat_id == 1: # V
                v = random.randint(0, m * p - 1)
                c = random.randint(0, rank - 1)
                j = v // p
                k = v % p
                
                # Affected T slice: T[:, j, k]
                # Change: delta * (U[i*m + j, c] * W[k*n + i, c]) for i in 0..n-1
                u_vec = U[j :: m, c] # length n (indices j, j+m, ..., j+(n-1)m)
                w_vec = W[k*n : k*n+n, c] # length n
                
                change = delta * (u_vec * w_vec)
                
                old_T_slice = T[:, j, k].copy()
                new_T_slice = old_T_slice + change
                
                old_err_slice = np.sum((old_T_slice - 1.0) ** 2)
                new_err_slice = np.sum((new_T_slice - 1.0) ** 2)
                
                if new_err_slice < error - old_err_slice + 1e-12:
                    T[:, j, k] = new_T_slice
                    V[v, c] += delta
                    error += new_err_slice - old_err_slice
                    if error < 1e-12:
                        return U, V, W, True
            else: # W
                w = random.randint(0, p * n - 1)
                c = random.randint(0, rank - 1)
                k = w // n
                i = w % n
                
                # Affected T slice: T[i, :, k]
                # Change: delta * (U[i*m + j, c] * V[j*p + k, c]) for j in 0..m-1
                u_vec = U[i*m : (i+1)*m, c] # length m
                v_vec = V[k :: p, c]       # length m (indices k, k+p, ..., k+(m-1)p)
                
                change = delta * (u_vec * v_vec)
                
                old_T_slice = T[i, :, k].copy()
                new_T_slice = old_T_slice + change
                
                old_err_slice = np.sum((old_T_slice - 1.0) ** 2)
                new_err_slice = np.sum((new_T_slice - 1.0) ** 2)
                
                if new_err_slice < error - old_err_slice + 1e-12:
                    T[i, :, k] = new_T_slice
                    W[w, c] += delta
                    error += new_err_slice - old_err_slice
                    if error < 1e-12:
                        return U, V, W, True
        
        return U, V, W, False

    # Try ranks from low to high
    best_result = None
    best_rank = n * m * p  # schoolbook

    # Try to find a better decomposition
    # Start with a reasonably low rank. 20 is a good guess for 3x3x3, so 30-40 for 3x3x8?
    # Let's try 20 to 72.
    for rank in range(20, best_rank + 1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break

        # Try a few random initializations
        for trial in range(5):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break

            U, V, W = random_decomposition(rank, max_val=2)
            U, V, W, found = local_search_fast(U, V, W, max_iters=2000)

            if found:
                # Convert to list of lists of floats, then round to 0.5
                U_list = [[float(round(val * 2) / 2) for val in row] for row in U]
                V_list = [[float(round(val * 2) / 2) for val in row] for row in V]
                W_list = [[float(round(val * 2) / 2) for val in row] for row in W]
                
                # Verify
                if check_valid(U_list, V_list, W_list):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        else:
            continue
        break

    if best_result is not None:
        return best_result

    # Return schoolbook as fallback
    return schoolbook()
# EVOLVE-BLOCK-END
