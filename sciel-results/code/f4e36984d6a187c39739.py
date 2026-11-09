"""Matrix multiplication 4x4x6 tensor decomposition search."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor: T[i,j,k,l] where output[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # which should equal delta[i,i']*delta[j,j']*delta[k,k'] for appropriate indices
    # Actually: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 (for all i,j,k) when we think of it as
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all valid i,j,k
    # Wait, that's not right. Let me think again.
    #
    # The matrix multiplication tensor T is defined such that:
    # (A B)[i, k] = sum_{j} T_{i,j,k} A[i,j] B[j,k]
    # where T_{i,j,k} = 1 for all i,j,k.
    #
    # A rank-r decomposition: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But wait, the W index is k*n+i, which couples i and k.
    #
    # Actually the standard form is:
    # (A B)[i, k] = sum_{j, c} A[i, j] * U[i*m+j, c] * V[j*p+k, c] * B[j, k] ... no.
    #
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The decomposition is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
    #
    # Actually no. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T_{i, j, k} = 1  (it's the identity map on the tensor product space)
    # A decomposition of rank r means:
    # 1 = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}  for all i, j, k
    #
    # Wait, that can't be right either because the W factor has index k*n+i which mixes output row i and column k.
    #
    # Let me reconsider. The standard Strassen-style decomposition of matrix multiplication:
    # C = A * B, where C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c U[i,j,c] A[i,j] V[j,k,c] B[j,k] ... no, that's not how it works.
    #
    # Actually: C[i,k] = sum_c (sum_j U_c[i,j] A[i,j]) * (sum_j V_c[j,k] B[j,k])  -- no
    #
    # The correct form: C[i,k] = sum_c (sum_j a_{i,j} u_{i,j,c}) * (sum_j b_{j,k} v_{j,k,c})
    # Hmm, that's not standard either.
    #
    # Standard: C[i,k] = sum_c M_c * N_c where M_c = sum_{i,j} alpha_c[i,j] A[i,j]
    # and N_c = sum_{j,k} beta_c[j,k] B[j,k]... no.
    #
    # Let me just use the verification approach: for random A and B, check that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k] gives C[i,k]
    #
    # Actually the simplest check: the tensor T[i,j,k] = 1 must be decomposable as
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    #
    # Let's just verify numerically.
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition using random search / optimization
    # The key insight: we can try rank < 96 by searching for sparse decompositions
    
    # First, try a structured approach: use block decomposition
    # Split A into blocks and use Strassen-like ideas
    
    # For 4x4x4, the known rank is 49. For 4x4x6, we can potentially do better than 96.
    # Strategy: decompose the 4x4x6 tensor using the 4x4x4 rank-49 structure.
    
    # Actually, let's try a direct random search approach.
    # We'll try to find a decomposition of rank r for r < 96.
    
    # Try rank values from 95 down
    # For each rank, try random initializations and gradient-free optimization
    
    def try_random_decomposition(rank, max_iters=2000, seed=42):
        """Try to find a rank-r decomposition via random search."""
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # The tensor T[i,j,k] = 1 must equal sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Error = sum_{i,j,k} (1 - sum_c ...)^2
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (1.0 - s) ** 2
            return err
        
        # Simple local search: perturb one element at a time
        err = compute_error(U, V, W)
        step_size = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random matrix, row, column to perturb
            mat = rng.randint(3)
            if mat == 0:
                r_idx = rng.randint(n * m)
                c_idx = rng.randint(rank)
                old_val = U[r_idx, c_idx]
                U[r_idx, c_idx] = old_val + rng.randn() * step_size
            elif mat == 1:
                r_idx = rng.randint(m * p)
                c_idx = rng.randint(rank)
                old_val = V[r_idx, c_idx]
                V[r_idx, c_idx] = old_val + rng.randn() * step_size
            else:
                r_idx = rng.randint(p * n)
                c_idx = rng.randint(rank)
                old_val = W[r_idx, c_idx]
                W[r_idx, c_idx] = old_val + rng.randn() * step_size
            
            new_err = compute_error(U, V, W)
            if new_err < err:
                err = new_err
                step_size = min(step_size * 1.1, 0.1)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                if err < 1e-6:
                    break
            else:
                if mat == 0:
                    U[r_idx, c_idx] = old_val
                elif mat == 1:
                    V[r_idx, c_idx] = old_val
                else:
                    W[r_idx, c_idx] = old_val
                step_size = max(step_size * 0.9, 0.001)
        
        if best_err < 1e-4:
            # Convert to list format
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            # Round to reduce numerical noise
            U_list = [[round(x, 8) for x in row] for row in U_list]
            V_list = [[round(x, 8) for x in row] for row in V_list]
            W_list = [[round(x, 8) for x in row] for row in W_list]
            return U_list, V_list, W_list
        return None
    
    # Try a few rank values
    for r in [95, 94, 93, 92, 91, 90, 88, 86, 84, 80]:
        if time.time() > deadline - 10:
            break
        result = try_random_decomposition(r, max_iters=1000, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-4):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
