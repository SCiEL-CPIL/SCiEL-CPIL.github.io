"""Matrix multiplication 3x3 by 3x8 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Find a low-rank decomposition of the 3x3 * 3x8 matrix multiplication tensor.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    def make_schoolbook():
        rank = schoolbook_rank
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        u_val = U[i * m + j][r]
                        v_val = V[j * p + k][r]
                        w_val = W[k * n + i][r]
                        actual += u_val * v_val * w_val
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def to_half_integer_matrix(mat):
        """Convert a matrix of floats to half-integers."""
        result = []
        for row in mat:
            r = []
            for val in row:
                v2 = round(val * 2)
                if v2 % 2 == 0:
                    r.append(v2 // 2)
                else:
                    r.append(v2 / 2.0)
            result.append(r)
        return result
    
    def try_greedy_rank(rank):
        """
        Try to find a rank-r decomposition using a greedy approach.
        We iteratively pick a rank-1 tensor that best reduces the residual.
        """
        # Target tensor T[i,j,k] = 1 for all i,j,k
        T = np.ones((n, m, p))
        residual = T.copy()
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            if time.time() > deadline - 10:
                return None
            
            # Find the best rank-1 tensor to subtract
            # A rank-1 tensor is u_i * v_j * w_k
            # We want to maximize the inner product with the residual
            # subject to ||u||=||v||=||w||=1 (or some normalization)
            
            # Simple heuristic: try to find u, v, w that maximize sum(u_i v_j w_k * residual[i,j,k])
            # This is equivalent to finding the top singular value of the tensor reshaped appropriately
            
            # Reshape residual to (n*m, p) for each j? No, that's not right.
            # The tensor is T[i,j,k]. We want u[i], v[j], w[k].
            # Let's try a random restart approach for the rank-1 step
            
            best_score = -1
            best_u = None
            best_v = None
            best_w = None
            
            # Try a few random initializations
            for _ in range(50):
                u = np.random.randn(n)
                v = np.random.randn(m)
                w = np.random.randn(p)
                
                # Normalize
                u = u / np.linalg.norm(u)
                v = v / np.linalg.norm(v)
                w = w / np.linalg.norm(w)
                
                # Compute the rank-1 tensor
                rank1 = np.einsum('i,j,k->ijk', u, v, w)
                
                # Compute inner product with residual
                score = np.sum(rank1 * residual)
                
                if score > best_score:
                    best_score = score
                    best_u = u
                    best_v = v
                    best_w = w
            
            if best_u is None:
                return None
            
            # Update residual
            rank1 = np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            residual -= rank1
            
            # Store in U, V, W
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = best_u[i]
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = best_v[j]
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r] = best_w[k]
        
        # Check if the decomposition is exact
        # Reconstruct T
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        if np.max(np.abs(T_recon - T)) < 1e-9:
            # Rationalize
            U_list = to_half_integer_matrix(U)
            V_list = to_half_integer_matrix(V)
            W_list = to_half_integer_matrix(W)
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try to find a decomposition with rank < schoolbook
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, n*p) = 24
    for rank in range(24, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_greedy_rank(rank)
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    return best_result
