"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _to_half_int_or_int(mat):
    """Convert a numpy array to a list of lists with integer or half-integer entries."""
    # Multiply by 2 to check if all entries are multiples of 0.5
    doubled = mat * 2
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=1e-6):
        # Entries are multiples of 0.5
        result = []
        for row in rounded:
            r = []
            for val in row:
                v = float(val)
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            result.append(r)
        return result
    return None


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For exact matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    # Actually, the matrix multiplication tensor has entries:
    # T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense that C[i,k] = sum_j A[i,j]B[j,k])
    
    # Let's verify by checking that for random matrices, the product is correct
    # C = A @ B should equal sum_c U[:,c] * V[:,c] * W[:,c] interpreted properly
    
    # Actually, the tensor for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # is a 3-tensor T where T[i,j,k] = 1 for all i,j,k
    # The rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the indices need to match. Let me re-read the task.
    
    # U has shape (n*m, rank), indexed as U[i*m+j, c]
    # V has shape (m*p, rank), indexed as V[j*p+k, c]  
    # W has shape (p*n, rank), indexed as W[k*n+i, c]
    
    # The tensor entry T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    
    # Let's verify with a few random matrices
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from the decomposition
        C_decomposed = np.zeros((n, p))
        rank = U.shape[1]
        for c in range(rank):
            # U[:, c] is a vector of length n*m, reshape to (n, m)
            U_c = U[:, c].reshape(n, m)
            # V[:, c] is a vector of length m*p, reshape to (m, p)
            V_c = V[:, c].reshape(m, p)
            # W[:, c] is a vector of length p*n, reshape to (p, n)
            W_c = W[:, c].reshape(p, n)
            
            # The contribution to C[i,k] from term c:
            # sum_j U_c[i,j] * V_c[j,k] * W_c[k,i]
            # Wait, that's not quite right. Let me think again.
            
            # Actually, the standard tensor decomposition for matrix multiplication:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_c sum_j U[i,j,c] A[i,j] * V[j,k,c] B[j,k] * W[k,i,c]
            # This doesn't directly give us C. Let me reconsider.
            
            # The correct interpretation: if we have the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # and T[i,j,k] = 1 for all i,j,k, then:
            # C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k]
            # So the decomposition is valid if T[i,j,k] = 1 for all i,j,k.
            
            pass
        
        # Better: directly compute T[i,j,k] from the decomposition
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        
        if not np.allclose(T, 1.0, atol=tol):
            return False
    return True


def _try_rank(r, n, m, p, max_iters=200, seed=42):
    """Try to find a rank-r decomposition using gradient-based optimization."""
    rng = np.random.RandomState(seed)
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize randomly
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    # Scale to reasonable magnitude
    scale = (n * m * p) ** 0.25
    U *= scale
    V *= scale
    W *= scale
    
    lr = 0.01
    for it in range(max_iters):
        # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # and gradient w.r.t. U, V, W of ||T - 1||^2
        
        # For efficiency, compute T and gradients
        T = np.zeros((n, m, p))
        grad_U = np.zeros((n * m, r))
        grad_V = np.zeros((m * p, r))
        grad_W = np.zeros((p * n, r))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err = T[i, j, k] - 1.0
                    
                    # grad_U[u_idx,c] += err * V[v_idx,c] * W[w_idx,c]
                    grad_U[u_idx, :] += err * V[v_idx, :] * W[w_idx, :]
                    grad_V[v_idx, :] += err * U[u_idx, :] * W[w_idx, :]
                    grad_W[w_idx, :] += err * U[u_idx, :] * V[v_idx, :]
        
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_loss < 1e-8:
            break
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        if it % 50 == 49:
            lr *= 0.5
    
    if best_loss < 1e-6:
        return best_U, best_V, best_W, best_loss
    return None, None, None, best_loss


def _strassen_like_2x4x6():
    """
    Try to construct a decomposition using block structure.
    For 2x4x6, we can think of it as multiplying blocks.
    A is 2x4, B is 4x6, C is 2x6.
    
    Split A into [A1 A2] where A1 is 2x2, A2 is 2x2
    Split B into [B1; B2] where B1 is 2x6, B2 is 2x6
    C = A1*B1 + A2*B2
    
    Each 2x2*2x6 can be done with rank 2*2*6=24 schoolbook, but maybe less.
    Actually, 2x2 times 2x6: this is like two independent 2x2*2x1 problems... 
    No, it's a 2x2 matrix times a 2x6 matrix.
    
    Let me think differently. The tensor for 2x4x6 matrix multiplication.
    Known results: M(2,4,6) - what's the minimal rank?
    
    Schoolbook gives 2*4*6 = 48.
    
    Can we do better? Using Strassen-like ideas on blocks:
    View A as 2x(2+2), B as (2+2)x6.
    C = A_left @ B_top + A_right @ B_bottom
    where A_left is 2x2, B_top is 2x6, etc.
    
    Each 2x2 @ 2x6 product: this is a tensor of size 2*2*6 = 24.
    But we can use Strassen on the 2x2 part:
    2x2 @ 2x6: Let B_top = [b1 b2] where b1, b2 are 2x3 columns.
    Then A_left @ B_top = A_left @ [b1 b2] = [A_left@b1, A_left@b2]
    Each A_left@b_i is 2x2 @ 2x3, which has tensor size 2*2*3=12.
    
    This is getting complicated. Let me just try optimization.
    """
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Try to find a lower rank decomposition
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Also rank >= n*m*p / max(n,m,p) = 48/6 = 8 (weak)
    # Realistically, we might be able to get below 48.
    
    # Try ranks from a reasonable lower bound up
    # Let's try some candidate ranks
    candidate_ranks = [24, 28, 32, 36, 40, 44]
    
    for r in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try a few random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W, loss = _try_rank(r, n, m, p, max_iters=100, seed=seed)
            
            if U is not None and loss < 1e-6:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    # Try to round to half-integers
                    U_int = _to_half_int_or_int(U)
                    V_int = _to_half_int_or_int(V)
                    W_int = _to_half_int_or_int(W)
                    
                    if U_int is not None and V_int is not None and W_int is not None:
                        # Convert back to verify
                        U_np = np.array(U_int, dtype=float)
                        V_np = np.array(V_int, dtype=float)
                        W_np = np.array(W_int, dtype=float)
                        if _verify(U_np, V_np, W_np, n, m, p, tol=1e-4):
                            best_result = {"U": U_int, "V": V_int, "W": W_int}
                            best_rank = r
                            found = True
                            break
                    else:
                        # Use the floating point version
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = r
                        found = True
                        break
        
        if found:
            # Found a decomposition with rank r, try lower ranks
            continue
    
    return best_result
# EVOLVE-BLOCK-END
