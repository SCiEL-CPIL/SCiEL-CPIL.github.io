"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_TIMEOUT = 145.0  # Leave 5s safety margin

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Reconstruct C from decomposition
        # C[i,k] = sum_t U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t] ... 
        # Wait, the decomposition is: T_{(ij),(jk),(ik)} = sum_t U_{ij,t} V_{jk,t} W_{ik,t}
        # So C_{ik} = sum_j (sum_t U_{ij,t} V_{jk,t} W_{ik,t})? No.
        # Actually: C_{ik} = sum_j A_{ij} B_{jk}
        # And the tensor says: the "coefficient" for (A_{ij}, B_{jk}) contributing to C_{ik} is T_{(ij),(jk),(ik)}
        # In the decomposition: T_{(ij),(jk),(ik)} = sum_t U_{ij,t} * V_{jk,t} * W_{ik,t}
        # So C_{ik} = sum_j A_{ij} B_{jk} * 1 = sum_j sum_t A_{ij} B_{jk} U_{ij,t} V_{jk,t} W_{ik,t}
        # Hmm, that's not quite right either. Let me think again.
        
        # The standard formulation: We want C_{ik} = sum_j A_{ij} B_{jk}
        # A rank-r decomposition writes: C_{ik} = sum_t (sum_j A_{ij} U_{ij,t}) * (sum_j B_{jk} V_{jk,t}) * W_{ik,t}
        # No, that's not standard either.
        
        # Let me reconsider. The multiplication tensor T has T_{(ij),(jk),(ik)} = 1.
        # A rank-r decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
        # where u_t is a vector indexed by (i,j), v_t by (j,k), w_t by (i,k)
        # T_{(ij),(jk),(ik)} = sum_t u_t[i,j] * v_t[j,k] * w_t[i,k]
        # For this to equal 1 for the "correct" (i,j,k) and 0 otherwise... 
        # Actually no. The tensor T_{(ij),(jk),(ik)} = 1 always (for all i,j,k), because 
        # the multiplication tensor is defined such that the bilinear form is:
        # <T, A ⊗ B> = sum_{i,j,k} T_{(ij),(jk),(ik)} A_{ij} B_{jk} = sum_{i,j,k} A_{ij} B_{jk} = sum_i,k (A@B)_{ik}
        # Wait, that's not right. Let me look at this more carefully.
        
        # The matrix multiplication tensor is defined as:
        # T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ik}
        # So T_{(ij),(jk),(ik)} = 1 for all valid i,j,k.
        
        # A rank-r decomposition: T = sum_{t=1}^{r} U_t ⊗ V_t ⊗ W_t
        # where U_t ∈ R^{n*m}, V_t ∈ R^{m*p}, W_t ∈ R^{p*n}
        # T_{(ij),(jk),(ik)} = sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1
        
        # So we need: for ALL i,j,k: sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1
        
        # Let's verify this condition:
        U_mat = np.array(U, dtype=float)  # shape (n*m, rank)
        V_mat = np.array(V, dtype=float)  # shape (m*p, rank)
        W_mat = np.array(W, dtype=float)  # shape (p*n, rank)
        
        # Check: for all i, j, k: sum_t U_mat[i*m+j, t] * V_mat[j*p+k, t] * W_mat[k*n+i, t] == 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*m+j, t] * V_mat[j*p+k, t] * W_mat[k*n+i, t] for t in range(rank))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            return False, max_err
    
    return True, 0.0

def construct():
    n, m, p = DIMENSIONS
    
    rank_schoolbook = n * m * p  # 90
    
    # Build schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # This should never happen, but just in case
        pass
    
    # Check time
    if time.time() - _start_time > _TIMEOUT:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # Strategy: Use the structure that if we can find a set of rank-1 tensors that sum to T,
    # we might be able to do better than schoolbook.
    
    # For now, return schoolbook as it's guaranteed valid.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
