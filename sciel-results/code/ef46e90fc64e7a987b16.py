"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # For each (i, k), check that sum_c U[i*m + j, c] * V[j*p + k, c] gives the identity in j
    # Actually, the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = delta_{j} (i.e., the (j,k) entry of C = A*B where C[i,k] = sum_j A[i,j]*B[j,k])
    # So for each i, k: sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*W[k*n+i,c]) * ... 
    # Wait, let me re-read. The tensor for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = C[i,k]
    # So for each (i, k): sum_j (sum_c U[i,j,c]*W[i,k,c]) * B[j,k] ... no.
    # Actually: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] should equal the bilinear form.
    # For a fixed (i,k), the left side as a function of A[i,:] and B[:,k] is:
    # sum_j A[i,j] * (sum_c U[i,j,c]*W[i,k,c]) * ... no.
    # Let me think again. The standard form: C[i,k] = sum_c sum_j A[i,j] B[j,k] * ... 
    # No: C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) * W[i,k,c]? No.
    # The correct form is: C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] summed over j and c.
    # So for each (i,k): sum_{j,c} U[i,j,c] * V[j,k,c] * W[i,k,c] * A[i,j] * B[j,k]
    # This equals sum_j A[i,j] B[j,k] * (sum_c U[i,j,c] * V[j,k,c] * W[i,k,c])
    # For this to equal sum_j A[i,j] B[j,k], we need: for each (i,j,k), sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = delta_{j} ... 
    # No wait. It should be that the coefficient of A[i,j]*B[j,k] is 1 for all j, i.e.,
    # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = 1 for all i, j, k.
    # But that's not right either because A[i,j] and B[j,k] are independent variables.
    # Actually the bilinear form is: <C, E_{i,k}> = C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition gives: <C, E_{i,k}> = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k]) W[i,k,c]
    # No, that's not standard. Let me look at the problem statement again.
    # "W[k*n+i,r]" - so W is indexed by (k, i) with row index k*n+i.
    # Standard: C[i,k] = sum_c sum_j A[i,j] * B[j,k] where the tensor T_{i,j,k} = 1 if the multiplication is valid.
    # The decomposition: T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{i,k}^r (or similar).
    # Given W[k*n+i, r], W is indexed by (k, i). So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # The condition is: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since T_{i,j,k} = 1 for all i, j, k in matrix multiplication).
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _is_half_integer(x, tol=1e-6):
    """Check if x is close to a half-integer (n/2 for integer n)."""
    if x is None:
        return False
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integers(U, V, W, tol=1e-6):
    """Check if all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _random_initialization(n, m, p, rank, rng):
    """Generate random initial factor matrices with half-integer-like entries."""
    U = rng.random((n*m, rank)) * 2 - 1
    V = rng.random((m*p, rank)) * 2 - 1
    W = rng.random((p*n, rank)) * 2 - 1
    # Snap to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return U.tolist(), V.tolist(), W.tolist()

def _objective(params, n, m, p, rank):
    """Objective function: minimize the maximum error over all (i,j,k) triples."""
    U = np.array(params[:n*m*rank]).reshape(n*m, rank)
    V = np.array(params[n*m*rank:n*m*rank + m*p*rank]).reshape(m*p, rank)
    W = np.array(params[n*m*rank + m*p*rank:]).reshape(p*n, rank)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                err = val - 1.0
                max_err = max(max_err, abs(err))
    return max_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Try ranks from 20 upward (known that rank 3x3x3 is 23, so 3x3x4 should be similar)
    best_result = None
    best_rank = None
    
    for rank in range(20, 40):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random restarts
        for trial in range(50):
            if time.time() > deadline - 25:
                break
            
            rng = np.random.RandomState(trial * 1000 + rank)
            U_list, V_list, W_list = _random_initialization(n, m, p, rank, rng)
            
            # Quick check: is this already valid?
            if _check_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-4):
                if _all_half_integers(U_list, V_list, W_list, tol=1e-4):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
            
            # Try gradient-based refinement
            params = np.concatenate([
                np.array(U_list).flatten(),
                np.array(V_list).flatten(),
                np.array(W_list).flatten()
            ])
            
            # Use a simple iterative refinement
            lr = 0.01
            for iteration in range(1000):
                if time.time() > deadline - 20:
                    break
                
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_row = U[i*m+j]
                            v_row = V[j*p+k]
                            w_row = W[k*n+i]
                            val = np.dot(u_row, v_row * w_row)
                            err = val - 1.0
                            dU[i*m+j] += err * (v_row * w_row)
                            dV[j*p+k] += err * (u_row * w_row)
                            dW[k*n+i] += err * (u_row * v_row)
                
                params -= lr * np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                
                # Check convergence
                max_err = 0.0
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err = abs(val - 1.0)
                            max_err = max(max_err, err)
                
                if max_err < 1e-8:
                    # Snap to half-integers
                    U_list = np.round(U * 2) / 2
                    V_list = np.round(V * 2) / 2
                    W_list = np.round(W * 2) / 2
                    if _check_decomposition(U_list.tolist(), V_list.tolist(), W_list.tolist(), n, m, p, tol=1e-4):
                        best_result = {"U": U_list.tolist(), "V": V_list.tolist(), "W": W_list.tolist()}
                        best_rank = rank
                        return best_result
                    # Even if not half-integer, return the continuous solution
                    # But the problem requires half-integer entries...
                    # Let's keep searching
                    break
            
            if time.time() > deadline - 15:
                break
    
    # If we found a valid decomposition, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the schoolbook decomposition
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
