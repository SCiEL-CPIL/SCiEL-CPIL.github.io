import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _schoolbook(n, m, p):
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

def _check_and_round(U, V, W, tol=1e-5):
    """Check if U, V, W are close to half-integers and return rounded versions if so."""
    # Flatten
    Uf = U.flatten()
    Vf = V.flatten()
    Wf = W.flatten()
    
    # Check distance to half-integers
    # Half integers are k/2. Distance to nearest half integer:
    # abs(x - round(2x)/2)
    
    dist_U = np.abs(Uf - np.round(2 * Uf) / 2.0)
    dist_V = np.abs(Vf - np.round(2 * Vf) / 2.0)
    dist_W = np.abs(Wf - np.round(2 * Wf) / 2.0)
    
    if np.max(dist_U) < tol and np.max(dist_V) < tol and np.max(dist_W) < tol:
        # Round to half-integers
        U_r = np.round(2 * Uf) / 2.0
        V_r = np.round(2 * Vf) / 2.0
        W_r = np.round(2 * Wf) / 2.0
        
        # Verify exact decomposition with rounded values
        # M_ijk = sum U_ir V_jr W_kr
        # n,m,p
        n, m, p = DIMENSIONS
        U_3d = U_r.reshape(n*m, -1)
        V_3d = V_r.reshape(m*p, -1)
        W_3d = W_r.reshape(p*n, -1)
        
        rank = U_3d.shape[1]
        
        # Construct tensor
        # M[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to verify this matches the identity tensor for matrix mult?
        # No, the tensor is the "multiplication tensor" T_{i,j,k} = delta_{i,j,k}??
        # No, the matrix multiplication tensor is defined such that
        # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor decomposition represents the bilinear map.
        # The "target" tensor T is such that T_{(i,j), (j,k), (i,k)} = 1?
        # Wait, the standard definition of the matrix multiplication tensor M_{n,m,p}
        # has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
        # The value is 1 if the indices correspond to a valid multiplication term?
        # No, the tensor is trilinear.
        # M_{ijk} = 1 if i,j,k form a "chain"?
        # Actually, the tensor entries are 1 for all (i,j,k) if we consider the map
        # (A, B) -> C.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # This is a bilinear map. The tensor decomposition U, V, W satisfies:
        # C_{i,k} = sum_r U_{i,r} (sum_j A_{i,j} V_{j,r}) W_{k,r} ... wait.
        
        # Standard decomposition:
        # C_{i,k} = sum_{r} U_{i,r} V_{k,r} (sum_j A_{i,j} B_{j,k} ... no)
        
        # Correct form:
        # Let A be n x m, B be m x p, C be n x p.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # Decomposition:
        # C_{i,k} = sum_r U_{i,r} V_{k,r} (sum_j U'_{j,r} A_{i,j} V'_{j,r} B_{j,k}?)
        
        # The prompt specifies:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # And the convention W[k*n+i, r].
        
        # The equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = Delta_{i,j,k}
        # Where Delta_{i,j,k} is 1 if... ?
        # For matrix multiplication, the tensor T has T_{i,j,k} = 1 for all i,j,k?
        # No.
        # The tensor T_{n,m,p} is the structure tensor of the algebra.
        # T_{(i,j), (j,k), (k,i)}? No.
        
        # Let's look at the seed code:
        # U[i*m+j][column] = 1
        # V[j*p+k][column] = 1
        # W[k*n+i][column] = 1
        # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
        # So the "Target Tensor" T has T_{i,j,k} = 1 for ALL i,j,k?
        # No, that would be the rank 1 tensor of all ones.
        
        # Let's re-read the tensor definition.
        # Matrix multiplication is a trilinear map (A, B, C) -> sum A_i B_j C_k? No.
        # The multiplication tensor T is defined such that:
        # T_{i,j,k} = 1 if the index triple (i,j,k) is "valid" in the product?
        # In the schoolbook case, each (i,j,k) triplet gets a unique rank term.
        # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Verification:
        # sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r} = 1 for all i,j,k.
        
        # So the target is a tensor of all 1s.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_3d[i*m+j, r] * V_3d[j*p+k, r] * W_3d[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        error = 1.0
                        break
                if error: break
            if error: break
        
        if error == 0.0:
            return True, U_r.reshape(U.shape), V_r.reshape(V.shape), W_r.reshape(W.shape)
        
    return False, None, None, None

def _try_find_decomposition(n, m, p, rank):
    # Variables
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    # Residual function
    def residual(x):
        Uf = x[:size_U].reshape(n*m, rank)
        Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
        Wf = x[size_U+size_V:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    u_row = Uf[i*m+j, :]
                    v_row = Vf[j*p+k, :]
                    w_row = Wf[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Initial guess
    # Random initialization
    np.random.seed(42) # For reproducibility in search, but we want diversity
    # Better: Use small random noise
    x0 = np.random.normal(0, 0.1, total_size)
    
    try:
        res = least_squares(residual, x0, method='lm', xtol=1e-10, ftol=1e-10, max_nfev=1000)
        
        if res.success or res.cost < 1e-8:
            x_sol = res.x
            U_sol = x_sol[:size_U].reshape(n*m, rank)
            V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
            W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
            
            valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
            if valid:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    except Exception as e:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a better decomposition
    for rank in range(22, 27):
        if time.time() > deadline:
            break
        
        result = _try_find_decomposition(n, m, p, rank)
        if result:
            return result
            
    # Fallback
    return _schoolbook(n, m, p)
