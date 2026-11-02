"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly represents matrix multiplication."""
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the standard basis? No.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 if we're computing (A*B)[i,k]
    # The bilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = (A*B)[i,k]
    # Wait, let me reconsider. The tensor for multiplying (n x m) by (m x p) to get (n x p)
    # has entries T[i, j, k] where i in [0,n), j in [0,m), k in [0,p).
    # The product (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i, j, k] = 1 for the coefficient of A[i,j]*B[j,k] in (A*B)[i,k].
    # But each output entry (A*B)[i,k] gets contributions from all j.
    # The standard way: the multiplication tensor T is a 3-tensor of size n x m x p
    # where T[i,j,k] = delta_{i,k}? No.
    
    # Let me think again. We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear algorithm: C[i,k] = sum_c U[i, c] * V[k, c] * ... no.
    
    # Actually the standard formulation: 
    # C = A * B
    # Each C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as: for each output (i,k), C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k' V[j*p+k, c] B[j,k'])
    # Wait, that's not right either. Let me use the standard tensor decomposition.
    
    # The matrix multiplication tensor T_{n,m,p} has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this term appears in some product. Actually for standard matrix mult:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the "tensor" here is really a collection of n*p bilinear forms, each depending on m inputs from A and m from B.
    
    # In the rank-r decomposition:
    # (AB)[i,k] = sum_c U[i, c] * V[k, c] ... no, that's for a different structure.
    
    # Let me look at the seed code more carefully:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for the multiplication tensor, 
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should be 1 for the "right" combinations.
    
    # Actually, I think the convention is:
    # We're decomposing the identity map on the tensor product, or rather:
    # The multiplication tensor is the sum over all (i,j,k) of e_i ⊗ e_j ⊗ e_k where...
    
    # Let me just verify with the schoolbook: in schoolbook, U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if there exists c such that the column c has
    # U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1, which happens exactly when c corresponds to (i,j,k).
    # So the "target tensor" is T[i,j,k] = 1 for all (i,j,k)? That would mean rank = n*m*p.
    
    # Wait, that can't be right for a meaningful problem. Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 8 matrix."
    # The output factor follows W[k*n+i, r].
    
    # I think the tensor being decomposed is the matrix multiplication tensor, which for computing
    # C = AB has the property that C[i,k] = sum_j A[i,j]B[j,k]. The tensor rank of this operation
    # is what we're minimizing. The standard tensor is T with T[i,j,k] = 1 if the (i,k) entry of the
    # product depends on A[i,j] and B[j,k]... 
    
    # Actually, I think the correct interpretation is:
    # We have a 3-way tensor T of shape (n*m, m*p, p*n) or similar, and we want its tensor rank.
    # But more likely, the problem is about the bilinear complexity of matrix multiplication.
    
    # For now, let me just verify numerically: generate random A (n x m) and B (m x p),
    # compute C = A @ B, and check if the decomposition reproduces it.
    
    # The decomposition should satisfy:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Hmm, that doesn't quite work dimensionally.
    
    # Let me try: C[i,k] = sum_c U[i, c] * V[k, c] where U and V are derived from A and B.
    # But U has shape (n*m, rank)...
    
    # I think the correct formula is:
    # Let X_c = sum_j U[i*m+j, c] * A[i,j] for each i
    # Let Y_c = sum_j V[j*p+k, c] * B[j,k] for each k  
    # Then C[i,k] = sum_c X_c[i] * Y_c[k] * W[k*n+i, c]
    
    # Let me verify with schoolbook:
    # For a given (i,j,k) with column c = (i,j,k):
    # U[i*m+j, c] = 1, all other U[i*m+j', c] = 0
    # So X_c[i] = A[i,j]
    # V[j*p+k, c] = 1, all other V[j'*p+k', c] = 0
    # So Y_c[k] = B[j,k]
    # W[k*n+i, c] = 1
    # So C[i,k] gets contribution A[i,j]*B[j,k]*1 from this term
    # Summing over all c (i.e., all (i',j',k') tuples), for fixed (i,k):
    # Only terms where i'=i and k'=k contribute (because W[k'*n+i', c] = 1 only when k'=k and i'=i)
    # And for those, j' can be anything, giving sum_j A[i,j]*B[j,k] = C[i,k]. Correct!
    
    n_val, m_val, p_val = n, m, p
    np.random.seed(42)
    A = np.random.randn(n_val, m_val)
    B = np.random.randn(m_val, p_val)
    C_expected = A @ B
    
    U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
    
    C_computed = np.zeros((n_val, p_val))
    rank = U_arr.shape[1]
    for c in range(rank):
        # X_c[i] = sum_j U[i*m+j, c] * A[i,j]
        X = np.zeros(n_val)
        for i in range(n_val):
            for j in range(m_val):
                X[i] += U_arr[i * m_val + j, c] * A[i, j]
        # Y_c[k] = sum_j V[j*p+k, c] * B[j,k]
        Y = np.zeros(p_val)
        for k in range(p_val):
            for j in range(m_val):
                Y[k] += V_arr[j * p_val + k, c] * B[j, k]
        # C[i,k] += X[i] * Y[k] * W[k*n+i, c]
        for i in range(n_val):
            for k in range(p_val):
                C_computed[i, k] += X[i] * Y[k] * W_arr[k * n_val + i, c]
    
    return np.allclose(C_computed, C_expected, atol=tol)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]
    
    # Try to find a lower rank decomposition using iterative methods
    # Strategy: use the known lower bound and try random initialization + optimization
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also rank >= ceil(n*p*m / max(n,m,p)) ... various bounds
    # For 2x3x8, known results: the rank is between 24 and 48 (schoolbook)
    # Let's try to find something better than 48.
    
    # Try various ranks from lower bound up to schoolbook
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists for the expected output format
    U_list = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_list = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_list = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
