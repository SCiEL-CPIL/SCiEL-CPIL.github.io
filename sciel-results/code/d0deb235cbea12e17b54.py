import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # The tensor is a direct sum of two 2x3x3 tensors.
    # So we search for a decomposition of 2x3x3 and duplicate it.
    
    def search_rank(n, m, p, target_rank, max_time=10):
        """
        Search for a decomposition of the n x m x p matrix multiplication tensor
        with the given target rank.
        Returns (U, V, W) if found, else None.
        """
        start_time = time.time()
        # Target tensor T[i, j, k] = 1 if i, j, k are valid indices
        # T is of shape (n*m, m*p, p*n) in the factorization convention?
        # No, the tensor is defined by T[i, j, k] = 1.
        # The factorization is T = sum_c U[:, c] V[:, c]^T W[:, c]^T?
        # No, the standard form is T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c].
        # Here i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # U has shape (n, rank), V has shape (m, rank), W has shape (p, rank).
        # But the interface requires U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
        # This corresponds to the flattening of the indices.
        # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c].
        # So the tensor element T_flat[i*m+j, j*p+k, k*n+i] is not directly T[i,j,k].
        # The product is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # This should equal 1 if the indices correspond to a valid term in the sum, else 0.
        # Actually, the matrix multiplication tensor has T[i, j, k] = 1 for all i, j, k.
        # Wait, no. The tensor is T[i, j, k] = 1.
        # The product C[i, k] = sum_j A[i, j] B[j, k].
        # So the tensor is T[i, j, k] = 1.
        # So we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
        
        # We can use the fact that T is a direct sum.
        # For 2x3x3, n=2, m=3, p=3.
        # We search for U, V, W of shapes (6, r), (9, r), (6, r).
        
        best_error = float('inf')
        best_factors = None
        
        for _ in range(100): # 100 restarts
            if time.time() - start_time > max_time:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating Least Squares
            for step in range(50):
                if time.time() - start_time > max_time:
                    break
                
                # Update U
                # For each c, U[:, c] = argmin ||T - sum U[:, c] V[:, c]^T W[:, c]^T||
                # This is a linear system.
                # We can update U by solving for each column c.
                # But it's easier to use gradient descent or a closed-form for one factor.
                # Let's use a simple gradient step.
                
                # Compute current product
                # P[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # This is expensive to compute explicitly for large ranks.
                # But n*m*p is small (6*9*6=324 for 2x3x6, 6*9*6=324? No, 2*3*3=18 terms? No, 2*3*3=18 elements in tensor).
                # The tensor has n*m*p elements.
                # For 2x3x3, it has 18 elements.
                # We can represent the tensor as a 3D array T[n, m, p].
                # T[i, j, k] = 1.
                
                # Residual R[i, j, k] = T[i, j, k] - sum_c U[i, c] V[j, c] W[k, c]
                # Note: U in the search is (n, r), V is (m, r), W is (p, r).
                # We will map back to the interface format at the end.
                
                # Let's use the flattened form in the search too to match the interface.
                # U_flat[i*m+j, c], V_flat[j*p+k, c], W_flat[k*n+i, c].
                
                # This is getting complicated. Let's stick to the simple interface format.
                
                # Compute the current approximation
                # Approx[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # This is not a standard matrix product.
                
                # Let's use the 3D array representation for the search.
                # U3[n, r], V3[m, r], W3[p, r].
                # Approx[i, j, k] = sum_c U3[i, c] V3[j, c] W3[k, c].
                
                # We will search in the 3D space and then map to the interface.
                
                # Initialize U3, V3, W3
                U3 = np.random.randn(n, target_rank)
                V3 = np.random.randn(m, target_rank)
                W3 = np.random.randn(p, target_rank)
                
                for _ in range(100):
                    if time.time() - start_time > max_time:
                        break
                    
                    # Compute Approx
                    Approx = np.einsum('ic,jc,kc->ijk', U3, V3, W3)
                    Error = Approx - 1.0
                    err_norm = np.linalg.norm(Error)
                    
                    if err_norm < 1e-6:
                        # Found a solution
                        # Map to interface
                        U_flat = np.zeros((n*m, target_rank))
                        V_flat = np.zeros((m*p, target_rank))
                        W_flat = np.zeros((p*n, target_rank))
                        for i in range(n):
                            for j in range(m):
                                U_flat[i*m+j] = U3[i]
                        for j in range(m):
                            for k in range(p):
                                V_flat[j*p+k] = V3[j]
                        for k in range(p):
                            for i in range(n):
                                W_flat[k*n+i] = W3[k]
                        
                        # Round to half-integers
                        U_flat = np.round(U_flat * 2) / 2
                        V_flat = np.round(V_flat * 2) / 2
                        W_flat = np.round(W_flat * 2) / 2
                        
                        # Verify
                        Approx_flat = np.einsum('ac,bc,dc->abd', U_flat, V_flat, W_flat)
                        # This is not the right einsum for the flattened tensor.
                        # We need to check if the decomposition is exact.
                        # The condition is: for all i, j, k, sum_c U_flat[i*m+j, c] V_flat[j*p+k, c] W_flat[k*n+i, c] = 1.
                        
                        is_exact = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                                    # Wait, the sum is over c.
                                    # val = sum_c U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                                    val = np.sum(U_flat[i*m+j] * V_flat[j*p+k] * W_flat[k*n+i])
                                    if abs(val - 1.0) > 1e-6:
                                        is_exact = False
                                        break
                                if not is_exact:
                                    break
                            if not is_exact:
                                break
                        
                        if is_exact:
                            return U_flat.tolist(), V_flat.tolist(), W_flat.tolist()
                    
                    # Gradient Descent Step
                    # dL/dU3[i, c] = sum_{j,k} Error[i, j, k] V3[j, c] W3[k, c]
                    grad_U = np.einsum('ijk,jc,kc->ic', Error, V3, W3)
                    grad_V = np.einsum('ijk,ic,kc->jc', Error, U3, W3)
                    grad_W = np.einsum('ijk,ic,jc->kc', Error, U3, V3)
                    
                    lr = 0.01
                    U3 -= lr * grad_U
                    V3 -= lr * grad_V
                    W3 -= lr * grad_W
                
                if err_norm < best_error:
                    best_error = err_norm
                    best_factors = (U3.copy(), V3.copy(), W3.copy())
        
        return None

    # Search for 2x3x3 decomposition
    U3, V3, W3 = None, None, None
    for r in range(15, 19):
        print(f"Searching for rank {r} for 2x3x3...")
        res = search_rank(2, 3, 3, r, max_time=5)
        if res:
            U3, V3, W3 = res
            print(f"Found rank {r} for 2x3x3")
            break
    
    if U3 is None:
        # Fallback to schoolbook for 2x3x3, rank 18
        print("Falling back to schoolbook for 2x3x3")
        r = 18
        U3 = np.zeros((2, r))
        V3 = np.zeros((3, r))
        W3 = np.zeros((3, r))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U3[i, col] = 1
                    V3[j, col] = 1
                    W3[k, col] = 1
                    col += 1
    
    # Duplicate for 2x3x6
    # The 2x3x6 tensor is the direct sum of two 2x3x3 tensors.
    # So we just double the rank.
    r_total = 2 * U3.shape[1]
    U_total = np.zeros((2*3, r_total))
    V_total = np.zeros((3*6, r_total))
    W_total = np.zeros((6*2, r_total))
    
    # First half (k=0,1,2)
    for i in range(2):
        for j in range(3):
            U_total[i*3+j, :U3.shape[1]] = U3[i]
    for j in range(3):
        for k in range(3):
            V_total[j*6+k, :U3.shape[1]] = V3[j]
    for k in range(3):
        for i in range(2):
            W_total[k*2+i, :U3.shape[1]] = W3[k]
    
    # Second half (k=3,4,5)
    offset = U3.shape[1]
    for i in range(2):
        for j in range(3):
            U_total[i*3+j, offset:] = U3[i]
    for j in range(3):
        for k in range(3, 6):
            V_total[j*6+k, offset:] = V3[j]
    for k in range(3, 6):
        for i in range(2):
            W_total[k*2+i, offset:] = W3[k]
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
