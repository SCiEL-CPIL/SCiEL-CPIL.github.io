import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Rank 28 decomposition using Strassen's algorithm for 4 groups of 2 columns.
    # Each group uses 7 Strassen products, total 4*7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrix multiplication.
    # For A (2x2) and B_sub (2x2), C_sub = A * B_sub:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    # Outputs:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {'A': {(0,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0, (1,1): 1.0}, 
         'C': {(0,0): 1.0, (1,1): 1.0}},
        # M2 = (A10+A11)B00
        {'A': {(1,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0}, 
         'C': {(1,0): 1.0, (1,1): -1.0}},
        # M3 = A00(B01-B11)
        {'A': {(0,0): 1.0}, 
         'B': {(0,1): 1.0, (1,1): -1.0}, 
         'C': {(0,1): 1.0, (1,1): 1.0}},
        # M4 = A11(B10-B00)
        {'A': {(1,1): 1.0}, 
         'B': {(1,0): 1.0, (0,0): -1.0}, 
         'C': {(0,0): 1.0, (1,0): 1.0}},
        # M5 = (A00+A01)B11
        {'A': {(0,0): 1.0, (0,1): 1.0}, 
         'B': {(1,1): 1.0}, 
         'C': {(0,0): -1.0, (0,1): 1.0}},
        # M6 = (A10-A00)(B00+B01)
        {'A': {(1,0): 1.0, (0,0): -1.0}, 
         'B': {(0,0): 1.0, (0,1): 1.0}, 
         'C': {(1,1): 1.0}},
        # M7 = (A01-A11)(B10+B11)
        {'A': {(0,1): 1.0, (1,1): -1.0}, 
         'B': {(1,0): 1.0, (1,1): 1.0}, 
         'C': {(0,0): 1.0}},
    ]
    
    # For each group of 2 columns (q = 0,1,2,3), apply Strassen.
    # Group q uses columns k1=2q, k2=2q+1.
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, term in enumerate(strassen_terms):
            t = q * 7 + r  # term index
            
            # Fill U: indexed by i*m + j
            for (i, j), coeff in term['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: indexed by j*p + k, where k = k1 + k_sub
            for (j, k_sub), coeff in term['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: indexed by k*n + i, where k = k1 + k_sub
            for (i, k_sub), coeff in term['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verify the decomposition
    if time.time() - start_time < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Reconstruct C from the tensor decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta for appropriate indices
        
        # Quick verification: check that the tensor decomposition is correct
        # For the matrix multiplication tensor, we need:
        # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the indices are coupled
        
        # Actually, let's verify by computing C from the decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In the tensor framework, this is encoded as:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, that's not quite right either. Let me think more carefully.
        
        # The correct interpretation: the tensor M_{n,m,p} has entries M[i,j,k] = 1
        # and the rank-r decomposition is M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This means for any A, B: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        
        # Hmm, this is getting complicated. Let me just do a direct numerical check.
        # Compute C using the decomposition and compare to A @ B.
        
        # Actually, the simplest check: verify that the bilinear form computed via
        # the decomposition matches the standard matrix product.
        
        # For each output entry C[i,k]:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This should equal: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # No wait, that's not right. The U, V, W factors are not separable that way.
        
        # Let me just trust the Strassen construction and do a spot check.
        
        # Spot check: for i=0, j=0, k=0, the tensor entry should be 1
        # sum_r U[0*2+0, r] * V[0*8+0, r] * W[0*2+0, r] should equal 1
        test_val = np.sum(U[0, :] * V[0, :] * W[0, :])
        
        # More thorough check: verify the full decomposition
        # Build the tensor T from the decomposition and check it matches the identity
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] should be 1 for all valid i,j,k
        
        T_decomposed = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            T_decomposed += np.einsum('ij,kp,pi->ijk', u_col, v_col, w_col)
        
        # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
        # Actually no, the tensor is more subtle. Let me just check that
        # C = A @ B can be recovered from the decomposition.
        
        # The correct check: for random A, B, verify that the decomposition
        # gives the correct C.
        
        # In the tensor decomposition framework for matrix multiplication:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The rank-r decomposition gives r bilinear forms that can be combined to get C.
        
        # Actually, I think the correct verification is:
        # For each r, define:
        #   A_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (scalar)
        #   B_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (scalar)
        #   Then C_r = A_r * B_r (scalar)
        # And C[i,k] = sum_r W[k*n+i, r] * C_r
        # No, that's not right either. The factors are not scalars.
        
        # Let me just do the most direct possible check.
        # The tensor M_{2,2,8} for matrix multiplication has the property that
        # M[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,7}
        # (in the sense that the bilinear form sum_j A[i,j]B[j,k] is represented)
        
        # I'll just verify numerically that the decomposition works by checking
        # a few specific entries.
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should give the coefficient of A[i,j]*B[j,k] in the expression for C.
        # For the matrix multiplication tensor, this should be 1 for all i,j,k.
        
        all_correct = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        all_correct = False
                        break
                if not all_correct:
                    break
            if not all_correct:
                break
        
        if not all_correct:
            # Fallback: use schoolbook rank 32
            rank = 32
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t = i * m * p + j * p + k
                        U[i * m + j, t] = 1.0
                        V[j * p + k, t] = 1.0
                        W[k * n + i, t] = 1.0
    
    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
