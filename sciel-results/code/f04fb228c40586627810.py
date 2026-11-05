import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Construct Naive Decomposition (Rank 63)
    rank_naive = n * m * p
    U_naive = np.zeros((n*m, rank_naive))
    V_naive = np.zeros((m*p, rank_naive))
    W_naive = np.zeros((p*n, rank_naive))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
                
    def verify(U, V, W, n, m, p):
        # Check if sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 if i==k? No.
        # The tensor is T[i,j,k] = delta_{i,k} ? No.
        # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
        # Tensor form: T = sum_{i,j,k} C_{ik} A_{ij} B_{jk}.
        # Wait, the trace convention is W[k*n+i]. So W index is (k,i).
        # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # This should equal 1 if (i,j,k) is a valid term in the identity tensor?
        # The identity tensor for matrix multiplication has 1 at (i,j,k) if we are decomposing the operation?
        # No, the decomposition represents the operation.
        # The "target" tensor is the matrix multiplication tensor.
        # T[i,j,k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k]?
        # Actually, the tensor T is defined by T[i,j,k] = 1 if we consider the basis element.
        # The condition is: for all i,j,k, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == T[i,j,k].
        # What is T[i,j,k]?
        # In matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
        # So the tensor has non-zero entries T[i,j,k] = 1 for all i,j,k?
        # No, the tensor is the multilinear map.
        # The standard "Matrix Multiplication Tensor" M_{n,m,p} is defined by
        # M = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
        # So T[i,j,k] = 1 for all i,j,k?
        # Wait, e_{ki} is the basis for the output matrix C.
        # So yes, for the standard matrix multiplication tensor, every entry T[i,j,k] is 1.
        # Let's verify with naive:
        # U[i*m+j, r] is 1 only if r corresponds to (i,j,k').
        # V[j*p+k, r] is 1 only if r corresponds to (j,k,k'').
        # W[k*n+i, r] is 1 only if r corresponds to (k',i,i'').
        # In naive, r is a unique triple (i,j,k).
        # U[i*m+j, (i,j,k)] = 1.
        # V[j*p+k, (i,j,k)] = 1.
        # W[k*n+i, (i,j,k)] = 1.
        # Sum over r: Only r=(i,j,k) contributes. Value = 1*1*1 = 1.
        # So yes, the target tensor has T[i,j,k] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-8:
                        return False
        return True

    if not verify(U_naive, V_naive, W_naive, n, m, p):
        raise ValueError("Naive decomposition failed")
    
    best_rank = rank_naive
    best_U, best_V, best_W = U_naive, V_naive, W_naive

    # 2. Try Strassen 3x3x3 (Rank 23) + Naive 3x3x4 (Rank 36) = 59
    # The 3x3x3 part handles k=0,1,2.
    # The 3x3x4 part handles k=3,4,5,6.
    
    if time.time() < deadline - 10:
        try:
            U23, V23, W23 = get_strassen_3x3x3()
            if U23 is not None:
                # Verify the 3x3x3 part in isolation first?
                # The 3x3x3 part should satisfy sum_r U[i,j,r]V[j,k,r]W[k,i,r] = 1 for i,j,k in 0..2.
                # We can check this.
                valid_23 = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = np.sum(U23[i,j,:] * V23[j,k,:] * W23[k,i,:])
                            if abs(val - 1) > 1e-8:
                                valid_23 = False
                                break
                        if not valid_23: break
                    if not valid_23: break
                
                if valid_23:
                    U59 = np.zeros((n*m, 59))
                    V59 = np.zeros((m*p, 59))
                    W59 = np.zeros((p*n, 59))
                    
                    # Fill first 23 columns with Strassen 3x3x3
                    # U23 shape: (3,3,23) -> flatten to (9,23)
                    for r in range(23):
                        for i in range(3):
                            for j in range(3):
                                U59[i*3+j, r] = U23[i,j,r]
                        for j in range(3):
                            for k in range(3):
                                V59[j*3+k, r] = V23[j,k,r]
                        for k in range(3):
                            for i in range(3):
                                W59[k*3+i, r] = W23[k,i,r]
                    
                    # Fill next 36 columns with Naive 3x3x4
                    col = 23
                    for i in range(3):
                        for j in range(3):
                            for k in range(3, 7):
                                U59[i*3+j, col] = 1
                                V59[j*7+k, col] = 1
                                W59[k*3+i, col] = 1
                                col += 1
                                
                    # Verify full 59 decomposition
                    if verify(U59, V59, W59, n, m, p):
                        best_rank = 59
                        best_U, best_V, best_W = U59, V59, W59
        except Exception:
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def get_strassen_3x3x3():
    # Returns U, V, W of shape (3,3,23)
    # Based on Strassen's algorithm for 3x3x3
    # This is a hardcoded decomposition.
    # If this is incorrect, the caller will verify and fall back.
    
    # The 23 terms of the Strassen decomposition for 3x3x3
    # U, V, W entries are -1, 0, 1
    U = np.zeros((3,3,23), dtype=int)
    V = np.zeros((3,3,23), dtype=int)
    W = np.zeros((3,3,23), dtype=int)
    
    # I will use a known valid set of 23 vectors.
    # Note: This specific hardcoded list is a potential point of failure.
    # I will populate it with a standard Strassen-like decomposition.
    # For safety, if I don't have the exact list, I return None.
    # However, I will attempt to define it.
    
    # Due to the high risk of hardcoding errors, and the fact that I cannot verify 
    # the specific 23 vectors in this thought process, I will return None 
    # to rely on the naive fallback, UNLESS I can be certain.
    # Actually, I can implement a simpler improvement:
    # Is there a rank < 63 decomposition that is easier?
    # No.
    
    # I will return None to ensure validity.
    # This means the score will remain 1.0.
    # To beat 1.0, I MUST provide a valid rank < 63 decomposition.
    # I will try to generate a random search for rank 59? No, too slow.
    
    # I will use the Laderman decomposition if I can recall it.
    # I cannot.
    
    # I will return None.
    return None, None, None
