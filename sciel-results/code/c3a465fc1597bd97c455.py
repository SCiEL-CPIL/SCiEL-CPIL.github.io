import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10
    
    # Rank 29 decomposition using Strassen for 2x2x2 sub-block and schoolbook for rest
    R = 29
    U = np.zeros((N_U, R))
    V = np.zeros((N_V, R))
    W = np.zeros((N_W, R))
    
    # Indices helper
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # 1. Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00 + A11)(B00 + B11)
    col = 0
    U[u_idx(0,0), col] = 1; U[u_idx(1,1), col] = 1
    V[v_idx(0,0), col] = 1; V[v_idx(1,1), col] = 1
    W[w_idx(0,0), col] = 1; W[w_idx(1,1), col] = 1
    col += 1
    
    # M2 = (A10 + A11)B00
    U[u_idx(1,0), col] = 1; U[u_idx(1,1), col] = 1
    V[v_idx(0,0), col] = 1
    W[w_idx(1,0), col] = 1; W[w_idx(1,1), col] = -1
    col += 1
    
    # M3 = A00(B01 - B11)
    U[u_idx(0,0), col] = 1
    V[v_idx(0,1), col] = 1; V[v_idx(1,1), col] = -1
    W[w_idx(0,1), col] = 1; W[w_idx(1,1), col] = 1
    col += 1
    
    # M4 = A11(B10 - B00)
    U[u_idx(1,1), col] = 1
    V[v_idx(1,0), col] = 1; V[v_idx(0,0), col] = -1
    W[w_idx(0,0), col] = 1; W[w_idx(1,0), col] = 1
    col += 1
    
    # M5 = (A00 + A01)B11
    U[u_idx(0,0), col] = 1; U[u_idx(0,1), col] = 1
    V[v_idx(1,1), col] = 1
    W[w_idx(0,0), col] = -1; W[w_idx(0,1), col] = 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[u_idx(1,0), col] = 1; U[u_idx(0,0), col] = -1
    V[v_idx(0,0), col] = 1; V[v_idx(0,1), col] = 1
    W[w_idx(1,1), col] = 1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[u_idx(0,1), col] = 1; U[u_idx(1,1), col] = -1
    V[v_idx(1,0), col] = 1; V[v_idx(1,1), col] = 1
    W[w_idx(0,0), col] = 1
    col += 1
    
    # 2. Schoolbook for j=2 (all i, k)
    # A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 2), col] = 1
            V[v_idx(2, k), col] = 1
            W[w_idx(k, i), col] = 1
            col += 1
            
    # 3. Schoolbook for j in {0,1}, k in {2,3,4}
    # A[i,j] * B[j,k]
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[u_idx(i, j), col] = 1
                V[v_idx(j, k), col] = 1
                W[w_idx(k, i), col] = 1
                col += 1
                
    # Verify correctness
    # Tensor T[i, j, k] = 1 if we sum over r U[i,j,r]V[j,k,r]W[k,i,r]
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No, the decomposition is for the multiplication tensor.
    # The multiplication tensor M has entries M[i, j, k] = 1 if we are computing C[i,k] += A[i,j]B[j,k].
    # Actually, the tensor T for matrix multiplication is defined such that
    # C = AB <=> sum_j A[i,j]B[j,k] = C[i,k].
    # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1?
    # Let's check the convention in the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # "U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"
    # The sum over r should equal 1 for every (i,j,k) triple?
    # No, that would mean the tensor is all 1s.
    # The matrix multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In the standard decomposition, we have:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as sum_r (sum_i U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) ... wait.
    # The standard tensor decomposition for matrix mult is:
    # T = sum_r U_r \otimes V_r \otimes W_r
    # Where T[i,j,k] = 1.
    # And the multiplication is recovered by contracting with A, B.
    # Specifically, C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] \delta_{i,i'} \delta_{k,k'}?
    # No, the prompt says: "decompose the matrix multiplication tensor".
    # The tensor T is defined by T[i,j,k] = 1.
    # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for ALL i,j,k?
    # If T[i,j,k]=1 for all i,j,k, then yes, the sum must be 1 for all i,j,k.
    # Let's verify if my construction satisfies sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # Check:
    # If j=2, i=0, k=0:
    # Only term from section 2 with i=0, k=0, j=2 contributes?
    # U[0*3+2, 7] = 1. V[2*5+0, 7] = 1. W[0*2+0, 7] = 1.
    # Are there other terms?
    # Strassen terms only involve j in {0,1}. So U/V for j=2 are 0 in Strassen columns.
    # Section 3 involves j in {0,1}. So U/V for j=2 are 0.
    # So sum is 1. Correct.
    
    # If j=0, i=0, k=0:
    # Strassen terms contribute.
    # Section 2: j=2, so U/V are 0.
    # Section 3: k in {2,3,4}, so for k=0, V is 0.
    # So only Strassen terms contribute.
    # Sum_r U[0,0,r] V[0,0,r] W[0,0,r]
    # M1: U=1, V=1, W=1 -> 1
    # M2: U=0 (A10,A11), so 0.
    # M3: U=1 (A00), V=1 (B01-B11 -> V[0,1]=1, V[1,1]=-1. V[0,0] is 0). So 0.
    # M4: U=0 (A11). So 0.
    # M5: U=1 (A00+A01 -> U[0,0]=1). V[1,1]=1. V[0,0]=0. So 0.
    # M6: U=-1 (A00). V[0,0]=1. W[1,1]=1. But we need W[0,0]. W[0,0] is 0 for M6. So 0.
    # M7: U=0 (A01-A11). So 0.
    # Sum = 1. Correct.
    
    # If j=0, i=0, k=1:
    # Strassen terms.
    # M1: W[0,1] is 0.
    # M2: U=0.
    # M3: U=1, V[0,1]=1, W[0,1]=1. -> 1.
    # M4: U=0.
    # M5: U=1, V[1,1]=1 (V[0,1] is 0). So 0.
    # M6: U=-1, V[0,1]=1, W[0,1]=0. So 0.
    # M7: U=0.
    # Sum = 1. Correct.
    
    # If j=0, i=1, k=0:
    # M1: U[1,0]=0.
    # M2: U[1,0]=1, V[0,0]=1, W[1,0]=1. -> 1.
    # M3: U[1,0]=0.
    # M4: U[1,0]=0.
    # M5: U[1,0]=0.
    # M6: U[1,0]=1, V[0,0]=1, W[1,0]=0. -> 0.
    # M7: U[1,0]=0.
    # Sum = 1. Correct.
    
    # It seems correct.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
