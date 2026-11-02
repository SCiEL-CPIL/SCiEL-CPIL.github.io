import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill
    
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: Use tensor product M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2} has rank 2 (schoolbook, optimal)
    # M_{3,4,4} = M_{3,2,2} ⊗ M_{1,2,2}
    # M_{1,2,2} has rank 4 (schoolbook, optimal for 1x2x2)
    # M_{3,2,2} = M_{3,1,1} ⊗ M_{1,2,2}
    # M_{3,1,1} has rank 3 (schoolbook, optimal)
    # So M_{3,2,2} rank = 3*4 = 12
    # M_{3,4,4} rank = 12*4 = 48
    # M_{3,4,8} rank = 48*2 = 96
    
    # That's just schoolbook. We need something better.
    
    # Alternative: Use Winograd/Strassen-like techniques for the 4x4 part
    # M_{1,4,4} can be done with rank < 16 using Strassen on 4x4
    # Actually, let me think about this differently.
    
    # M_{3,4,8}: We can view the 8 as 4+4, so we're computing [A|B] where A and B are 3x4
    # This is equivalent to multiplying a 3x4 matrix by a 4x8 matrix.
    
    # Key insight: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # The tensor product of M_{1,1,2} (rank 2) with M_{3,4,4}
    # If we can find a good decomposition of M_{3,4,4}, multiplying rank by 2 gives M_{3,4,8}
    
    # For M_{3,4,4}:
    # M_{3,4,4} = M_{3,2,2} ⊗ M_{1,2,2}
    # M_{1,2,2} rank 4 (optimal)
    # M_{3,2,2} = M_{3,1,1} ⊗ M_{1,2,2}, rank = 3*4 = 12
    # So M_{3,4,4} rank = 12*4 = 48, and M_{3,4,8} rank = 48*2 = 96
    
    # This is still schoolbook. Let me try a different factorization.
    
    # M_{3,4,8} = M_{3,1,1} ⊗ M_{1,4,8}
    # M_{1,4,8} = M_{1,4,4} ⊗ M_{1,1,2}
    # M_{1,4,4}: Can we do better than 16? 
    # For 1x4x4, the tensor is just a 4x4 matrix. Rank of a 4x4 matrix as a tensor...
    # Actually M_{1,4,4} is the tensor T[i,j,k] = 1 if i=0, else 0. Wait no.
    # M_{a,b,c}[i,j,k] = 1 if i,j,k in range, 0 otherwise? No.
    # The matrix multiplication tensor M_{a,b,c} has entries T[i,j,k] = 1 for all valid i,j,k
    # where the "multiplication" gives output[i,k] += input1[i,j]*input2[j,k]
    # So T[i,j,k] = 1 for all i in [0,a), j in [0,b), k in [0,c)
    
    # So M_{1,4,4} has T[0,j,k] = 1 for j in [0,4), k in [0,4). That's 16 ones.
    # This is just a 4x4 identity-like structure... actually it's the all-ones 4x4 matrix viewed as a tensor.
    # The rank of the all-ones matrix of size r x s is 1 (it's rank 1 as a matrix).
    # But as a tensor M_{1,4,4}, it's T[0,j,k] = delta... no, T[i,j,k] = 1 for all valid indices.
    # So M_{1,4,4}[0,j,k] = 1 for all j,k. This is a 1x4x4 tensor that is all ones.
    # As a tensor, this can be written as sum of rank-1 tensors.
    # T[0,j,k] = 1 = (1/4) * sum_{l=0}^{3} delta[j,l]*delta[k,l] ... no that gives identity.
    # Actually T[0,j,k] = 1 for all j,k. This is u_j * v_k where u = [1,1,1,1], v=[1,1,1,1].
    # So it's rank 1 as a tensor! U[0,0]=1, V[j,0]=1 for all j, W[k,0]=1 for all k.
    # Wait, let me check: sum_c U[0,c]*V[j,c]*W[k,c] = U[0,0]*V[j,0]*W[k,0] = 1*1*1 = 1. Yes!
    
    # So M_{1,4,4} has rank 1! That's much better than 16.
    # Similarly M_{1,4,8} = M_{1,4,4} ⊗ M_{1,1,2}, rank = 1*2 = 2
    # And M_{3,4,8} = M_{3,1,1} ⊗ M_{1,4,8}, rank = 3*2 = 6
    
    # Wait, that seems too good. Let me double-check.
    # M_{a,b,c} tensor: T[i,j,k] = 1 if the (i,k) entry of (A*B) depends on A[i,j]*B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    # So M_{1,4,4}[0,j,k] = 1 for all j in [0,4), k in [0,4).
    # Rank-1 decomposition: T[0,j,k] = U[0,c]*V[j,c]*W[k,c]
    # With c=0: U[0,0]=1, V[j,0]=1 for all j, W[k,0]=1 for all k.
    # Check: U[0,0]*V[j,0]*W[k,0] = 1*1*1 = 1. Correct!
    
    # So M_{1,4,4} rank = 1. 
    # M_{1,4,8} = M_{1,4,4} ⊗ M_{1,1,2}:
    # M_{1,1,2}[0,0,k] = 1 for k in [0,2). Rank 1 similarly.
    # So M_{1,4,8} rank = 1*1 = 1? Let me verify.
    # M_{1,4,8}[0,j,k] = 1 for j in [0,4), k in [0,8).
    # U[0,0]=1, V[j,0]=1 for all j in [0,4), W[k,0]=1 for all k in [0,8).
    # Yes! Rank 1.
    
    # M_{3,4,8} = M_{3,1,1} ⊗ M_{1,4,8}:
    # M_{3,1,1}[i,0,0] = 1 for i in [0,3). Rank 1.
    # So M_{3,4,8} rank = 1*1 = 1?? That can't be right.
    
    # Wait, I think I'm confusing the tensor product structure.
    # M_{a1,b1,c1} ⊗ M_{a2,b2,c2} = M_{a1*a2, b1*b2, c1*c2}
    # The tensor product of two tensors T1[i1,j1,k1] and T2[i2,j2,k2] gives
    # T[i1*a2+i2, j1*b2+j2, k1*c2+k2] = T1[i1,j1,k1]*T2[i2,j2,k2]
    
    # For M_{3,1,1} ⊗ M_{1,4,8} = M_{3*1, 1*4, 1*8} = M_{3,4,8}. Correct.
    # M_{3,1,1}[i,0,0] = 1 for i=0,1,2.
    # M_{1,4,8}[0,j,k] = 1 for j=0..3, k=0..7.
    # Product: T[i, j, k] = M_{3,1,1}[i,0,0] * M_{1,4,8}[0,j,k] = 1*1 = 1 for all valid i,j,k.
    # So M_{3,4,8}[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,8). Correct!
    
    # Now, if M_{3,1,1} has rank 1 and M_{1,4,8} has rank 1, then M_{3,4,8} has rank 1*1 = 1.
    # Let me verify: 
    # M_{3,1,1}: U[i,0]=1 for i=0,1,2; V[0,0]=1; W[0,0]=1.
    # M_{1,4,8}: U[0,0]=1; V[j,0]=1 for j=0..3; W[k,0]=1 for k=0..7.
    # Tensor product: U_prod[i*1+0, 0*1+0] = U1[i,0]*U2[0,0] = 1*1 = 1
    # V_prod[0*4+j, 0*8+k] = V1[0,0]*V2[j,0] = 1*1 = 1
    # W_prod[0*3+i, 0*1+0] = W1[0,0]*W2[0,0] = 1*1 = 1
    # Check: sum_c U[i*4+j, c]*V[j*8+k, c]*W[k*3+i, c]
    # With c=0: U[i*4+j,0]*V[j*8+k,0]*W[k*3+i,0] = 1*1*1 = 1. Correct!
    
    # So rank 1?! That seems impossibly low. Let me reconsider what the tensor actually is.
    
    # OH WAIT. I think I've been misunderstanding the tensor.
    # In matrix multiplication C = A*B where A is n x m and B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    # 
    # A rank-r decomposition means: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (using the trace convention W[k*n+i, r])
    
    # For M_{3,4,8}: T[i,j,k] = 1 for i in [0,3), j in [0,4), k in [0,8).
    # Can this be rank 1? That would mean T[i,j,k] = U[i,j,0]*V[j,k,0]*W[k,i,0] for all i,j,k.
    # This requires U[i,j,0]*V[j,k,0]*W[k,i,0] = 1 for all i,j,k.
    # Taking logs: log U[i,j] + log V[j,k] + log W[k,i] = 0 for all i,j,k.
    # This is a system that may not have a solution in general.
    # For example, fix i=0: log U[0,j] + log V[j,k] + log W[k,0] = 0 for all j,k.
    # Fix i=1: log U[1,j] + log V[j,k] + log W[k,1] = 0 for all j,k.
    # Subtracting: log U[0,j] - log U[1,j] + log W[k,0] - log W[k,1] = 0 for all j,k.
    # This means log W[k,0] - log W[k,1] = log U[1,j] - log U[0,j] for all j,k.
    # So the LHS depends only on k and RHS only on j, so both must be constant.
    # This is possible! E.g., set all U, V, W entries to 1.
    # Then T[i,j,k] = 1*1*1 = 1. 
    
    # So YES, rank 1 works! The all-ones decomposition.
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    
    # Let me verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 = T[i,j,k]. Correct!
    
    # So the answer is rank 1!
    
    U = np.ones((nm, 1))
    V = np.ones((mp, 1))
    W = np.ones((pn, 1))
    
    # Verify
    def verify(U, V, W, n, m, p):
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False, (i, j, k, val)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p)
    if not valid:
        print(f"Verification failed: {info}")
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # This shouldn't happen for rank 1, but safety
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
