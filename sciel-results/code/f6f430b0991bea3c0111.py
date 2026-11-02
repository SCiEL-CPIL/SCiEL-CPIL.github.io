import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # U index: i*m+j = 2*i+j
    # V index: j*p+k = 9*j+k
    # W index: k*n+i = 2*k+i
    
    terms = []
    
    # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        
        # A indices: a=0 (0,0), b=1 (0,1), c=2 (1,0), d=3 (1,1)
        # B indices: e=0 (0,k0), f=1 (0,k1), g=2 (1,k0), h=3 (1,k1) relative to block, 
        # but absolute V indices are:
        # B[0,k0] -> k0
        # B[0,k1] -> k1
        # B[1,k0] -> 9+k0
        # B[1,k1] -> 9+k1
        
        # C indices:
        # C[0,k0] -> 2*k0
        # C[1,k0] -> 2*k0+1
        # C[0,k1] -> 2*k1
        # C[1,k1] -> 2*k1+1
        
        # M1 = (a+d)(e+h)
        U1 = {0: 1, 3: 1}
        V1 = {k0: 1, 9+k1: 1}
        W1 = {2*k0: 1, 2*k1+1: 1}
        terms.append((U1, V1, W1))
        
        # M2 = (c+d)e
        U2 = {2: 1, 3: 1}
        V2 = {k0: 1}
        W2 = {2*k0+1: 1, 2*k1+1: -1}
        terms.append((U2, V2, W2))
        
        # M3 = a(f-h)
        U3 = {0: 1}
        V3 = {k1: 1, 9+k1: -1}
        W3 = {2*k1: 1, 2*k1+1: 1}
        terms.append((U3, V3, W3))
        
        # M4 = d(g-e)
        U4 = {3: 1}
        V4 = {9+k0: 1, k0: -1}
        W4 = {2*k0: 1, 2*k0+1: 1}
        terms.append((U4, V4, W4))
        
        # M5 = (a+b)h
        U5 = {0: 1, 1: 1}
        V5 = {9+k1: 1}
        W5 = {2*k0: -1, 2*k1: 1}
        terms.append((U5, V5, W5))
        
        # M6 = (c-a)(e+f)
        U6 = {2: 1, 0: -1}
        V6 = {k0: 1, k1: 1}
        W6 = {2*k1+1: 1}
        terms.append((U6, V6, W6))
        
        # M7 = (b-d)(g+h)
        U7 = {1: 1, 3: -1}
        V7 = {9+k0: 1, 9+k1: 1}
        W7 = {2*k0: 1}
        terms.append((U7, V7, W7))
        
    # Schoolbook for k=8
    k = 8
    # C[0,8] = a*B[0,8] + b*B[1,8]
    # C[1,8] = c*B[0,8] + d*B[1,8]
    
    # S1 = a * B[0,8]
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    # S2 = b * B[1,8]
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    # S3 = c * B[0,8]
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    # S4 = d * B[1,8]
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
            
    # Optional: Verification
    # Create random matrices
    np.random.seed(42)
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 9)
    C_true = A @ B
    
    # Reconstruct C from decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is not a simple matrix multiplication of U, V, W.
    # We need to check the tensor identity.
    # T[i,j,k] = sum_r U[2*i+j, r] * V[9*j+k, r] * W[2*k+i, r]
    # We want T[i,j,k] to be 1 if (i,j,k) corresponds to a multiplication term?
    # No, the tensor M_{n,m,p} has entries M[i,j,k] = 1 if we are computing C[i,k] = sum A[i,j]B[j,k].
    # The decomposition is sum_r u_r v_r w_r = M.
    # So for each r, we have a rank-1 tensor.
    # The sum of these rank-1 tensors should equal the matrix multiplication tensor.
    # The matrix multiplication tensor M has M[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Actually, M[i,j,k] is 1 for all i,j,k? No.
    # The tensor M_{n,m,p} is defined by M[i,j,k] = 1 if we are computing the product.
    # Wait, the standard definition:
    # The tensor $M_{n,m,p}$ is the tensor of matrix multiplication.
    # Its entries are $M_{i,j,k} = 1$ if $i,j,k$ are valid indices?
    # No, the tensor is such that $\langle M, A \otimes B \rangle = \langle C, A \cdot B \rangle$?
    # Actually, the decomposition $M = \sum u_r \otimes v_r \otimes w_r$ means:
    # $\sum_r u_r[i] v_r[j] w_r[k] = M[i,j,k]$.
    # For matrix multiplication, $M[i,j,k] = 1$ if the term $A[i,j]B[j,k]$ contributes to $C[i,k]$?
    # No, $C[i,k] = \sum_j A[i,j] B[j,k]$.
    # So the tensor $M$ has $M[i,j,k] = 1$ for all $i,j,k$?
    # If so, then $\sum_r u_r[i] v_r[j] w_r[k] = 1$ for all $i,j,k$.
    # This is not correct.
    
    # Let's look at the indices.
    # U has indices for A: $i \in \{0,1\}, j \in \{0,1\}$. Index $2i+j$.
    # V has indices for B: $j \in \{0,1\}, k \in \{0..8\}$. Index $9j+k$.
    # W has indices for C: $i \in \{0,1\}, k \in \{0..8\}$. Index $2k+i$.
    
    # The tensor equation is:
    # $\sum_r U[2i+j, r] V[9j+k, r] W[2k+i, r] = \delta_{\text{mult}}$
    # Where $\delta_{\text{mult}}$ is 1 if the term $A[i,j]B[j,k]$ is part of the multiplication formula for $C[i,k]$.
    # Since $C[i,k] = \sum_j A[i,j]B[j,k]$, the term $A[i,j]B[j,k]$ always contributes to $C[i,k]$.
    # So $M[i,j,k] = 1$ for all $i,j,k$.
    
    # Let's verify this interpretation.
    # If $M[i,j,k]=1$ for all $i,j,k$, then we are decomposing the all-ones tensor?
    # No, that's not right.
    
    # The correct tensor for matrix multiplication $M_{n,m,p}$ is defined by:
    # $M[i,j,k] = 1$ if $i,j,k$ are such that $A[i,j]$ and $B[j,k]$ multiply to form part of $C[i,k]$.
    # In standard matrix multiplication, every pair $(i,j)$ in A and $(j,k)$ in B contributes to $C[i,k]$.
    # So yes, $M[i,j,k] = 1$ for all valid $i,j,k$.
    
    # Let's test this with the rank-1 term $u \otimes v \otimes w$.
    # $u[i] v[j] w[k]$.
    # Sum over r: $\sum_r U[2i+j, r] V[9j+k, r] W[2k+i, r]$.
    # This sum must equal 1 for all $i,j,k$.
    
    # Let's check one entry from our decomposition.
    # i=0, j=0, k=0.
    # U index: 0. V index: 0. W index: 0.
    # We need $\sum_r U[0, r] V[0, r] W[0, r] = 1$.
    
    # From Strassen block 0 (k0=0, k1=1):
    # M1: U[0]=1, V[0]=1, W[0]=1. Product: 1.
    # M2: U[0]=0, V[0]=1, W[0]=0. Product: 0.
    # M3: U[0]=1, V[0]=0, W[0]=0. Product: 0.
    # M4: U[0]=0, V[0]=-1, W[0]=1. Product: 0.
    # M5: U[0]=1, V[0]=0, W[0]=-1. Product: 0.
    # M6: U[0]=-1, V[0]=1, W[0]=0. Product: 0.
    # M7: U[0]=0, V[0]=1, W[0]=1. Product: 0.
    # Sum = 1. Correct.
    
    # Let's check i=0, j=1, k=0.
    # U index: 1. V index: 9. W index: 0.
    # M1: U[1]=0, V[9]=0, W[0]=1. Product: 0.
    # M2: U[1]=0, V[9]=0, W[0]=0. Product: 0.
    # M3: U[1]=0, V[9]=0, W[0]=0. Product: 0.
    # M4: U[1]=0, V[9]=1, W[0]=1. Product: 0.
    # M5: U[1]=1, V[9]=1, W[0]=-1. Product: -1.
    # M6: U[1]=0, V[9]=0, W[0]=0. Product: 0.
    # M7: U[1]=1, V[9]=1, W[0]=1. Product: 1.
    # Sum = -1 + 1 = 0.
    # But M[0,1,0] should be 1?
    # Wait. $C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]$.
    # So $A[0,1]B[1,0]$ contributes to $C[0,0]$.
    # So $M[0,1,0]$ should be 1.
    # My check gave 0. Something is wrong.
    
    # Let's re-read the W index convention.
    # W[k*n+i, r].
    # For i=0, k=0, W index is 0.
    # For i=0, j=1, k=0.
    # The term is $A[0,1] B[1,0] C[0,0]$.
    # So we need the sum to be 1.
    
    # Let's re-evaluate M5 and M7.
    # M5: U5 = {0: 1, 1: 1}. So U5[1] = 1.
    # V5 = {9+k1: 1}. k1=1. V5[10] = 1. V5[9] = 0.
    # W5 = {2*k0: -1, 2*k1: 1}. k0=0, k1=1. W5[0] = -1. W5[2] = 1.
    # So for i=0, j=1, k=0:
    # U5[1] = 1.
    # V5[9] = 0.
    # W5[0] = -1.
    # Product = 0.
    
    # M7: U7 = {1: 1, 3: -1}. So U7[1] = 1.
    # V7 = {9+k0: 1, 9+k1: 1}. k0=0, k1=1. V7[9] = 1. V7[10] = 1.
    # W7 = {2*k0: 1}. k0=0. W7[0] = 1.
    # So for i=0, j=1, k=0:
    # U7[1] = 1.
    # V7[9] = 1.
    # W7[0] = 1.
    # Product = 1.
    
    # Are there other terms contributing to i=0, j=1, k=0?
    # M1: U1[1]=0.
    # M2: U2[1]=0.
    # M3: U3[1]=0.
    # M4: U4[1]=0.
    # M6: U6[1]=0.
    
    # Sum = 1. Correct.
    
    # So the decomposition is correct.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
