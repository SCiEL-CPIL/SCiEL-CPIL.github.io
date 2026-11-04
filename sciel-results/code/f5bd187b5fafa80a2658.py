import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Rank 49 Coefficients (Kronecker product of Strassen 2x2)
    # This is a known valid decomposition.
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutations to match standard 4x4 indexing
    # A: i*4+j
    # B: j*4+k
    # C: k*4+i (Trace convention W[k*n+i])
    
    perm_uv = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i1, i0 = divmod(i, 2)
        j1, j0 = divmod(j, 2)
        P1 = i1 * 2 + j1
        P2 = i0 * 2 + j0
        K = P1 * 4 + P2
        perm_uv[S] = K
        
    perm_w = np.zeros(16, dtype=int)
    for S_c in range(16):
        k = S_c // 4
        i = S_c % 4
        k1, k0 = divmod(k, 2)
        i1, i0 = divmod(i, 2)
        Q1 = k1 * 2 + i1
        Q2 = k0 * 2 + i0
        K = Q1 * 4 + Q2
        perm_w[S_c] = K
        
    U4_fixed = U4[perm_uv, :]
    V4_fixed = V4[perm_uv, :]
    W4_fixed = W4[perm_w, :]
    
    # Verification
    # Check if sum_r U[:,r] V[:,r] W[:,r] == Tensor
    # Tensor T[i,j,k] = 1 if A[i,j]B[j,k] contributes to C[i,k]
    # Actually, we check the decomposition identity:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta? No.
    # Identity: sum_r W[k,i,r] U[i,j,r] V[j,k,r] = 1 for all i,j,k?
    # No, the identity is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomp: C[i,k] = sum_r W[k,i,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    # So sum_r W[k,i,r] U[i,j,r] V[j,k,r] must be 1 for all i,j,k?
    # Let's check for i=0, j=0, k=0.
    # Term A[0,0] B[0,0] contributes to C[0,0].
    # So sum_r W[0,0,r] U[0,0,r] V[0,0,r] should be 1.
    # And for i=0, j=1, k=0.
    # A[0,1] B[1,0] contributes to C[0,0].
    # So sum_r W[0,0,r] U[0,1,r] V[1,0,r] should be 1.
    # Wait, the sum is over r.
    # For a fixed i,j,k, the coefficient of A[i,j]B[j,k] in the decomp is sum_r W[k,i,r] U[i,j,r] V[j,k,r].
    # This must be 1 if the indices match the multiplication rule, and 0 otherwise?
    # No, for any i,j,k, A[i,j]B[j,k] ALWAYS contributes to C[i,k].
    # So the sum must be 1 for ALL i,j,k in 0..3.
    
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(49):
                    s += W4_fixed[k*4+i, r] * U4_fixed[i*4+j, r] * V4_fixed[j*4+k, r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {
            "U": U4_fixed.tolist(),
            "V": V4_fixed.tolist(),
            "W": W4_fixed.tolist()
        }
    else:
        # Fallback to Rank 64 (Schoolbook)
        # U[i,j,r] = 1 if r == i*16 + j*4 + k? No.
        # Rank 64: r corresponds to (i,j,k)? No, r is the term index.
        # Term r corresponds to j?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # Rank 64 is sum over i,j,k? No, that's not a decomposition.
        # Schoolbook is rank 64?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This is already a sum of 4 terms for each (i,k).
        # Total terms = 4*4*4 = 64.
        # U[i,j,r] = 1 if r encodes (i,j,k) and the term is A[i,j]?
        # Let r = i*16 + j*4 + k.
        # U[i',j',r] = 1 if i'=i, j'=j.
        # V[j',k',r] = 1 if j'=j, k'=k.
        # W[k',i',r] = 1 if k'=k, i'=i.
        # Then sum_r W[k,i,r] U[i,j,r] V[j,k,r] = 1 if r matches (i,j,k).
        # Sum over r gives 1.
        
        U64 = np.zeros((16, 64), dtype=int)
        V64 = np.zeros((16, 64), dtype=int)
        W64 = np.zeros((16, 64), dtype=int)
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i*16 + j*4 + k
                    U64[i*4+j, r] = 1
                    V64[j*4+k, r] = 1
                    W64[k*4+i, r] = 1
                    
        return {
            "U": U64.tolist(),
            "V": V64.tolist(),
            "W": W64.tolist()
        }
