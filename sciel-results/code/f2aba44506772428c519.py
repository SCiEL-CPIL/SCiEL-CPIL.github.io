import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # The tensor for matrix multiplication C = A B is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In the decomposition: T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all i,j,k (since every term A_{i,j}*B_{j,k} contributes to C_{i,k})
    
    # Wait, I need to reconsider. The tensor T[i,j,k] represents the coefficient
    # such that C_{i,k} = sum_j T[i,j,k] A_{i,j} B_{j,k}.
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    
    # But this means the rank is 1 (all-ones tensor), which contradicts known results.
    
    # Let me reconsider the problem. The "matrix multiplication tensor" in the
    # context of bilinear complexity is actually:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices are linked.
    
    # Actually, I think the issue is that the tensor is NOT simply all-ones.
    # Let me re-examine.
    
    # The bilinear map is: (A, B) -> C where C = AB.
    # This can be viewed as a tensor T in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} such that:
    # <T, A ⊗ B> = C (appropriately indexed)
    
    # More precisely, T_{(i,j),(j',k),(i',k')} = 1 if i=i', j=j', k=k' else 0.
    # So T is a "permutation-like" tensor, not all-ones.
    
    # When we flatten: U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    # The reconstruction is: T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'
    
    # The target tensor T[a,b,c] = 1 if a, b, c are "compatible", i.e.,
    # a = i*m+j, b = j*p+k, c = k*n+i where j from a equals j from b
    # and i from a equals i from c, k from b equals k from c.
    
    # In other words, T_{i,j,k} = 1 for all i, j, k (using the multi-index notation
    # where i is row of A, j is column of A / row of B, k is column of B).
    
    # So the target tensor IS the all-ones tensor of size (n*m) x (m*p) x (p*n)
    # restricted to compatible indices. But when we write it as a 3D array T[i,j,k],
    # it's 1 for all i, j, k.
    
    # Therefore, the rank-1 decomposition works:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_r U V W = 1*1*1 = 1 = T[i,j,k]. 
    
    # So the rank is 1?!
    
    # Let me verify this makes sense. If rank is 1, then:
    # C_{i,k} = (sum_j U_{i,j,0} A_{i,j}) * (sum_k' V_{j',k',0} B_{j',k'}) * W_{k,i,0}
    # With all 1s: C_{i,k} = (sum_j A_{i,j}) * (sum_{j',k'} B_{j',k'}) * 1
    # This is NOT matrix multiplication!
    
    # So I was wrong. The rank-1 decomposition does NOT compute matrix multiplication.
    
    # Let me reconsider. The correct formulation is:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) * (sum_{j'} V_{j',k,r} B_{j',k}) * W_{k,i,r}
    # No, that's not right either.
    
    # Let me look at this more carefully. The standard formulation for matrix
    # multiplication as a tensor decomposition is:
    # C_{i,k} = sum_{r=1}^R (sum_j U_{i,j,r} A_{i,j}) * (sum_k' V_{j',k',r} B_{j',k'}) ... no.
    
    # Actually, I think the correct formulation is:
    # C_{i,k} = sum_r sum_j U_{i,j,r} V_{j,k,r} W_{k,i,r} A_{i,j} B_{j,k}
    # No, that's not right.
    
    # Let me just use the schoolbook decomposition which is guaranteed to be correct.
    # Schoolbook: rank = n*m*p = 3*5*7 = 105
    
    # For each (i, j, k), we have a term:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m*p + j*p + k
    
    # This gives: T_recon[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # = 1 (only when r corresponds to (i,j,k))
    # = 0 (otherwise)
    
    # And T[i,j,k] = 1 for all i,j,k, so this matches.
    
    # Wait, but that would mean T_recon[i,j,k] = 1 only for the specific (i,j,k)
    # that corresponds to r, not for all (i,j,k).
    
    # I think I'm confusing myself. Let me just implement the schoolbook and move on.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
