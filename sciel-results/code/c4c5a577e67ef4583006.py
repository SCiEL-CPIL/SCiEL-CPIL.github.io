"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W (lists of lists) form a valid decomposition.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    """
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = U_mat.shape[1]
    if V_mat.shape[1] != rank or W_mat.shape[1] != rank:
        return False
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # And compare with W[k*n+i,c]... wait, no.
    # The decomposition is: T_{ijk} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}? 
    # No, standard matrix multiplication tensor: T_{ijk} = sum_c U_{ij,c} V_{jk,c} ... 
    # Actually, the convention is: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has entries T[i,j,k] = 1 if we're multiplying A_{ij} * B_{jk} to get C_{ik}.
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] should equal 1 for the "1" entries?
    # No. The decomposition is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]?
    # No, that's not right either. Let's re-read the task.
    # "U, V, W factor matrices... exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T is a 3-tensor where T[i,j,k] = 1 if the product A_{ij}B_{jk} contributes to C_{ik}.
    # Actually, the standard definition is: C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k? No, that's not a tensor decomposition.
    # The tensor T is defined by T[i,j,k] = 1. We want to write T as sum_c U_c V_c W_c where
    # U_c is a vector of size n*m, V_c is a vector of size m*p, W_c is a vector of size p*n.
    # And the decomposition is: T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
    # Wait, the task says "output factor follows the official trace convention W[k*n+i,r]".
    # So W is indexed by (k*n+i).
    # Let's check the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 (since only one c matches).
    # So the tensor entry T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
    # And we want T[i,j,k] = 1 for all i,j,k? No, that would mean the tensor is all ones.
    # But matrix multiplication tensor is not all ones.
    # Let's think again. The matrix multiplication tensor T_{n,m,p} is a tensor in (R^{n*m} \otimes R^{m*p} \otimes R^{p*n}).
    # Its entries are T[i,j,k] = 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, T[i,j,k] = 1 for all i,j,k is not correct.
    # The correct definition: T[i,j,k] = 1 if we multiply A_{ij} and B_{jk} to get a contribution to C_{ik}.
    # But C_{ik} = sum_j A_{ij} B_{jk}. So for each (i,k), we sum over j.
    # The tensor T is defined such that the bilinear form <T, A \otimes B> = C.
    # So T[i,j,k] = 1 means that A_{ij} * B_{jk} contributes to C_{ik}.
    # But in the sum C_{ik} = sum_j A_{ij} B_{jk}, each term A_{ij} B_{jk} contributes to C_{ik}.
    # So T[i,j,k] = 1 for all i,j,k? That would make T the all-ones tensor.
    # But that's not right. Let's look at the schoolbook decomposition again.
    # In the schoolbook, for each (i,j,k), we have one rank-1 term: U_{ij,c}=1, V_{jk,c}=1, W_{ki,c}=1.
    # So T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    # So the tensor T is indeed the all-ones tensor? That can't be right.
    # Wait, I think I'm confusing the tensor with the multiplication map.
    # The matrix multiplication map is: (A, B) -> C where C = AB.
    # This is a bilinear map. The associated tensor T is in V* \otimes W* \otimes Z where V=R^{n*m}, W=R^{m*p}, Z=R^{p*n}.
    # The tensor T is defined by <T, a \otimes b> = c where c = a * b (matrix product).
    # So T[i,j,k] is the coefficient of e_i \otimes e_j \otimes e_k in T.
    # Actually, T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_k where the sum is over all (i,j,k) such that...
    # No, T is defined by: for any vectors a in R^{n*m}, b in R^{m*p}, the vector c = T(a,b) in R^{p*n} is given by
    # c_k = sum_{i,j} T[i,j,k] a_i b_j.
    # And we want c = A*B where A is n x m, B is m x p, C is n x p.
    # So C_{ik} = sum_j A_{ij} B_{jk}.
    # In terms of the flattened vectors: a_{i*m+j} = A_{ij}, b_{j*p+k} = B_{jk}, c_{k*n+i} = C_{ik}.
    # So c_{k*n+i} = sum_j a_{i*m+j} b_{j*p+k}.
    # This means T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # So the tensor T is sparse: T[x,y,z] = 1 if x = i*m+j, y = j*p+k, z = k*n+i for some i,j,k.
    # And the decomposition is: T[x,y,z] = sum_c U[x,c] V[y,c] W[z,c].
    # In the schoolbook, for each (i,j,k), we set U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Then T[x,y,z] = sum_c U[x,c] V[y,c] W[z,c] = 1 if (x,y,z) corresponds to some (i,j,k), else 0.
    # So the tensor is indeed sparse, and the schoolbook decomposition is correct.
    
    # So to verify: for all x in [0, n*m), y in [0, m*p), z in [0, p*n):
    #   sum_c U[x,c] V[y,c] W[z,c] should equal T[x,y,z]
    # where T[x,y,z] = 1 if there exist i,j,k such that x=i*m+j, y=j*p+k, z=k*n+i, else 0.
    
    # Let's build T explicitly.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Now check: sum_c U[x,c] V[y,c] W[z,c] == T[x,y,z] for all x,y,z.
    # This is equivalent to: the tensor U \otimes V \otimes W (with the right indices) equals T.
    # We can check this by computing the 3-tensor from U,V,W and comparing.
    
    # Efficient check: for each c, the rank-1 tensor is U[:,c] \otimes V[:,c] \otimes W[:,c].
    # The sum over c is the 3-tensor.
    # We can check by contracting: for each x,y, sum_c U[x,c] V[y,c] W[z,c] as a function of z.
    # Alternatively, we can check: for each x,y, the vector (sum_c U[x,c] V[y,c] W[:,c]) should equal T[x,y,:].
    
    # Let's do: M = U @ V.T  # (n*m) x (m*p), M[x,y] = sum_c U[x,c] V[y,c]
    # Then for each x,y, we need sum_c U[x,c] V[y,c] W[:,c] = T[x,y,:].
    # This is not directly M. We need: for each x,y, the vector sum_c (U[x,c] V[y,c]) W[:,c] = T[x,y,:].
    # This is: (U[x,:] \otimes V[y,:]) @ W.T? No.
    # Let's just do it directly for small sizes.
    
    # Since n*m, m*p, p*n are small (24, 12, 8), we can afford a direct check.
    for x in range(n*m):
        for y in range(m*p):
            for z in range(p*n):
                val = 0.0
                for c in range(rank):
                    val += U_mat[x, c] * V_mat[y, c] * W_mat[z, c]
                if abs(val - T[x, y, z]) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition using random initialization and ALS-like refinement."""
    # This is a placeholder; for now, we just return the schoolbook if r >= n*m*p.
    # A proper implementation would use numerical optimization.
    if r >= n * m * p:
        return _schoolbook(n, m, p)
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try to find a better decomposition.
    # For now, we just return the schoolbook decomposition.
    # Future work: implement numerical search for lower rank.
    
    result = _schoolbook(n, m, p)
    return result
# EVOLVE-BLOCK-END
