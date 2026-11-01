import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def check_decomposition(U, V, W, n, m, p):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T to be the matrix multiplication tensor.
    # T[i, j, k] is 1 if the term A[i,j]B[j,k] exists?
    # The tensor for mat mult is sum_{i,j,k} e_i \otimes e_j \otimes e_k ? No.
    # The bilinear map is (A, B) -> C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has indices i, j, k.
    # T[i, j, k] = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    # In the standard basis, the tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # No, the decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # The action is: C[i, k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i, j] B[j, k] ?
    # Wait, the definition of the tensor decomposition for mat mult:
    # We write the multiplication map as sum_c (u_c \cdot A) (v_c \cdot B) w_c^T ?
    # Let's check the indices in the seed.
    # Seed: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Sum over c: U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # For a fixed c, this is 1 only if i,j,k match the specific c.
    # In schoolbook, each c corresponds to a unique (i,j,k).
    # So the sum is 1 for the specific (i,j,k) and 0 otherwise.
    # This reconstructs the tensor T where T[i,j,k] = 1 for all i,j,k?
    # No, the mat mult tensor is NOT all ones.
    # The mat mult tensor $T_{n,m,p}$ is defined by $T(e_i, e_j, e_k) = 1$ if the indices match the multiplication structure?
    # Actually, the tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ is the all-ones tensor.
    # The matrix multiplication tensor is $T_{MM} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # No.
    # Let's look at the bilinear form.
    # $\langle T, A \otimes B \rangle = \text{trace}(A B)$? No.
    # The standard tensor for $M_{n,m,p}$ is $T = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p e_i \otimes e_j \otimes e_k$?
    # If so, the schoolbook decomposition with 1s is correct for the all-ones tensor.
    # But matrix multiplication is NOT the all-ones tensor.
    # The matrix multiplication tensor has non-zero entries only where the indices connect?
    # In the basis of the vector space of matrices, the tensor is:
    # $T = \sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ik}$? No.
    # Let's check the seed again.
    # The seed constructs U, V, W such that for each (i,j,k), there is a rank-1 term.
    # This implies the tensor being decomposed is $\sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ik}$?
    # If the tensor is the "matrix multiplication tensor", its entries in the standard basis are:
    # $T_{(i,j), (j',k'), (i',k)}$?
    # The interface says: "exactly decompose the matrix multiplication tensor".
    # And the seed is considered valid (score 1.0).
    # So the seed's construction MUST be a valid decomposition of the MM tensor.
    # This implies that the MM tensor in this coordinate system IS the sum of rank-1 tensors corresponding to each (i,j,k) triple.
    # This means the tensor is $T = \sum_{i,j,k} u_{ijk} \otimes v_{ijk} \otimes w_{ijk}$ where these are basis vectors.
    # So the target tensor has 1s at positions corresponding to valid (i,j,k) triples in the multiplication sum.
    # In $C_{ik} = \sum_j A_{ij} B_{jk}$, every triple (i,j,k) is a valid term in the sum.
    # So the tensor is indeed the sum of all $n \cdot m \cdot p$ basis tensors.
    # Therefore, the target tensor is the all-ones tensor in the flattened indices?
    # No, the indices are coupled.
    # But the seed works. So I just need to match the seed's structure.
    
    # To check validity:
    # Reconstruct the tensor from U, V, W.
    # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T_target[i, j, k] = 1 for all i, j, k.
    # Wait, is T_target all ones?
    # If T_target is all ones, then the rank is the rank of the all-ones tensor.
    # The rank of the all-ones tensor of size $N \times M \times P$ is 1? No.
    # The all-ones tensor $J$ has rank 1? $J = \mathbf{1} \otimes \mathbf{1} \otimes \mathbf{1}$.
    # If the target was all ones, rank would be 1.
    # But the seed uses rank 45.
    # So the target is NOT the all-ones tensor in the flat sense.
    # The target is the tensor that defines the bilinear map.
    # The bilinear map is linear in A and B.
    # The tensor is in $V^* \otimes W^* \otimes (V \otimes W)$.
    # The seed's U, V, W are in the domain/codomain spaces.
    # Let's trust the seed's validity.
    # I will verify my candidate by computing the residual against the "implicit" target defined by the seed's logic?
    # No, I can compute the target tensor explicitly.
    # The target tensor $T$ has $T[i, j, k] = 1$ if the term $A_{ij}B_{jk}$ contributes to $C_{ik}$.
    # In standard matrix multiplication, ALL terms $A_{ij}B_{jk}$ contribute to $C_{ik}$.
    # So $T[i, j, k] = 1$ for all $i \in [n], j \in [m], k \in [p]$.
    # If $T$ is all ones, why is the rank 45?
    # Because the decomposition is constrained?
    # No, tensor rank is the minimum number of rank-1 tensors summing to T.
    # If $T = \mathbf{1} \otimes \mathbf{1} \otimes \mathbf{1}$, rank is 1.
    # So $T$ is NOT the all-ones tensor.
    # Why?
    # Ah, the indices in U, V, W are coupled to the matrix entries.
    # $U$ has rows $i, j$. $V$ has rows $j, k$. $W$ has rows $k, i$.
    # The term is $U[i, j, c] V[j, k, c] W[k, i, c]$.
    # The sum over c gives the coefficient for the term $A_{ij} B_{jk}$ in the product?
    # No, the product $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The bilinear map is $\sum_{i,j,k} A_{ij} B_{jk} E_{ik}$.
    # The tensor is $\sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ik}$.
    # In this tensor, the entry at $(E_{ij}, E_{jk}, E_{ik})$ is 1.
    # All other entries are 0.
    # So the tensor is sparse.
    # It has $n \cdot m \cdot p$ non-zero entries.
    # The schoolbook decomposition uses one rank-1 tensor for each non-zero entry.
    # So rank $\le n \cdot m \cdot p$.
    # The question is: can we sum fewer rank-1 tensors to get this sparse tensor?
    # Yes, that's the tensor rank problem.
    
    # So, to check validity, I must reconstruct the sparse tensor and compare it to the target sparse tensor.
    
    pass

def generate_target(n, m, p):
    # Target tensor T[i, j, k] = 1 if i in [n], j in [m], k in [p]
    # Wait, the tensor is in the space of bilinear forms.
    # The basis elements are $E_{ij} \otimes E_{jk} \otimes E_{ik}$.
    # The tensor is the sum of these basis elements.
    # So T[i, j, k] = 1 for all valid i, j, k.
    # My previous confusion: The "all ones" tensor in the flattened vector space is different.
    # Here, the tensor is defined on the specific basis of matrix units.
    # So yes, T[i, j, k] = 1 for all i, j, k.
    
    # But if T[i, j, k] = 1 for all i, j, k, then T is the sum of all basis vectors.
    # Is the rank of this tensor 1?
    # No, because the basis vectors are linearly independent in the tensor product space?
    # No, rank-1 tensors are of the form $u \otimes v \otimes w$.
    # Can $\sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ik}$ be written as a single $u \otimes v \otimes w$?
    # Only if the tensor is separable.
    # This tensor is NOT separable because the indices are coupled in a way that prevents factorization into a single product of vectors in the respective spaces.
    # Specifically, $U$ depends on $(i,j)$, $V$ on $(j,k)$, $W$ on $(k,i)$.
    # The coupling makes it non-separable.
    
    # So, to check validity:
    # Compute $R[i, j, k] = \sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]$.
    # Check if $R[i, j, k] == 1$ for all i, j, k.
    
    pass
