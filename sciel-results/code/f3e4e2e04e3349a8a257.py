import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, coeff
    def add_term(u, v, w, c, col):
        U[u, col] += c
        V[v, col] += c
        W[w, col] += c

    col = 0
    
    # Left Half (k in 0, 1, 2)
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # M1: A00 (B00 + B11)
    # u: A00 (i=0, j=0) -> 0*3+0 = 0
    # v: B00 (j=0, k=0) -> 0*6+0 = 0, B11 (j=1, k=1) -> 1*6+1 = 7
    # w: C00 (k=0, i=0) -> 0*2+0 = 0, C01 (k=1, i=0) -> 1*2+0 = 2
    # Wait, Strassen sums products.
    # We need to decompose each M into rank-1 tensors?
    # No, M is a product of two linear forms.
    # M = (sum c1 A) * (sum c2 B)
    # This is not rank 1 in the tensor sense unless one of the forms is constant?
    # No, $u \otimes v \otimes w$ means $u_i v_j w_k$.
    # The term $A_{00} B_{00}$ is rank 1: $u_0=1, v_0=1, w_0=1$.
    # The term $A_{00} B_{11}$ is rank 1: $u_0=1, v_7=1, w_2=1$.
    # Strassen combines them: $M_1 = A_{00} B_{00} + A_{00} B_{11}$.
    # This is NOT a single rank-1 tensor.
    # It is a sum of two rank-1 tensors.
    # So Strassen reduces the number of MULTIPLICATIONS, but does it reduce the RANK of the tensor?
    # YES. The rank of the tensor is the number of rank-1 tensors in the decomposition.
    # Strassen's algorithm provides a decomposition of the tensor into 7 rank-1 tensors?
    # NO. Strassen computes 7 intermediate values $M_k$.
    # Each $M_k$ is a bilinear form.
    # The final result $C$ is a linear combination of $M_k$.
    # The tensor decomposition is $\sum_{k=1}^7 u^{(k)} \otimes v^{(k)} \otimes w^{(k)}$.
    # Where $u^{(k)}$ are the coefficients of $A$ in $M_k$, $v^{(k)}$ are coefficients of $B$, and $w^{(k)}$ are coefficients in the final sum?
    # No.
    # The rank-1 tensor is defined by the coefficients of the product.
    # If $M_k = \sum A_{ij} a_{ij}^{(k)} \sum B_{jk} b_{jk}^{(k)}$, this is not rank 1.
    # Rank 1 means $T[i,j,k] = u_i v_j w_k$.
    # The term $A_{ij} B_{jk}$ corresponds to $u_i=1, v_j=1, w_k=1$ for that specific $(i,j,k)$.
    # Strassen works by changing the basis.
    # It shows that the tensor can be written as a sum of 7 rank-1 tensors in a transformed basis?
    # No, the rank is an invariant.
    # If Strassen uses 7 multiplications, it implies the rank is at most 7?
    # YES. The rank of $M_{2,2,2}$ is 7.
    # The decomposition is:
    # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
    # The vectors $u_r, v_r, w_r$ are determined by the Strassen algorithm.
    # Specifically, the $r$-th rank-1 tensor corresponds to the $r$-th multiplication step?
    # Not exactly.
    # The 7 multiplications are $M_r = (u_r \cdot A) (v_r \cdot B)$.
    # Then $C_{ik} = \sum_r w_{r, ik} M_r$.
    # This can be rewritten as:
    # $C_{ik} = \sum_r w_{r, ik} (u_r \cdot A) (v_r \cdot B)$.
    # $C_{ik} = \sum_r (\sum_j u_{r, ij} A_{ij}) (\sum_j v_{r, jk} B_{jk}) w_{r, ik}$.
    # This is not quite $u_i v_j w_k$.
    # The tensor is $T[i,j,k]$.
    # The decomposition is $\sum_r u_r[i] v_r[j] w_r[k]$.
    # For $M_{2,2,2}$, the 7 rank-1 tensors are:
    # 1. $u=[1,0,0,1]^T$? No, $u$ has dim 4.
    # The indices are $(i,j)$.
    # The 7 vectors $u_r, v_r, w_r$ are:
    # We can extract them from the Strassen algorithm.
    
    # Let's just use the known rank-7 decomposition for 2x2x2.
    # And combine them.
    
    pass
