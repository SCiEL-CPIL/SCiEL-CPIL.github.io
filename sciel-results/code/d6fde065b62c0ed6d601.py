import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build the decomposition term by term.
    # Each term is (u_vec, v_vec, w_vec)
    terms = []
    
    def add_term(u_idx, v_idx, w_idx):
        """
        u_idx: (i, j) for A[i,j]
        v_idx: (j, k) for B[j,k]
        w_idx: (i, k) for C[i,k]
        """
        u_row = u_idx[0] * m + u_idx[1]
        v_row = v_idx[0] * p + v_idx[1]
        w_row = w_idx[1] * n + w_idx[0] # W[k*n+i]
        
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        u_vec[u_row] = 1
        v_vec[v_row] = 1
        w_vec[w_row] = 1
        
        terms.append((u_vec, v_vec, w_vec))

    def add_strassen_2x2x2(a_offset, b_offset, c_offset):
        """
        Adds 7 terms for a 2x2x2 block.
        a_offset: (i0, j0) top-left of A block
        b_offset: (j0, k0) top-left of B block
        c_offset: (i0, k0) top-left of C block
        """
        i0, j0 = a_offset
        # b_offset should be (j0, k0)
        # c_offset should be (i0, k0)
        
        # Strassen multipliers
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to express these as linear combinations of basis tensors.
        # This is tricky because Strassen uses sums/differences.
        # The rank-1 decomposition of the Strassen algorithm is not just 7 basis tensors.
        # It is 7 terms, each of which is a product of linear combinations.
        # Wait, the tensor decomposition requires rank-1 tensors $u \otimes v \otimes w$.
        # Strassen's algorithm computes $C$ using 7 multiplications $M_k$.
        # Each $M_k$ is a scalar (or matrix element) computed from $A$ and $B$.
        # The contribution to $C$ is linear.
        # To get a tensor decomposition, we need to expand the linear combinations.
        # Actually, the "rank" of the Strassen algorithm is 7 in the sense of arithmetic complexity,
        # but the tensor rank might be higher if we require the vectors $u,v,w$ to be basis vectors?
        # No, the vectors $u,v,w$ can be any vectors.
        # In a rank-1 tensor $u \otimes v \otimes w$, $u$ is a vector in $U$, $v$ in $V$, $w$ in $W$.
        # The product $u \otimes v \otimes w$ corresponds to the outer product.
        # If $u = e_i + e_j$, $v = e_k + e_l$, $w = e_m$,
        # then $u \otimes v \otimes w = (e_i+e_j) \otimes (e_k+e_l) \otimes e_m = e_i e_k e_m + e_i e_l e_m + e_j e_k e_m + e_j e_l e_m$.
        # This is a sum of 4 basis tensors.
        # So one "Strassen term" might expand to multiple basis tensors in the standard basis.
        # However, the rank of the tensor is the minimum number of rank-1 tensors (with arbitrary vectors) needed to sum to $T$.
        # So if we use Strassen, we are using 7 rank-1 tensors where the vectors are linear combinations of basis vectors.
        # This is valid! The vectors $U, V, W$ columns don't have to be basis vectors.
        
        # So, for each Strassen multiplier $M_k$, we define the vectors $u_k, v_k, w_k$.
        # $M_k$ is a scalar.
        # $M_k = u_k^T A v_k$? No.
        # $M_k$ is formed by linear combinations of $A$ and $B$ elements.
        # Let's define the vectors explicitly.
        
        # M1: (A11+A22)(B11+B22)
        # u1: A11, A22 -> 1. Others 0.
        # v1: B11, B22 -> 1. Others 0.
        # w1: C11, C22? No, M1 contributes to C11 and C22.
        # This is the problem. A single rank-1 tensor $u \otimes v \otimes w$ produces a matrix $u v^T$?
        # No, $u \in U (n \times m)$, $v \in V (m \times p)$, $w \in W (n \times p)$.
        # The tensor is $\sum_{i,j,k} T_{ijk} e_i^U e_j^V e_k^W$.
        # A rank-1 term is $u \otimes v \otimes w$.
        # $u$ has components $u_{ij}$, $v$ has $v_{jk}$, $w$ has $w_{ik}$.
        # The term is $\sum_{i,j,k} u_{ij} v_{jk} w_{ik} e_i^U e_j^V e_k^W$.
        # This is NOT the same as computing a scalar $M$ and adding it to $C$.
        # In Strassen, we compute $M = \text{lin}(A) \cdot \text{lin}(B)$.
        # Then $C = \sum M \cdot \text{lin}(C)$.
        # This structure $M \cdot w$ is not a rank-1 tensor $u \otimes v \otimes w$ unless $u$ and $v$ are such that $u^T \dots v$ is the scalar.
        # Actually, the bilinear form $A, B \mapsto C$ can be written as $\sum_{r} (u_r^T A) (v_r^T B) w_r$?
        # No, the tensor is $T = \sum_{i,j,k} T_{ijk} e_i \otimes e_j \otimes e_k$.
        # A rank-1 decomposition is $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
        # This means $T_{ijk} = \sum_{r=1}^R u_{r,ij} v_{r,jk} w_{r,ik}$?
        # No. $u_r$ is a vector in $\mathbb{R}^{n \times m}$. $v_r$ in $\mathbb{R}^{m \times p}$. $w_r$ in $\mathbb{R}^{n \times p}$.
        # The term $u_r \otimes v_r \otimes w_r$ is a tensor.
        # Its component at $(i,j,k)$ is $u_{r,ij} v_{r,jk} w_{r,ik}$?
        # No, the indices must match.
        # $u$ is indexed by $(i,j)$. $v$ by $(j,k)$. $w$ by $(i,k)$.
        # The tensor product $u \otimes v \otimes w$ is not defined in the same way as matrix multiplication.
        # The tensor $T$ lives in $U \otimes V \otimes W$.
        # A rank-1 element is $u \otimes v \otimes w$.
        # The coefficient of $e_{ij}^U \otimes e_{jk}^V \otimes e_{ik}^W$ in $u \otimes v \otimes w$ is $u_{ij} v_{jk} w_{ik}$.
        # So yes, $T_{ijk} = \sum_{r} u_{r,ij} v_{r,jk} w_{r,ik}$.
        
        # For Strassen:
        # We want to compute $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # Strassen computes $C$ using 7 scalars $M_r$.
        # $M_r = \sum_{ij} \alpha_{r,ij} A_{ij} \sum_{jk} \beta_{r,jk} B_{jk}$?
        # No, $M_r$ is a product of two linear forms.
        # $M_r = L_r(A) \cdot R_r(B)$.
        # And $C = \sum_r M_r K_r$ where $K_r$ is a fixed matrix.
        # $C_{ik} = \sum_r L_r(A) R_r(B) (K_r)_{ik}$.
        # This can be written as $\sum_{r} \sum_{ij} \sum_{jk} \alpha_{r,ij} A_{ij} \beta_{r,jk} B_{jk} \gamma_{r,ik}$.
        # This is a sum of terms $A_{ij} B_{jk} C_{ik}$ with coefficients.
        # Each term $A_{ij} B_{jk} C_{ik}$ is a basis tensor.
        # So Strassen expands to many basis tensors.
        # The rank of the tensor is the number of terms in the sum $\sum u_r \otimes v_r \otimes w_r$.
        # If we use Strassen, we are using 7 terms of the form $u_r \otimes v_r \otimes w_r$?
        # No. $M_r = L_r(A) R_r(B)$.
        # $L_r(A) = \sum_{ij} \alpha_{r,ij} A_{ij}$.
        # $R_r(B) = \sum_{jk} \beta_{r,jk} B_{jk}$.
        # $C = \sum_r L_r(A) R_r(B) K_r$.
        # $C_{ik} = \sum_r (\sum_{ij} \alpha_{r,ij} A_{ij}) (\sum_{jk} \beta_{r,jk} B_{jk}) \gamma_{r,ik}$.
        # This is NOT a rank-1 tensor decomposition in the form $\sum u_r \otimes v_r \otimes w_r$.
        # A rank-1 tensor $u \otimes v \otimes w$ has entries $u_{ij} v_{jk} w_{ik}$.
        # The Strassen term contributes $\sum_{ij, jk} \alpha_{ij} \beta_{jk} \gamma_{ik} A_{ij} B_{jk}$.
        # This is a sum of $|supp(\alpha)| |supp(\beta)|$ basis tensors.
        # So the rank of the Strassen algorithm in terms of tensor rank is NOT 7.
        # It is the sum of the supports.
        
        # Wait. The tensor rank is the minimum $R$ such that $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
        # If we set $u_r, v_r, w_r$ to be the vectors from Strassen, does it work?
        # $u_r$ is a vector in $U$. $v_r$ in $V$. $w_r$ in $W$.
        # $u_r \otimes v_r \otimes w_r$ has entries $u_{r,ij} v_{r,jk} w_{r,ik}$.
        # We need $\sum_r u_{r,ij} v_{r,jk} w_{r,ik} = \delta_{ij} \delta_{jk} \delta_{ik}$? No.
        # We need the tensor to be the multiplication tensor.
        # $T_{ijk} = 1$ if $j=j$ (always), so $T_{ijk} = 1$ for all $i,j,k$?
        # No. $T_{ijk}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So $T_{ijk} = 1$ for all $i,j,k$.
        # So we need $\sum_r u_{r,ij} v_{r,jk} w_{r,ik} = 1$ for all $i,j,k$.
        
        # For Strassen:
        # $C_{ik} = \sum_r M_r (K_r)_{ik}$.
        # $M_r = \sum_{ij} \alpha_{r,ij} A_{ij} \sum_{jk} \beta_{r,jk} B_{jk}$.
        # This form is NOT separable as $u_{r,ij} v_{r,jk} w_{r,ik}$.
        # It is a sum over $ij$ and $jk$.
        # So Strassen does NOT give a rank-7 tensor decomposition.
        # It gives a decomposition into 7 terms, each of which is a sum of many rank-1 basis tensors.
        # The total number of rank-1 basis tensors is the rank.
        
        # So my previous calculation of rank 99 was WRONG.
        # The rank of $2 \times 2 \times 2$ is 7, but that is the rank of the *algorithm* (number of multiplications).
        # The tensor rank of $2 \times 2 \times 2$ is 7.
        # Why?
        # Because there exist vectors $u_r, v_r, w_r$ such that $\sum_{r=1}^7 u_{r,ij} v_{r,jk} w_{r,ik} = 1$.
        # Yes, Strassen's algorithm provides such vectors.
        # $u_r$ is the vector of coefficients $\alpha_{r,ij}$.
        # $v_r$ is the vector of coefficients $\beta_{r,jk}$.
        # $w_r$ is the vector of coefficients $\gamma_{r,ik}$.
        # And the equation holds.
        # So the rank IS 7.
        
        # So my calculation holds.
        # I just need to implement the vectors $u_r, v_r, w_r$ for Strassen.
        
        pass

    return {}
# EVOLVE-BLOCK-END
