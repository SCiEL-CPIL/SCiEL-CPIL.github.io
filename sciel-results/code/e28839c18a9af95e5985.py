import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target: Find rank < n*m*p = 63
    # Schoolbook rank is 63.
    # We will try to find a lower rank using ALS and rounding.
    
    # Build the target tensor T[i, j, k] = delta_{i,k} for the multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T[i, j, k] = 1 if i==k else 0? No.
    # The multiplication tensor T has T[i, j, k] = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T[i, j, k] = 1 if i is the row of C, j is the inner index, k is the col of C.
    # Specifically, T_{i,j,k} = 1 if the output index is (i,k) and input indices are (i,j) and (j,k).
    # This is always 1 for valid i,j,k. Wait.
    # The tensor for matrix multiplication $C = AB$ is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # $T = \sum_{i,j,k} (e_i \otimes e_j \otimes e_k)$ is the identity? No.
    # The tensor is $T_{i,j,k} = \delta_{i,k}$? No.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This corresponds to the tensor $T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
    # In the standard basis, the tensor has entries $T[i, j, k] = 1$ if the multiplication $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
    # Since every $A_{i,j}$ and $B_{j,k}$ contributes to exactly one $C_{i,k}$, the tensor is simply:
    # T[i, j, k] = 1 for all i, j, k? No, that would be the all-ones tensor.
    # The tensor is defined by the bilinear map.
    # $T(u, v) = u^T (I \otimes I) v$?
    # Let's construct the tensor T of shape (n, m, p) such that $C_{i,k} = \sum_{j} T_{i,j,k} A_{i,j} B_{j,k}$? No.
    # The standard definition: $T = \sum_{i=0}^{n-1} \sum_{j=0}^{m-1} \sum_{k=0}^{p-1} e_i \otimes e_j \otimes e_k$?
    # No, the tensor for $3 \times 3 \times 7$ multiplication is:
    # $T = \sum_{i=0}^{2} \sum_{j=0}^{2} \sum_{k=0}^{6} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
    # Actually, the tensor is just the set of trilinear forms.
    # $T[i, j, k] = 1$ if the term $A_{i,j} B_{j,k}$ is part of the sum for $C_{i,k}$.
    # Yes, for standard matrix multiplication, $T[i, j, k] = 1$ for all $i, j, k$ is INCORRECT.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # The tensor $T$ has components $T_{i,j,k}$ such that $C_{i,k} = \sum_{j} A_{i,j} B_{j,k} T_{i,j,k}$? No.
    # The tensor is $T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
    # Let's use the unfolded matrices.
    # $T_{(1)}$ is $n \times (m \cdot p)$.
    # $T_{(2)}$ is $m \times (n \cdot p)$.
    # $T_{(3)}$ is $p \times (n \cdot m)$.
    # For $C_{i,k} = \sum_j A_{i,j} B_{j,k}$:
    # The term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
    # So $T_{(3)}[k, i \cdot m + j] = 1$.
    # $T_{(1)}[i, j \cdot p + k] = 1$.
    # $T_{(2)}[j, k \cdot n + i] = 1$.
    # This is the schoolbook structure.
    
    # We want to find U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # sum_c U[:, c] V[:, c]^T ... no, it's a tensor contraction.
    # T = sum_c u_c \otimes v_c \otimes w_c
    # This is equivalent to:
    # T_{(1)} = U V^T ? No.
    # T_{(1)} = U (V \otimes W)^T ? No.
    # The condition is:
    # For all i, j, k: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's build the target tensor T_flat of size n*m*p.
    # T[i, j, k] = 1 if i is row, j is inner, k is col.
    # Actually, T[i, j, k] is 1 for all valid i, j, k?
    # No, $C_{i,k}$ depends on $A_{i,j}$ and $B_{j,k}$.
    # The tensor is the identity in the appropriate sense.
    # T[i, j, k] = 1 for all i, j, k is the tensor for the sum of all products, which is not matrix multiplication.
    # Matrix multiplication tensor:
    # T[i, j, k] = 1 if the indices match the multiplication pattern.
    # The pattern is: A is (n, m), B is (m, p), C is (n, p).
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # The tensor $T$ has $T[i, j, k] = 1$?
    # Let's check the schoolbook code:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Product: 1*1*1 = 1.
    # Sum over c: For a fixed (i, j, k), there is exactly one c (the one where i,j,k match) that contributes 1.
    # So T[i, j, k] = 1 for all i, j, k?
    # Wait. If T[i, j, k] = 1 for all i, j, k, then the tensor is the all-ones tensor.
    # But the all-ones tensor corresponds to $C_{i,k} = (\sum_i A_{i,j}) (\sum_k B_{j,k})$? No.
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # Let's re-read the schoolbook loop.
    # It iterates i, j, k.
    # For each (i, j, k), it creates a rank-1 term with 1s at the specific positions.
    # The sum of these rank-1 terms gives 1 at position (i, j, k) in the tensor space.
    # So the target tensor T has T[i, j, k] = 1 for all i, j, k.
    # Is this correct?
    # If T[i, j, k] = 1 for all i, j, k, then $C_{i,k} = \sum_{j} \sum_{c} U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
    # If we use the schoolbook, $C_{i,k} = \sum_{j} 1 \cdot 1 \cdot 1 = m$.
    # This is NOT matrix multiplication.
    # Matrix multiplication is $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # The tensor must be such that the bilinear form yields this.
    # The tensor is $T_{i,j,k} = \delta_{i,k}$? No.
    # The tensor is defined by the mapping $(A, B) \to C$.
    # The rank-1 decomposition is $C_{i,k} = \sum_c (U_{i,j,c} A_{i,j}) (V_{j,k,c} B_{j,k}) W_{k,i,c}$?
    # No, the decomposition is:
    # $C_{i,k} = \sum_c W_{k,n+i,c} ( \sum_j U_{i,m+j,c} A_{i,j} ) ( \sum_j V_{j,p+k,c} B_{j,k} )$?
    # No, the standard definition is:
    # $T = \sum_c u_c \otimes v_c \otimes w_c$
    # where $u_c \in \mathbb{R}^{n \cdot m}$, $v_c \in \mathbb{R}^{m \cdot p}$, $w_c \in \mathbb{R}^{p \cdot n}$.
    # And $T[i, j, k] = \sum_c u_c[i \cdot m + j] v_c[j \cdot p + k] w_c[k \cdot n + i]$.
    # For matrix multiplication, $T[i, j, k] = 1$ if the term $A_{i,j} B_{j,k}$ is used in $C_{i,k}$.
    # Since $C_{i,k} = \sum_j A_{i,j} B_{j,k}$, the term $A_{i,j} B_{j,k}$ is used for all i, j, k.
    # So $T[i, j, k] = 1$ for all i, j, k?
    # If so, the schoolbook gives $\sum_j 1 = m$ for each $C_{i,k}$?
    # No, the schoolbook creates SEPARATE terms for each (i, j, k).
    # Term c_{i,j,k} has $u_{i,j}=1, v_{j,k}=1, w_{k,i}=1$.
    # Then $\sum_{i',j',k'} T[i',j',k'] = \sum_{i',j',k'} \sum_c u_c[i',j'] v_c[j',k'] w_c[k',i']$.
    # For a specific (i, j, k), only the term $c_{i,j,k}$ contributes 1.
    # So $T[i, j, k] = 1$.
    # So the target tensor IS the all-ones tensor?
    # If T is all-ones, then $C_{i,k} = \sum_j \sum_c u_c[i,j] v_c[j,k] w_c[k,i]$.
    # With schoolbook, $C_{i,k} = \sum_j 1 = 3$.
    # This is wrong. $C_{i,k}$ should be $\sum_j A_{i,j} B_{j,k}$.
    # The error is in the interpretation of the tensor indices.
    # The tensor $T$ represents the bilinear map.
    # $T(A, B) = C$.
    # The components are $T_{i,j,k}$ such that $C_{i,k} = \sum_{j} T_{i,j,k} A_{i,j} B_{j,k}$?
    # No, the indices of T are (index of A, index of B, index of C).
    # $T[a, b, c] = 1$ if $A_a B_b$ contributes to $C_c$.
    # Here $a = (i, j)$, $b = (j, k)$, $c = (i, k)$.
    # So $T[(i,j), (j,k), (i,k)] = 1$.
    # This is 1 for all valid i, j, k.
    # So the tensor IS the all-ones tensor in the subspace of valid indices?
    # Yes.
    # So why does schoolbook work?
    # Because it decomposes the all-ones tensor into rank-1 terms.
    # Rank of all-ones tensor of size $n \cdot m \times m \cdot p \times p \cdot n$?
    # No, the tensor is sparse in the sense that it's defined on the manifold of valid indices.
    # But mathematically, it's the tensor with $T[i,j,k] = 1$.
    # The rank of the all-ones tensor $1 \in \mathbb{R}^{A \times B \times C}$ is 1.
    # $1 = u \otimes v \otimes w$ with $u, v, w$ all-ones vectors.
    # If we use rank 1:
    # $U[i,j] = 1, V[j,k] = 1, W[k,i] = 1$.
    # Then $C_{i,k} = \sum_j 1 \cdot 1 \cdot 1 = m = 3$.
    # This computes $3 \times C_{i,k}$? No, it computes a constant 3.
    # It does NOT compute the matrix product.
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor is $T_{(i,j), (j,k), (i,k)} = 1$.
    # This is a specific tensor.
    # Let's check the schoolbook again.
    # It uses $n \cdot m \cdot p$ terms.
    # Each term is localized.
    # This implies the tensor is NOT rank 1.
    # Why? Because the indices are coupled.
    # $T_{a,b,c} = 1$ if $a=(i,j), b=(j,k), c=(i,k)$.
    # This is a "permutation-like" tensor, not all-ones.
    # For a fixed (i,j), the non-zero entries in b are only those with the same j.
    # So it's block diagonal in a sense.
    # The rank is indeed $n \cdot m \cdot p$ for the schoolbook.
    
    # To beat 63, we need a better decomposition of this specific tensor.
    # We will use a local search to find a lower rank.
    
    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)?
    # No, we need the flat tensor T_flat of size (n*m) * (m*p) * (p*n)?
    # No, the tensor is defined on the indices (i, j, k).
    # We can represent it as a matrix of size (n*m) x (m*p) for each k? No.
    # We will use the unfolded matrices for ALS.
    # T_1: (n*m) x (m*p * p*n)
    # T_2: (m*p) x (n*m * p*n)
    # T_3: (p*n) x (n*m * m*p)
    
    # For each (i, j, k), T[i, j, k] = 1.
    # Index in T_1: row = i*m + j.
    # Col = (j*p + k) * (p*n) + (k*n + i).
    # Set T_1[row, col] = 1.
    
    T_1 = np.zeros((n*m, (m*p)*(p*n)))
    T_2 = np.zeros((m*p, (n*m)*(p*n)))
    T_3 = np.zeros((p*n, (n*m)*(m*p)))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                col_1 = idx_v * (p*n) + idx_w
                T_1[idx_u, col_1] = 1
                
                col_2 = idx_u * (p*n) + idx_w
                T_2[idx_v, col_2] = 1
                
                col_3 = idx_u * (m*p) + idx_v
                T_3[idx_w, col_3] = 1

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try to find a lower rank
    # Start from 62 down to 50
    for rank in range(best_rank - 1, 40, -1):
        # Initialize U, V, W randomly
        # Use a few restarts
        for restart in range(10):
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS steps
            for step in range(50):
                # Solve for U
                # T_1 = U (V \otimes W)^T ?
                # T_1[u, v*w] = sum_c U[u, c] V[v, c] W[w, c]
                # This is not a simple matrix product.
                # We can vectorize:
                # For each u, T_1[u, :] = sum_c U[u, c] (V[:, c] * W[:, c])
                # Let X = V * W (element-wise product), shape (m*p, p*n) x rank?
                # No, V is (m*p, rank), W is (p*n, rank).
                # V * W is not defined directly.
                # We need to form the matrix M_vw of size (m*p * p*n) x rank?
                # M_vw[ (v, w), c ] = V[v, c] * W[w, c].
                # This is the Khatri-Rao product of V and W.
                # M_vw = np.kron(V, W) ? No, Khatri-Rao is column-wise Kronecker.
                # M_vw[:, c] = V[:, c] \otimes W[:, c].
                
                M_vw = np.zeros(((m*p)*(p*n), rank))
                for c in range(rank):
                    M_vw[:, c] = np.kron(V[:, c], W[:, c])
                
                # T_1 = U M_vw^T
                # U = T_1 M_vw (M_vw^T M_vw)^{-1}
                try:
                    U = np.linalg.lstsq(M_vw, T_1, rcond=None)[0].T
                except:
                    break
                
                # Solve for V
                M_uw = np.zeros(((n*m)*(p*n), rank))
                for c in range(rank):
                    M_uw[:, c] = np.kron(U[:, c], W[:, c])
                try:
                    V = np.linalg.lstsq(M_uw, T_2, rcond=None)[0].T
                except:
                    break
                    
                # Solve for W
                M_uv = np.zeros(((n*m)*(m*p), rank))
                for c in range(rank):
                    M_uv[:, c] = np.kron(U[:, c], V[:, c])
                try:
                    W = np.linalg.lstsq(M_uv, T_3, rcond=None)[0].T
                except:
                    break
                    
            # Check error
            # Reconstruct T_1
            M_vw = np.zeros(((m*p)*(p*n), rank))
            for c in range(rank):
                M_vw[:, c] = np.kron(V[:, c], W[:, c])
            T_1_rec = U @ M_vw.T
            err = np.linalg.norm(T_1 - T_1_rec)
            
            if err < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                M_vw_r = np.zeros(((m*p)*(p*n), rank))
                for c in range(rank):
                    M_vw_r[:, c] = np.kron(V_rounded[:, c], W_rounded[:, c])
                T_1_rec_r = U_rounded @ M_vw_r.T
                err_r = np.linalg.norm(T_1 - T_1_rec_r)
                
                if err_r < 1e-6:
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
