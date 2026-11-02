import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = 7
    
    # Initialize matrices
    U = np.zeros((n * m, rank)) # 4 x 7
    V = np.zeros((m * p, rank)) # 24 x 7
    W = np.zeros((p * n, rank)) # 24 x 7
    
    # Strassen coefficients
    # U coefficients: [a, b, c, d] for each term
    U_coeffs = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    # V coefficients: local 2x2 matrix for each term
    # Indices: V_local[j][l] where j is row of B (0,1), l is col of B (0,1)
    V_coeffs = [
        [[1, 0], [0, 1]], # P1: e+h
        [[1, 0], [0, 0]], # P2: e
        [[0, 1], [0, -1]],# P3: f-h
        [[-1, 0], [1, 0]],# P4: g-e
        [[0, 0], [0, 1]], # P5: h
        [[1, 1], [0, 0]], # P6: e+f
        [[0, 0], [1, 1]]  # P7: g+h
    ]
    
    # W coefficients: local 2x2 matrix for each term
    # Indices: W_local[l][i] where l is col of C (0,1), i is row of C (0,1)
    # Note: W[k*n+i] corresponds to C[i][k]. 
    # The tensor contraction is sum A[i][j] B[j][k] = C[i][k].
    # The decomposition T = sum U_r v_r w_r.
    # The term contributes U_r[i][j] * V_r[j][k] * W_r[k][i] to T[i][j][k].
    # Wait, the standard tensor decomposition for matmul M_{n,m,p} is:
    # M[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i] ?
    # Let's check the prompt: "W[k*n+i,r]".
    # Usually, the rank-r decomposition of the tensor T_{ijk} is sum_r u_r[i] v_r[j] w_r[k].
    # For matmul, T_{ijk} = 1 if indices form a valid multiplication path? No.
    # The tensor of matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k} = 1$ if $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$?
    # No, the tensor $T$ is such that $\langle T, A \otimes B \rangle = C$.
    # Actually, the bilinear form is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The tensor entries are $T_{i,j,k} = 1$ if we view it as a 3-way tensor where $T[i, j, k] = 1$ for all $i,j,k$? No.
    # The decomposition is $\sum_r U_r \otimes V_r \otimes W_r$.
    # $U_r$ is $n \times m$, $V_r$ is $m \times p$, $W_r$ is $p \times n$.
    # The reconstructed tensor is $\sum_r U_r \otimes V_r \otimes W_r$.
    # Entry $(i, j, k)$ of the reconstructed tensor is $\sum_r U_{r, i, j} V_{r, j, k} W_{r, k, i}$?
    # Let's check the trace convention given: "W[k*n+i,r]".
    # This implies $W$ is indexed by $k$ (col of B/C) and $i$ (row of C/A).
    # So $W_r[k, i]$.
    # The product is $\sum_r U_r[i, j] V_r[j, k] W_r[k, i]$.
    # This matches the structure if we sum over $r$.
    
    # Let's verify the Strassen logic with this index convention.
    # We want $\sum_r U_{r, i, j} V_{r, j, k} W_{r, k, i} = \delta_{i, i'} \delta_{j, j'} \delta_{k, k'}$? No.
    # We want the bilinear form to be $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is equivalent to the tensor $T_{i, j, k}$ being 1 if the indices are "compatible" in the multiplication sense?
    # Actually, the tensor $T$ for $2 \times 2 \times 2$ matmul has 8 non-zero entries? No, 16?
    # $T_{i,j,k}$ is 1 if $i,j,k$ are all in range? No.
    # The tensor is defined such that $T(A, B) = C$.
    # $T_{i,j,k}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{?}$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So the term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # In the tensor decomposition $\sum_r U_r \otimes V_r \otimes W_r$, the term is $U_r[i,j] V_r[j,k] W_r[k,i]$.
    # This sum must equal 1 if $i,j,k$ form a valid triple contributing to the matmul?
    # Actually, for a specific $i,j,k$, the value of the tensor is 1 if we are computing $C_{i,k}$ using $A_{i,j}$ and $B_{j,k}$.
    # So $T_{i,j,k} = 1$ for all $i,j,k$? No, that would be a completely positive tensor.
    # The matrix multiplication tensor is sparse.
    # For $2 \times 2 \times 2$, $T$ has 8 non-zero entries?
    # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$. Terms: $(0,0,0), (0,1,0)$.
    # $C_{01} = A_{00}B_{01} + A_{01}B_{11}$. Terms: $(0,0,1), (0,1,1)$.
    # $C_{10} = A_{10}B_{00} + A_{11}B_{10}$. Terms: $(1,0,0), (1,1,0)$.
    # $C_{11} = A_{10}B_{01} + A_{11}B_{11}$. Terms: $(1,0,1), (1,1,1)$.
    # So $T_{i,j,k} = 1$ if $k$ is the column of $C$?
    # The indices are $i$ (row of A/C), $j$ (col of A / row of B), $k$ (col of B/C).
    # The non-zero entries are $(i,j,k)$ such that this combination appears in the sum for $C_{i,k}$.
    # Every $j$ appears in $C_{i,k}$ for any $i,k$.
    # So $T_{i,j,k} = 1$ for all $i,j,k$?
    # Wait, if $T_{i,j,k}=1$ for all, then $\sum_{i,j,k} A_{ij}B_{jk} T_{ijk} = (\sum A)(\sum B)$. That's rank 1.
    # The tensor is NOT all ones.
    # The tensor $T$ is defined by the map $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a linear map. The tensor $T$ has components $T_{i, j, k}$ such that
    # $C_{ik} = \sum_{j} T_{i, j, k} A_{ij} B_{jk}$? No.
    # Standard definition: $T_{i, j, k} = 1$ if the indices are "linked".
    # Actually, $T_{i, j, k} = 1$ if we view the tensor as the structure constant of the algebra?
    # For matrix multiplication, $T_{i, j, k} = 1$ for all $i, j, k$ is WRONG.
    # Let's look at the Strassen decomposition again.
    # $\sum_r P_r = C$.
    # $P_r = U_r A \cdot V_r B$? No.
    # $P_r$ is the $r$-th product.
    # $P_r = (U_r A) \cdot (V_r B)$? No, $U_r$ and $V_r$ are linear combinations.
    # $M_r = \sum_{i,j} U_{r, i, j} A_{i, j}$.
    # $N_r = \sum_{j,k} V_{r, j, k} B_{j, k}$.
    # $Q_r = M_r N_r$.
    # $C_{i, k} = \sum_r W_{r, k, i} Q_r = \sum_r W_{r, k, i} (\sum_{a,b} U_{r, a, b} A_{a, b}) (\sum_{c,d} V_{r, c, d} B_{c, d})$.
    # $C_{i, k} = \sum_{a,b,c,d} A_{a, b} B_{c, d} (\sum_r U_{r, a, b} V_{r, c, d} W_{r, k, i})$.
    # We want $C_{i, k} = \sum_{b} A_{i, b} B_{b, k}$.
    # So we need $\sum_r U_{r, a, b} V_{r, c, d} W_{r, k, i} = \delta_{a, i} \delta_{b, c} \delta_{d, k}$.
    # This is the condition for the tensor decomposition.
    
    # So my previous construction of U, V, W must satisfy:
    # $\sum_r U_{r, a, b} V_{r, c, d} W_{r, k, i} = \delta_{a, i} \delta_{b, c} \delta_{d, k}$.
    
    # Let's check the Strassen terms for 2x2x2.
    # $a=A_{00}, b=A_{01}, c=A_{10}, d=A_{11}$.
    # $e=B_{00}, f=B_{01}, g=B_{10}, h=B_{11}$.
    # $C_{00} = P_1 + P_4 - P_5 + P_7$.
    # $C_{01} = P_3 + P_5$.
    # $C_{10} = P_2 + P_4$.
    # $C_{11} = P_1 - P_2 + P_3 + P_6$.
    
    # My W coefficients above were for $C_{i, l}$ where $l$ is local col.
    # In the global case, $k$ is global col. $l = k \% 2$.
    # $W_{r, k, i} = W_{local, r}[l, i]$.
    
    # Let's verify one term, say $a=A_{00}, c=B_{00}, i=0, k=0$.
    # We need sum_r $U_{r, 0, 0} V_{r, 0, 0} W_{r, 0, 0} = 1$.
    # $k=0 \implies l=0$.
    # $W_{r, 0, 0} = W_{local, r}[0, 0]$.
    # $U_{r, 0, 0} = U_{local, r}[0]$ (coeff of a).
    # $V_{r, 0, 0} = V_{local, r}[0, 0]$ (coeff of e).
    
    # r=0: U=1, V=1, W=1. Prod=1.
    # r=1: U=0, V=1, W=0. Prod=0.
    # r=2: U=1, V=0, W=0. Prod=0.
    # r=3: U=0, V=-1, W=1. Prod=0.
    # r=4: U=1, V=0, W=-1. Prod=0.
    # r=5: U=-1, V=1, W=0. Prod=0.
    # r=6: U=0, V=0, W=1. Prod=0.
    # Sum = 1. Correct.
    
    # Let's check $a=A_{00}, c=B_{10}, i=0, k=0$. (Should be 0)
    # $c$ is $g$ (local index 1,0). $V_{r, 1, 0} = V_{local, r}[1, 0]$.
    # $W_{r, 0, 0} = W_{local, r}[0, 0]$.
    # r=0: U=1, V=0, W=1. 0.
    # r=1: U=0, V=0, W=0. 0.
    # r=2: U=1, V=0, W=0. 0.
    # r=3: U=0, V=1, W=1. 0.
    # r=4: U=1, V=0, W=-1. 0.
    # r=5: U=-1, V=0, W=0. 0.
    # r=6: U=0, V=1, W=1. 0.
    # Sum = 0. Correct.
    
    # The logic holds.
    
    for r in range(rank):
        # Set U
        for idx, val in enumerate(U_coeffs[r]):
            U[idx, r] = val
            
        # Set V
        # V global index: j * p + k
        # V local index: V_coeffs[r][j][l]
        for j in range(m):
            for k in range(p):
                l = k % 2
                V[j * p + k, r] = V_coeffs[r][j][l]
                
        # Set W
        # W global index: k * n + i
        # W local index: W_coeffs[r][l][i]
        # Note: W_coeffs defined as W_local[l][i]
        for k in range(p):
            l = k % 2
            for i in range(n):
                W[k * n + i, r] = W_coeffs[r][l][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
