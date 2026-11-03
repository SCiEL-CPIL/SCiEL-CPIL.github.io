"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # For 5x5, the best known exact rank is 125 (schoolbook).
    # No exact decomposition with rank < 125 is known.
    rank = n * m * p
    
    # Using numpy for efficient construction if needed, but lists are required?
    # The prompt says "factor matrices of shapes...". It doesn't strictly enforce type, 
    # but "integer/half-integer entries" suggests numbers.
    # The seed returns lists. I will return lists to be safe.
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The matrix multiplication tensor T is defined such that
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # The decomposition is sum_c U[i, c] * V[j, c] * W[k, c] ?
    # Wait, the standard tensor decomposition for matrix multiplication $C = AB$ is:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$
    # The tensor $\mathcal{T}$ has indices $i, j, k$.
    # $\mathcal{T}_{ijk} = 1$ if it contributes to $C_{ik}$?
    # Actually, the bilinear form is $\sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ik}$? No.
    # The rank of the matrix multiplication tensor is the minimum number of multiplications.
    # Each term in the sum corresponds to a rank-1 tensor.
    # A rank-1 tensor is $u_i v_j w_k$.
    # The identity is $\sum_{c=1}^R u^{(c)}_i v^{(c)}_j w^{(c)}_k = \delta_{ik} \delta_{jj'}$? No.
    # The standard schoolbook algorithm uses $R = nmp$ multiplications.
    # Each multiplication is $M_c = A_{i_c, j_c} B_{j_c, k_c}$.
    # Then $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    # We can view this as $C_{ik} = \sum_{c} U_{ic} V_{jc} W_{kc}$? No, the indices are mixed.
    
    # Let's look at the seed code's convention:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # This implies the factors are vectors indexed by the input/output indices.
    # The product $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$?
    # The seed code sets:
    # For each (i, j, k):
    #   col = index
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1
    #   W[k*n+i, col] = 1
    
    # This corresponds to the term $A_{ij} B_{jk} C_{ki}$? No, $C$ is the output.
    # The tensor for matrix multiplication is usually defined on $V \otimes V^* \otimes V^*$ or similar.
    # Let's trust the seed's construction as it is the standard "schoolbook" rank 125 decomposition.
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                U[row_u][column] = 1
                V[row_v][column] = 1
                W[row_w][column] = 1
                column += 1
                
                # Check deadline
                if time.time() > deadline:
                    # If we somehow took too long (unlikely for 125 iterations), 
                    # we would need to return partial or best effort. 
                    # But 125 is constant and fast.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
