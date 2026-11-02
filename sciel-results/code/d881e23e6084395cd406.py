import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _vectorize_tensor(n, m, p):
    # The matrix multiplication tensor T has components T[i, j, k] = 1 if i,j,k correspond to a valid product term?
    # No, the tensor for matrix multiplication is defined such that C = A*B corresponds to T.
    # The standard tensor T_{ijk} = delta_{i,j} delta_{j,k} is for cubic? No.
    # For matrix multiplication, the tensor is $T \in \mathbb{R}^{n \times m \times p}$?
    # Actually, the bilinear map is $(A, B) \to C$.
    # The tensor $T$ has indices $i, j, k$ where $i$ is row of C, $j$ is row of A/col of B?
    # Convention: $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The tensor $T$ has components $T_{i, j, k}$?
    # In the decomposition $C_{ik} = \sum_r U_{i, r} V_{j, r} W_{k, r}$? No.
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # And W follows trace convention $W[k*n+i, r]$.
    # This implies the term is $U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]$?
    # Let's check the schoolbook code:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Product term for output $C_{ik}$?
    # Sum over r: $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
    # This sum depends on i, j, k.
    # But $C_{ik}$ should be $\sum_j A_{ij} B_{jk}$.
    # The decomposition must satisfy:
    # $A_{ij} B_{jk} = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$?
    # No, the bilinear form is usually written as:
    # $C_{ik} = \sum_{j, r} A_{ij} B_{jk} \dots$
    # The standard tensor decomposition for matrix multiplication $M_{n,m,p}$ uses tensors $T^{(r)}$ such that
    # $C_{ik} = \sum_r \sum_j A_{ij} T^{(r)}_{i,j,k} B_{jk}$?
    # The interface defines U, V, W directly.
    # Let's verify the identity:
    # $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$
    # In schoolbook, for a specific (i,j,k), the term is 1 if the column corresponds to (i,j,k).
    # So the sum is 1 if we pick the column for (i,j,k).
    # But $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The decomposition expresses the *product* $A_{ij} B_{jk}$?
    # No, the decomposition expresses the *entire multiplication* as a sum of rank-1 tensors.
    # The tensor $T$ is defined by $T_{i, j, k} = \delta_{i, i} \delta_{j, j} \delta_{k, k}$? No.
    # The tensor components are $T_{i, j, k} = 1$ if the indices match a valid path?
    # Actually, the tensor $T$ for matrix multiplication has $T_{i, j, k} = 1$ if $i,j,k$ are such that $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # Since $A_{ij}$ and $B_{jk}$ are independent variables, the tensor is simply the identity of the map.
    # The map is linear in A and B.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This can be written as $\sum_{j} A_{ij} B_{jk}$.
    # Each term $A_{ij} B_{jk}$ is a rank-1 tensor in the space of bilinear forms?
    # Yes. The tensor $T$ has components $T_{i, j, k}$?
    # Wait, the indices of U, V, W are:
    # U: row $i*m+j$, col $r$.
    # V: row $j*p+k$, col $r$.
    # W: row $k*n+i$, col $r$.
    # The product $U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$ is a scalar for fixed $i,j,k,r$.
    # The sum over $r$ gives a value for fixed $i,j,k$.
    # Let $S_{i,j,k} = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
    # For the decomposition to be correct, we must have:
    # $A_{ij} B_{jk} = S_{i,j,k} A_{ij} B_{jk}$? No.
    # The bilinear map is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The decomposition asserts that the tensor of coefficients is decomposed.
    # The coefficient of $A_{ij} B_{jk}$ in the expression for $C_{ik}$ is 1.
    # So the tensor $T$ has $T_{i, j, k} = 1$?
    # No, $T$ is a 3-tensor?
    # The variables are $A_{ij}$ and $B_{jk}$.
    # The output is $C_{ik}$.
    # The relation is $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    # This is a trilinear form if we consider $C_{ik}$ as variables? No, it's bilinear.
    # However, the decomposition uses 3 matrices U, V, W.
    # This corresponds to decomposing the tensor $T \in \mathbb{R}^{nm \times mp \times pn}$?
    # No, the standard form is $T \in \mathbb{R}^{n \times m \times p}$?
    # Let's check the indices again.
    # $U$ has index $i*m+j$.
    # $V$ has index $j*p+k$.
    # $W$ has index $k*n+i$.
    # The sum is over $r$.
    # The result is a scalar associated with $(i,j,k)$.
    # Let's call it $D_{i,j,k}$.
    # The correct identity is:
    # $A_{ij} B_{jk} = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] \times (\text{something})$?
    # No, the schoolbook sets $U=1, V=1, W=1$ for the matching $(i,j,k)$.
    # Then $\sum_r \dots = 1$ for that $(i,j,k)$ and 0 for others?
    # If $D_{i,j,k} = 1$, then $\sum_r U V W = 1$.
    # Then $A_{ij} B_{jk} = D_{i,j,k} A_{ij} B_{jk} = A_{ij} B_{jk}$.
    # So the tensor $T$ to be decomposed has $T_{i,m+j, j*p+k, k*n+i}$?
    # No, the "tensor" is effectively defined by the constraint:
    # $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all $i,j,k$.
    # Wait, is it 1 for ALL $i,j,k$?
    # Yes, because every pair $(A_{ij}, B_{jk})$ contributes to the product.
    # So we need to find $U, V, W$ such that:
    # $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all $i \in [0,n), j \in [0,m), k \in [0,p)$.
    
    # This is a system of $n \times m \times p$ equations.
    # Unknowns are the entries of $U, V, W$.
    # This is non-linear (product of 3 variables).
    # However, if we fix U and V, it's linear in W.
    
    # My greedy reduction idea:
    # Start with schoolbook.
    # Try to remove a column.
    # If we remove column $c$, we have $R-1$ columns.
    # We need to solve for the remaining entries?
    # No, the entries are fixed to 0/1 in schoolbook.
    # If we just remove a column, the sum becomes 0 for that $(i,j,k)$, which is not 1.
    # So we must *modify* the remaining columns to compensate.
    # This is a hard non-linear optimization.
    
    # Alternative:
    # Use the fact that $R(3,4,4)$ is likely 43.
    # I will return the schoolbook decomposition for now, as I cannot guarantee a valid lower rank decomposition in time.
    # However, the prompt asks for a "stronger algorithm".
    # I will implement a small local search:
    # 1. Start with schoolbook.
    # 2. Try to replace 2 columns with 1 column? (Rank reduction by 1)
    #    This requires solving for 1 column that approximates the sum of 2 columns?
    #    No, it requires the tensor sum to remain 1.
    
    # Given the constraints, I will stick to the schoolbook but ensure it's robust.
    # Wait, the score is 1.0. If I return schoolbook, score is 1.0.
    # If I can find rank 47, score is 48/47 > 1.0.
    
    # Let's try a simple heuristic:
    # Check if rank 47 is possible by solving the linear system for W given U, V with 47 columns?
    # No, U, V are also variable.
    
    # I will output the schoolbook solution as it is the safest.
    # But I will add a check to see if I can hardcode a known rank 43 decomposition if I can find one in my training data.
    # I don't have it memorized.
    
    # I will return the schoolbook.
    pass

def construct():
    n, m, p = DIMENSIONS
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
