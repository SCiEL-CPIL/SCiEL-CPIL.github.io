def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    # Reconstruct C for all i, k
    # C[i, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: U is indexed by (i,j). V by (j,k). W by (k,i).
    # We need to sum over j and r.
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Compare with schoolbook:
    # C_school[i,k] = sum_j A[i,j] B[j,k]
    # This is not a fixed number, it's a tensor.
    # The tensor T[i,j,k] = 1 if we are looking at the term for (i,j,k)?
    # No, the decomposition is of the multiplication *map*.
    # The tensor is $M_{i,j,k}$.
    # $M_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # Wait, the seed code sets $U, V, W$ such that the product matches the schoolbook.
    # So I just need to check if my new $U, V, W$ produce the same tensor as the schoolbook $U, V, W$.
    
    # I can compute the tensor entries $T_{i,j,k}$ from the factors.
    # $T_{i,j,k} = \sum_r U_{i \cdot m + j, r} V_{j \cdot p + k, r} W_{k \cdot n + i, r}$
    # And check if it matches the standard multiplication tensor.
    # The standard tensor has $T_{i,j,k} = 1$ if $j$ is the middle index?
    # No, the tensor is defined by the identity:
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # The rank-1 decomposition is $C_{i,k} = \sum_r ( \sum_j A_{i,j} U_{i,j,r} ) ( \sum_j B_{j,k} V_{j,k,r} ) W_{k,i,r}$.
    # This is not a simple tensor product $U \otimes V \otimes W$.
    # The seed code implies:
    # $U_{i \cdot m + j, r}$ is the coefficient of $A_{i,j}$.
    # $V_{j \cdot p + k, r}$ is the coefficient of $B_{j,k}$.
    # $W_{k \cdot n + i, r}$ is the coefficient for $C_{i,k}$.
    
    # So the condition is:
    # For all $i, k$:
    # $\sum_{j} A_{i,j} B_{j,k} = \sum_r ( \sum_{j} A_{i,j} U_{i \cdot m + j, r} ) ( \sum_{j} B_{j,k} V_{j \cdot p + k, r} ) W_{k \cdot n + i, r}$
    
    # This must hold for ALL $A, B$.
    # This implies that the tensors formed by the coefficients must be equal.
    # Specifically, the coefficient of $A_{i,j} B_{j',k}$ in the RHS must be 1 if $j=j'$ and 0 otherwise?
    # No, the sum over $j$ in the first parenthesis and the sum over $j$ in the second parenthesis are independent?
    # No, they are separate sums.
    # Let $X_r = \sum_j A_{i,j} U_{i,j,r}$ and $Y_r = \sum_j B_{j,k} V_{j,k,r}$.
    # Then $C_{i,k} = \sum_r X_r Y_r W_{k,i,r}$.
    # $X_r$ is linear in $A$. $Y_r$ is linear in $B$.
    # The product $X_r Y_r$ is bilinear.
    # The coefficient of $A_{i,j} B_{j',k}$ in $X_r Y_r$ is $U_{i,j,r} V_{j',k,r}$.
    # So the total coefficient of $A_{i,j} B_{j',k}$ in $C_{i,k}$ is:
    # $\sum_r U_{i,j,r} V_{j',k,r} W_{k,i,r}$.
    # This must equal $\delta_{j,j'}$ (1 if $j=j'$, 0 otherwise).
    
    # So I can verify by checking:
    # For all $i, j, j', k$:
    # $\sum_r U[i \cdot m + j, r] V[j' \cdot p + k, r] W[k \cdot n + i, r] == (1 \text{ if } j == j' \text{ else } 0)$
    
    # This is $3 \cdot 3 \cdot 3 \cdot 6 = 162$ checks.
    # Each check sums over $R$ (46).
    # $162 \cdot 46 \approx 7452$ ops. Very fast.
