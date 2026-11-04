"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 9 matrix.
    
    The rank is 54 (schoolbook algorithm).
    U shape: (n*m, rank) = (6, 54)
    V shape: (m*p, rank) = (27, 54)
    W shape: (p*n, rank) = (18, 54)
    
    The decomposition satisfies:
    C_{i,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} ?
    Wait, the standard bilinear form is:
    C = sum_r U_r * V_r^T * W_r ?
    The interface specifies:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The multiplication is A (n x m) * B (m x p) = C (n x p).
    Entry C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    
    In the tensor decomposition T = sum_r u_r \otimes v_r \otimes w_r,
    the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    
    The matrices U, V, W are such that:
    For each r, the term contributes to C_{i,k} via:
    U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} ?
    
    Let's check the indices.
    Usually, the tensor is defined on indices (i,j), (j,k), (i,k).
    U has rows indexed by (i,j) from A.
    V has rows indexed by (j,k) from B.
    W has rows indexed by (i,k) from C.
    
    The condition for exact decomposition is:
    sum_r U_{(i,j), r} * V_{(j,k), r} * W_{(i,k), r} = delta_{i,k?} No.
    The product is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    So for a fixed (i,k), the coefficient of A_{i,j} B_{j,k} in the expansion of C_{i,k} is 1 if j is the summation index.
    Actually, the tensor T is defined such that T_{(i,j), (j,k), (i,k)} = 1 if the indices match the multiplication rule.
    Specifically, for each j, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    So for each triple (i, j, k), we have a term in the sum.
    The sum is over r.
    For a specific (i, j, k), we need:
    sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1 ?
    No, the bilinear form is:
    C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    This can be written as:
    C_{i,k} = sum_{j=0}^{m-1} sum_{r} (U_{i*m+j, r} A_{i,j}) (V_{j*p+k, r} B_{j,k}) W_{k*n+i, r} ?
    This doesn't look right.
    
    Standard format:
    C_{i,k} = sum_{r=1}^{R} W_{i,k,r} ( sum_{j} U_{i,j,r} A_{i,j} ) ( sum_{j} V_{j,k,r} B_{j,k} ) ?
    No, the inner products are over j.
    So for a fixed r, we compute z_r = (U_r^T A) (V_r^T B)? No.
    z_r = sum_{i,j} U_{i,j,r} A_{i,j} * sum_{j,k} V_{j,k,r} B_{j,k} ?
    No, the multiplication is bilinear in A and B.
    The term is $z_r = (\sum_{i,j} U_{i*m+j, r} A_{i,j}) \cdot (\sum_{j,k} V_{j*p+k, r} B_{j,k})$.
    Then $C_{i,k} = \sum_r W_{k*n+i, r} z_r$.
    
    For the schoolbook algorithm:
    Each term corresponds to a single (i, j, k) triple.
    Let r index the triple (i, j, k).
    $U_{i*m+j, r} = 1$ if r corresponds to (i,j,k) and the i,j match.
    $V_{j*p+k, r} = 1$ if r corresponds to (i,j,k) and the j,k match.
    $W_{k*n+i, r} = 1$ if r corresponds to (i,j,k) and the i,k match.
    
    Then $z_r = A_{i,j} B_{j,k}$.
    $C_{i',k'} = \sum_r W_{k'*n+i', r} A_{i,j} B_{j,k}$.
    $W_{k'*n+i', r}$ is 1 only if $i'=i$ and $k'=k$.
    So $C_{i',k'}$ gets $A_{i,j} B_{j,k}$ only if $i'=i, k'=k$.
    Summing over all r (which iterates over all i,j,k), we get:
    $C_{i',k'} = \sum_{j} A_{i',j} B_{j,k'}$.
    This is correct.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                
                if time.time() > deadline:
                    # Should not happen for fixed small size, but safety
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
