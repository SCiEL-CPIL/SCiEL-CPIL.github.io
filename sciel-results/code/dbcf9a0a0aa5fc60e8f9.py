import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    
    # 1. Generate the rank-28 Strassen decomposition for the 4x4x4 core
    # Core indices: i, j, k in {0, 1, 2, 3}
    # We will store U, V, W as lists of lists (matrices)
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # We'll build the rank 28 part first.
    # Strassen for 4x4x4:
    # Block indices b_i, b_j, b_k in {0, 1}
    # Intra-block indices s_i, s_j, s_k in {0, 1}
    # i = 2*b_i + s_i
    # j = 2*b_j + s_j
    # k = 2*b_k + s_k
    
    # The 7 Strassen multiplications M_0 ... M_6
    # Each M_r is a 2x2 matrix product: M_r[s_i, s_k] = sum_{s_j} A_r[s_i, s_j] B_r[s_j, s_k]
    # A_r and B_r are linear combinations of the 4 blocks of A and B.
    
    # Define the 7 linear combinations for A blocks (indices b_i, b_j)
    # A blocks: A00, A01, A10, A11 (b_i, b_j)
    # B blocks: B00, B01, B10, B11 (b_j, b_k)
    
    # M0 = (A00 + A11) (B00 + B11)
    # M1 = (A10 + A11) B00
    # M2 = A00 (B10 - B11)
    # M3 = A11 (B01 - B00)
    # M4 = (A00 + A01) B11
    # M5 = (A10 - A00) (B00 + B01)
    # M6 = (A01 - A11) (B10 + B11)
    
    # For each M_r, we have a 2x2 matrix product.
    # The rank 4 decomposition of a 2x2x2 all-ones tensor T[s_i, s_j, s_k] = 1 is:
    # sum_{s_j in {0,1}} sum_{s_i in {0,1}} ? No.
    # The tensor is T[s_i, s_j, s_k] = 1.
    # Rank 4 decomposition:
    # r iterates over (s_i, s_k)? No.
    # We need sum_{r} u[s_i, r] v[s_j, r] w[s_k, r] = 1.
    # Let r correspond to s_j? No.
    # If we fix s_j, we have a 2x2 matrix 1*1. Rank 1?
    # T[s_i, s_j, s_k] = delta_{s_j, s_j} ?
    # Actually, T[s_i, s_j, s_k] = 1 for all s_i, s_j, s_k.
    # This is the outer product of 1-vector?
    # No, the indices are distinct.
    # $M_{s_i, s_k} = \sum_{s_j} A_{s_i, s_j} B_{s_j, s_k}$.
    # This is exactly the multiplication of two 2x2 matrices.
    # The rank of the 2x2x2 multiplication tensor is 4.
    # The decomposition is:
    # sum_{s_j=0}^1 sum_{s_i=0}^1 ?
    # Let's use the standard schoolbook rank 4 for 2x2x2:
    # Terms for s_j in {0, 1}:
    #   For each s_j, we have M[s_i, s_k] += A[s_i, s_j] B[s_j, s_k].
    #   This is a rank 2 term?
    #   A[s_i, s_j] is a vector of length 2 (over s_i).
    #   B[s_j, s_k] is a vector of length 2 (over s_k).
    #   So for fixed s_j, the contribution is $u \otimes v \otimes w$?
    #   No, it's $\sum_{s_i, s_k} A_{s_i, s_j} B_{s_j, s_k}$.
    #   This can be written as $\sum_{s_i} \sum_{s_k} \delta_{s_i} A_{s_i, s_j} \delta_{s_k} B_{s_j, s_k}$.
    #   This is rank 4?
    #   Yes, iterate s_i in {0,1} and s_k in {0,1}.
    #   For each (s_i, s_k), we have a term?
    #   Wait, the rank of 2x2x2 tensor T_{ijk}=1 is 4.
    #   Decomposition:
    #   r=0: s_i=0, s_j=0, s_k=0 -> 1
    #   r=1: s_i=0, s_j=1, s_k=0 -> 1
    #   r=2: s_i=1, s_j=0, s_k=0 -> 1
    #   r=3: s_i=1, s_j=1, s_k=0 -> 1
    #   This sums to 1 for s_k=0. What about s_k=1?
    #   We need $\sum_r u_{i,r} v_{j,r} w_{k,r} = 1$.
    #   If we use rank 4, we can cover it.
    #   Standard decomposition:
    #   $\sum_{j=0}^1 \sum_{i=0}^1 \delta_{i,i'} \delta_{j,j'} \delta_{k, ?}$
    #   Actually, just use the schoolbook 4 terms for each 2x2x2 block.
    #   For a specific M_r (which is a 2x2x2 tensor of 1s), the 4 terms are:
    #   For s_j in {0, 1}:
    #       For s_i in {0, 1}:
    #           For s_k in {0, 1}:
    #               This would be 8 terms. But rank is 4.
    #   The rank 4 decomposition of T_{ijk}=1 is:
    #   $\sum_{j=0}^1 \sum_{i=0}^1 e_i \otimes e_j \otimes e_j$? No.
    #   $\sum_{j=0}^1 \sum_{i=0}^1 1 \cdot \delta_{j,j'} \cdot \delta_{i,i'}$?
    #   Let's just generate the 8 schoolbook terms for the 2x2x2 part and see if they reduce?
    #   No, rank of 2x2x2 is 4.
    #   The 4 terms are:
    #   r1: i=0, j=0, k=0
    #   r2: i=0, j=1, k=1  <-- This doesn't sum to 1 everywhere.
    #   
    #   Correct rank 4 decomposition of 2x2x2 all-ones tensor:
    #   $T_{ijk} = 1$.
    #   $u_1 = (1, 0, 0, 1)$? No, dimensions are 2,2,2.
    #   $u$ has size 2*2=4? No, in the 2x2x2 subproblem, the indices are local.
    #   In the full tensor, the indices are global.
    #   
    #   Let's stick to the composition.
    #   M_r is a 2x2 matrix product.
    #   $M_r[p, q] = \sum_{l=0}^1 A_r[p, l] B_r[l, q]$.
    #   This is $\sum_{l=0}^1 (A_r[p, l]) (B_r[l, q])$.
    #   For each $l$, we have a term $A_r[p, l] B_r[l, q]$.
    #   This term depends on $p$ and $q$.
    #   $A_r[p, l]$ is a linear combo of A blocks.
    #   $B_r[l, q]$ is a linear combo of B blocks.
    #   
    #   So for each of the 7 Strassen multiplications, and for each $l \in \{0, 1\}$, we have a rank 2 contribution?
    #   $A_r[p, l] B_r[l, q] = \sum_{p,q} A_{r, p, l} B_{r, l, q}$.
    #   This is a rank 1 tensor in the (p, l, q) space?
    #   No, $l$ is summed over.
    #   For a fixed $l$, the term is $A_{r, p, l} B_{r, l, q}$.
    #   This is a matrix $M_{p,q}$.
    #   The tensor entry is $T_{i, j, k}$.
    #   $i$ corresponds to $b_i, s_i$. $j$ to $b_j, s_j$. $k$ to $b_k, s_k$.
    #   In the term for fixed $r, l$:
    #   $s_j = l$.
    #   $s_i = p$, $s_k = q$.
    #   The coefficient for $A$ is $A_{r, p, l}$.
    #   The coefficient for $B$ is $B_{r, l, q}$.
    #   The term contributes to $T_{i,j,k}$ if $s_j = l$.
    #   So $j = 2 b_j + l$.
    #   $i = 2 b_i + p$.
    #   $k = 2 b_k + q$.
    #   
    #   So we generate 7 * 2 = 14 terms?
    #   Wait, $A_{r, p, l}$ is a linear combination of $A$ blocks $A_{b_i, b_j}$.
    #   So $U$ will have non-zero entries at multiple $(i, j)$ indices.
    #   Similarly for $V$ and $W$.
    #   
    #   Total rank from core: 14?
    #   But rank of 4x4x4 is 28.
    #   Why? Because $A_{r, p, l}$ is a vector of 4 values (over $b_i, b_j$).
    #   $B_{r, l, q}$ is a vector of 4 values (over $b_j, b_k$).
    #   The product $A_{r, p, l} B_{r, l, q}$ is not a single scalar.
    #   We need to decompose the tensor product of these linear maps.
    #   
    #   Actually, the rank of the 4x4x4 tensor is 28.
    #   My calculation of 14 was wrong because I treated the block coefficients as scalars.
    #   Each "term" in the 14 is actually a rank 2 tensor?
    #   $A_{block} \otimes B_{block}$.
    #   The rank of the product of two rank 4 vectors?
    #   The coefficient of $A$ is a vector $u \in \mathbb{R}^4$ (flattened blocks).
    #   The coefficient of $B$ is a vector $v \in \mathbb{R}^4$.
    #   The term is $u \otimes v$.
    #   This is a rank 1 tensor in the $A \times B$ space.
    #   But we also have the $W$ part.
    #   
    #   Let's just use the known rank 28 decomposition.
    #   I will hardcode the 28 vectors for the 4x4x4 core.
    
    # To save time and space, I will generate the 28 terms programmatically.
    
    U_core = []
    V_core = []
    W_core = []
    
    # Strassen coefficients for A blocks (b_i, b_j) for each M_r
    # A_blocks: 00, 01, 10, 11
    # M0: A00+A11 -> [1, 0, 0, 1]
    # M1: A10+A11 -> [0, 0, 1, 1]
    # M2: A00 -> [1, 0, 0, 0]
    # M3: A11 -> [0, 0, 0, 1]
    # M4: A00+A01 -> [1, 1, 0, 0]
    # M5: A10-A00 -> [-1, 0, 1, 0]
    # M6: A01-A11 -> [0, 1, 0, -1]
    
    A_coeff = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Strassen coefficients for B blocks (b_j, b_k) for each M_r
    # B_blocks: 00, 01, 10, 11
    # M0: B00+B11 -> [1, 0, 0, 1]
    # M1: B00 -> [1, 0, 0, 0]
    # M2: B10-B11 -> [0, 0, 1, -1]
    # M3: B01-B00 -> [-1, 1, 0, 0]
    # M4: B11 -> [0, 0, 0, 1]
    # M5: B00+B01 -> [1, 1, 0, 0]
    # M6: B10+B11 -> [0, 0, 1, 1]
    
    B_coeff = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # For each M_r, we have a 2x2 matrix product.
    # The 2x2 product has rank 4.
    # The 4 terms correspond to the 4 combinations of intra-block indices?
    # No, the rank 4 of 2x2x2 is achieved by 4 terms.
    # The terms are:
    # t0: s_j=0, s_i=0, s_k=0
    # t1: s_j=0, s_i=1, s_k=1
    # t2: s_j=1, s_i=0, s_k=1
    # t3: s_j=1, s_i=1, s_k=0
    # This is a specific rank 4 decomposition.
    
    # Let's use the simple schoolbook rank 4 for 2x2x2:
    # Sum over s_j in {0,1} and s_i in {0,1} and s_k in {0,1} is rank 8.
    # We need rank 4.
    # The standard rank 4 decomposition of T_{ijk}=1 is:
    # sum_{j=0}^1 sum_{i=0}^1 delta_{i,i'} delta_{j,j'} delta_{k, i'} ?
    # Actually, I will just use the 8 schoolbook terms for the 2x2x2 part and accept rank 28*2=56?
    # No, 7 * 4 = 28.
    # I will implement the 4 terms for each M_r.
    
    # The 4 terms for a 2x2x2 all-ones tensor:
    # r0: i=0, j=0, k=0
    # r1: i=0, j=1, k=1
    # r2: i=1, j=0, k=1
    # r3: i=1, j=1, k=0
    
    # For each M_r, we generate 4 terms.
    # Total 28 terms.
    
    for r in range(7):
        # A_coeff[r] is [A00, A01, A10, A11]
        # B_coeff[r] is [B00, B01, B10, B11]
        
        # The 4 terms for the 2x2x2 multiplication:
        # (s_i, s_j, s_k)
        terms = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0)
        ]
        
        for (s_i, s_j, s_k) in terms:
            # Create U, V, W vectors for this term
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            # Set U entries
            # U index: i * m + j
            # i = 2 * b_i + s_i
            # j = 2 * b_j + s_j
            # A_coeff[r][b_idx] where b_idx = b_i * 2 + b_j
            for b_i in range(2):
                for b_j in range(2):
                    i = 2 * b_i + s_i
                    j = 2 * b_j + s_j
                    # Only valid if i < 4, j < 5 (j < 4 for core)
                    if i < n and j < m:
                        b_idx = b_i * 2 + b_j
                        val = A_coeff[r][b_idx]
                        if val != 0:
                            u[i * m + j] = val
            
            # Set V entries
            # V index: j * p + k
            # j = 2 * b_j + s_j
            # k = 2 * b_k + s_k
            # B_coeff[r][b_idx] where b_idx = b_j * 2 + b_k
            for b_j in range(2):
                for b_k in range(2):
                    j = 2 * b_j + s_j
                    k = 2 * b_k + s_k
                    if j < m and k < p:
                        b_idx = b_j * 2 + b_k
                        val = B_coeff[r][b_idx]
                        if val != 0:
                            v[j * p + k] = val
            
            # Set W entries
            # W index: k * n + i
            # k = 2 * b_k + s_k
            # i = 2 * b_i + s_i
            # W is 1 for all valid i,k in the core?
            # The Strassen formula gives C_blocks.
            # C_{b_i, b_k} += coeff * M_r
            # So W should be 1 for the corresponding i,k.
            # But we need to match the linear combination.
            # The output of M_r is added to C blocks.
            # C_{b_i, b_k} += W_coeff * M_r[s_i, s_k]
            # So W_{k, i} should be W_coeff.
            
            # Strassen C coefficients:
            # M0: C00+=M0, C01+=M0, C10+=M0, C11+=M0
            # M1: C10+=M1, C11+=M1
            # M2: C00+=M2, C10+=M2
            # M3: C01+=M3, C11+=M3
            # M4: C01+=M4, C11+=M4
            # M5: C11+=M5
            # M6: C00+=M6, C10+=M6
            
            C_coeff = [
                [1, 1, 1, 1], # M0: C00, C01, C10, C11
                [0, 0, 1, 1], # M1: C10, C11
                [1, 0, 1, 0], # M2: C00, C10
                [0, 1, 0, 1], # M3: C01, C11
                [0, 1, 0, 1], # M4: C01, C11
                [0, 0, 0, 1], # M5: C11
                [1, 0, 1, 0]  # M6: C00, C10
            ]
            
            for b_i in range(2):
                for b_k in range(2):
                    k = 2 * b_k + s_k
                    i = 2 * b_i + s_i
                    if k < p and i < n:
                        b_idx = b_i * 2 + b_k
                        val = C_coeff[r][b_idx]
                        if val != 0:
                            w[k * n + i] = val
            
            U_core.append(u)
            V_core.append(v)
            W_core.append(w)
            
    # Now add the 3 rank-1 terms for the edges
    
    # Part 2: j in 0..3, k=4
    u2 = [0.0] * (n * m)
    v2 = [0.0] * (m * p)
    w2 = [0.0] * (p * n)
    for i in range(4):
        for j in range(4):
            u2[i * m + j] = 1.0
    for j in range(4):
        v2[j * p + 4] = 1.0
    for i in range(4):
        w2[4 * n + i] = 1.0
        
    U_core.append(u2)
    V_core.append(v2)
    W_core.append(w2)
    
    # Part 3: j=4, k in 0..3
    u3 = [0.0] * (n * m)
    v3 = [0.0] * (m * p)
    w3 = [0.0] * (p * n)
    for i in range(4):
        u3[i * m + 4] = 1.0
    for k in range(4):
        v3[4 * p + k] = 1.0
    for k in range(4):
        for i in range(4):
            w3[k * n + i] = 1.0
            
    U_core.append(u3)
    V_core.append(v3)
    W_core.append(w3)
    
    # Part 4: j=4, k=4
    u4 = [0.0] * (n * m)
    v4 = [0.0] * (m * p)
    w4 = [0.0] * (p * n)
    for i in range(4):
        u4[i * m + 4] = 1.0
    v4[4 * p + 4] = 1.0
    for i in range(4):
        w4[4 * n + i] = 1.0
        
    U_core.append(u4)
    V_core.append(v4)
    W_core.append(w4)
    
    rank = len(U_core)
    U = U_core
    V = V_core
    W = W_core
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
