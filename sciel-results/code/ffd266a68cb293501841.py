"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition as baseline (rank 90)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Strassen-based block decomposition
    # Partitions:
    # i (rows of A, rows of C): 3 -> {0,1}, {2}
    # j (cols of A, rows of B): 5 -> {0,1}, {2,3}, {4}
    # k (cols of B, cols of C): 6 -> {0,1}, {2,3}, {4,5}
    
    i_blocks = [
        [0, 1],
        [2]
    ]
    j_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    k_blocks = [
        [0, 1],
        [2, 3],
        [4, 5]
    ]
    
    # We will accumulate terms into a list of (u_idx, v_idx, w_idx, val)
    # Then we can determine the rank and build matrices.
    # However, since Strassen terms are linear combinations, we can just add to the matrices directly.
    # But we don't know the final rank ahead of time easily without simulating the structure.
    # Let's simulate the structure to find the rank.
    
    # Each (2,2,2) block contributes 7 terms.
    # Each other block contributes i_size * j_size * k_size terms.
    
    total_rank = 0
    block_terms = [] # List of lists, where each inner list is a list of (u_idx, v_idx, w_idx) for that block's terms
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                i_size = len(i_blk)
                j_size = len(j_blk)
                k_size = len(k_blk)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen 2x2x2
                    # Local indices 0,1 map to global i_blk[0], i_blk[1], etc.
                    # i0 = i_blk[0], i1 = i_blk[1]
                    # j0 = j_blk[0], j1 = j_blk[1]
                    # k0 = k_blk[0], k1 = k_blk[1]
                    
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    
                    # Strassen terms M1..M7
                    # Each term is a product of a linear combo of A and a linear combo of B.
                    # We represent each term as a set of (u_idx, v_idx, w_idx) contributions.
                    # Actually, a single rank-1 term in the decomposition corresponds to one column in U, V, W.
                    # Strassen's algorithm expresses the product of 2x2 matrices as sum of 7 rank-1 terms
                    # in the tensor sense? No, it's 7 scalar multiplications.
                    # The tensor decomposition rank of 2x2x2 is 7.
                    # The 7 terms correspond to the 7 scalar multiplications.
                    # Let's define the 7 rank-1 tensors.
                    
                    # Term 1: (A00+A11)(B00+B11)
                    # U: A00 (1), A11 (1)
                    # V: B00 (1), B11 (1)
                    # W: C00 (1), C11 (1)
                    # Wait, the rank-1 tensor is $u \otimes v \otimes w$.
                    # The contribution to $C_{ik}$ is $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
                    # For a specific rank-1 term, $U$ is a vector over $(i,j)$, $V$ over $(j,k)$, $W$ over $(k,i)$.
                    # The product $U_{ij} V_{jk} W_{ki}$ must sum to $C_{ik}$ contributions.
                    
                    # Let's list the 7 terms explicitly as (U_support, V_support, W_support)
                    # where supports are dicts mapping index to coefficient.
                    
                    terms_222 = []
                    
                    # M1 = (A00+A11)(B00+B11)
                    # Contributes to C00 (+1) and C11 (+1)
                    # U: { (i0,j0): 1, (i1,j1): 1 }
                    # V: { (j0,k0): 1, (j1,k1): 1 }
                    # W: { (k0,i0): 1, (k1,i1): 1 }
                    # Check: 
                    # C00 gets U(i0,j0)V(j0,k0)W(k0,i0) = 1*1*1 = 1. Correct.
                    # C11 gets U(i1,j1)V(j1,k1)W(k1,i1) = 1*1*1 = 1. Correct.
                    # C01 gets U(i0,j0)V(j0,k1)W(k1,i0) = 1*1*0 = 0. Correct.
                    # C10 gets U(i1,j0)V(j0,k0)W(k0,i1) = 0*1*0 = 0. Correct.
                    terms_222.append(
                        {((i0,j0), (j0,k0), (k0,i0)): 1, ((i1,j1), (j1,k1), (k1,i1)): 1}
                    )
                    
                    # M2 = (A10+A11)B00
                    # Contributes to C10 (+1) and C11 (-1)
                    # U: { (i1,j0): 1, (i1,j1): 1 }
                    # V: { (j0,k0): 1 }
                    # W: { (k0,i1): 1, (k1,i1): -1 }
                    # Check:
                    # C10: U(i1,j0)V(j0,k0)W(k0,i1) = 1*1*1 = 1.
                    # C11: U(i1,j1)V(j1,k0)W(k0,i1) ... wait V(j1,k0) is 0.
                    # M2 is (A10+A11)*B00.
                    # A10 is at (i1,j0). A11 is at (i1,j1).
                    # B00 is at (j0,k0).
                    # Product is A10*B00 + A11*B00.
                    # A10*B00 contributes to C10.
                    # A11*B00 contributes to C11.
                    # So U: {(i1,j0):1, (i1,j1):1}, V: {(j0,k0):1}, W: {(k0,i1):1, (k0,i1):1}?
                    # No, W index is (k,i).
                    # For C10 (k0,i1), W is 1.
                    # For C11 (k1,i1), W is 1.
                    # But V only has (j0,k0). So V(j1,k1) is 0.
                    # So A11*B00 term: U(i1,j1)=1, V(j0,k0)=1? No, V index is (j,k).
                    # The term is $U_{ij} V_{jk} W_{ki}$.
                    # If V is non-zero only at (j0,k0), then $j$ must be $j0$ and $k$ must be $k0$.
                    # So $i$ can be anything, but $U_{i,j0}$ must be non-zero.
                    # So $U$ must be supported on $j=j0$.
                    # But M2 uses $A_{10}$ and $A_{11}$. $A_{11}$ is at $j=j1$.
                    # So $V$ must also have a component at $j1$?
                    # No, Strassen's M2 is $(A_{10}+A_{11})B_{00}$.
                    # This is a scalar multiplication.
                    # In the tensor decomposition, this corresponds to one rank-1 term?
                    # No, the rank-1 term corresponds to a single multiplication of two linear forms.
                    # The multiplication is $L_A \cdot L_B$.
                    # $L_A = A_{10} + A_{11}$.
                    # $L_B = B_{00}$.
                    # The contribution to $C$ is $L_A L_B$.
                    # $C_{10} += A_{10}B_{00}$.
                    # $C_{11} += A_{11}B_{00}$.
                    # So $W$ must have $W_{k0,i1}=1$ and $W_{k1,i1}=1$?
                    # Let's check the product:
                    # $\sum U_{ij} V_{jk} W_{ki}$.
                    # If $V$ is only non-zero at $(j0,k0)$, then $j=j0, k=k0$.
                    # Then the sum is $U_{i,j0} V_{j0,k0} W_{k0,i}$.
                    # This can only produce terms where $j=j0$ and $k=k0$.
                    # But $A_{11}$ has $j=j1$. So $U_{i1,j1}$ is non-zero.
                    # This term would require $V_{j1,k}$ to be non-zero.
                    # So $V$ must have support on $j1$ as well?
                    # If $V$ has support on $j1$, then $L_B$ is not just $B_{00}$.
                    
                    # Ah, I see. The "rank-1" in tensor decomposition means $U \otimes V \otimes W$.
                    # $U$ is a vector in $\mathbb{R}^{n \times m}$, $V$ in $\mathbb{R}^{m \times p}$, $W$ in $\mathbb{R}^{p \times n}$.
                    # The term is $\sum_{i,j,k} U_{ij} V_{jk} W_{ki} C_{ik}$? No.
                    # The tensor $T_{ijk} = \sum_c U_{ij,c} V_{jk,c} W_{ki,c}$.
                    # We want $T_{ijk} = 1$ if $i,j,k$ are valid indices for the product?
                    # No, the matrix multiplication tensor is defined such that
                    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
                    # The tensor $T$ has indices $i,j,k$ and $T_{ijk} = 1$ if we are computing $C_{ik}$ from $A_{ij}B_{jk}$.
                    # So $T_{ijk} = 1$ for all valid $i,j,k$.
                    # We want to decompose this tensor.
                    
                    # For the 2x2x2 block:
                    # $T_{ijk} = 1$ for $i,j,k \in \{0,1\}$.
                    # Strassen says this tensor has rank 7.
                    # The 7 terms are:
                    # 1. $U_1 = e_{00} + e_{11}$, $V_1 = e_{00} + e_{11}$, $W_1 = e_{00} + e_{11}$ (indices for W are $k,i$)
                    #    $W_1$ has $W_{00}=1, W_{11}=1$.
                    #    Term: $(e_{00}+e_{11}) \otimes (e_{00}+e_{11}) \otimes (e_{00}+e_{11})$.
                    #    $T_{000} = 1\cdot1\cdot1 = 1$.
                    #    $T_{111} = 1\cdot1\cdot1 = 1$.
                    #    $T_{001} = 1\cdot0\cdot0 = 0$.
                    #    $T_{010} = 0\cdot1\cdot0 = 0$.
                    #    This only covers $T_{000}$ and $T_{111}$.
                    #    We need $T_{ijk}=1$ for ALL 8 combinations.
                    
                    # Let's look up the standard Strassen tensor decomposition.
                    # The 7 rank-1 tensors are:
                    # 1. $(A_{00}+A_{11}) (B_{00}+B_{11}) \rightarrow C_{00}+=M_1, C_{11}+=M_1$
                    #    $U = e_{00} + e_{11}$
                    #    $V = e_{00} + e_{11}$
                    #    $W = e_{00} + e_{11}$ (where $e_{kl}$ in W corresponds to $k,l$)
                    #    Contribution to $T_{ijk}$:
                    #    $i=0,j=0,k=0: 1\cdot1\cdot1=1$.
                    #    $i=1,j=1,k=1: 1\cdot1\cdot1=1$.
                    #    Others 0.
                    
                    # 2. $(A_{10}+A_{11}) B_{00} \rightarrow C_{10}+=M_2, C_{11}-=M_2$
                    #    $U = e_{10} + e_{11}$
                    #    $V = e_{00}$
                    #    $W = e_{01} - e_{11}$ (Wait, $C_{10}$ is $i=1,k=0$. $W_{k,i}$. So $W_{0,1}=1$. $C_{11}$ is $i=1,k=1$. $W_{1,1}=-1$.)
                    #    $W = e_{01} - e_{11}$ in $(k,i)$ indices.
                    #    Contribution:
                    #    $i=1,j=0,k=0: 1\cdot1\cdot1=1$.
                    #    $i=1,j=1,k=0: 1\cdot0\cdot1=0$. (V is 0 for j=1)
                    #    $i=1,j=0,k=1: 1\cdot1\cdot0=0$. (W is 0 for k=1? No, W_{1,1} is -1. V is 0 for k=1? No, V is e_{00} so k=0 only.)
                    #    So only $i=1,j=0,k=0$ gets 1.
                    #    But we need $T_{100}=1$.
                    #    What about $T_{110}$? It should be 1.
                    #    Here $j=1$. $U_{11}=1$. $V_{10}=0$. So 0.
                    #    This doesn't cover $T_{110}$.
                    
                    # I think I am misinterpreting the tensor indices.
                    # The tensor $T$ is such that $C = A \cdot B$.
                    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
                    # So the "tensor" of the multiplication map is $T_{ijk} = 1$ for all $i,j,k$.
                    # We want $\sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1$ for all $i,j,k$.
                    
                    # Let's re-evaluate Term 2:
                    # $U = e_{10} + e_{11}$
                    # $V = e_{00}$
                    # $W = e_{01} - e_{11}$
                    # $i=1, j=0, k=0: U_{10}=1, V_{00}=1, W_{01}=1 \rightarrow 1$.
                    # $i=1, j=1, k=0: U_{11}=1, V_{10}=0, W_{01}=1 \rightarrow 0$.
                    # $i=1, j=0, k=1: U_{10}=1, V_{01}=0, W_{11}=-1 \rightarrow 0$.
                    # $i=1, j=1, k=1: U_{11}=1, V_{11}=0, W_{11}=-1 \rightarrow 0$.
                    # So this term only contributes to $T_{100}$.
                    
                    # We need $T_{110}=1$.
                    # Is there another term?
                    # Strassen's 7 terms cover all 8 $T_{ijk}$?
                    # Let's check the sum of all 7 terms for each $i,j,k$.
                    
                    # Actually, it is known that the rank of 2x2x2 is 7.
                    # I will use the standard explicit decomposition found in literature.
                    
                    # Term 1: U=e00+e11, V=e00+e11, W=e00+e11
                    # Term 2: U=e10+e11, V=e00, W=e01-e11
                    # Term 3: U=e00, V=e10-e11, W=e00
                    # Term 4: U=e11, V=e01-e00, W=e10
                    # Term 5: U=e00+e01, V=e11, W=e10-e00
                    # Term 6: U=e10-e00, V=e00+e01, W=e11
                    # Term 7: U=e01-e11, V=e10+e11, W=e01-e10
                    
                    # Let's verify Term 2 again.
                    # $i=1,j=0,k=0 \rightarrow 1$.
                    # $i=1,j=1,k=0 \rightarrow 0$.
                    # $i=1,j=0,k=1 \rightarrow 0$.
                    # $i=1,j=1,k=1 \rightarrow 0$.
                    
                    # Term 4: U=e11, V=e01-e00, W=e10
                    # $i=1,j=1,k=0 \rightarrow 1\cdot(-1)\cdot1 = -1$.
                    # $i=1,j=1,k=1 \rightarrow 1\cdot1\cdot0 = 0$.
                    
                    # This is getting complicated to verify manually.
                    # I will trust the standard Strassen decomposition.
                    
                    pass # Placeholder for actual implementation below

                else:
                    # Schoolbook for this block
                    pass

    # Since manual construction is error-prone, I will implement the schoolbook fallback 
    # and return it if Strassen implementation is not verified. 
    # However, the prompt asks to combine strategies. 
    # Given the complexity and time limit, returning the schoolbook (rank 90) is safe. 
    # But I can try to implement Strassen for the 2x2x2 blocks carefully.

    # Let's implement Strassen properly.
    
    def strassen_222_terms(i0, i1, j0, j1, k0, k1):
        """
        Returns 7 rank-1 tensor terms for the 2x2x2 matrix multiplication tensor.
        Each term is (u_dict, v_dict, w_dict) where:
          u_dict maps (i,j) -> coeff
          v_dict maps (j,k) -> coeff
          w_dict maps (k,i) -> coeff
        such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k in {0,1}.
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = { (i0,j0): 1, (i1,j1): 1 }
        v1 = { (j0,k0): 1, (j1,k1): 1 }
        w1 = { (k0,i0): 1, (k1,i0): 1, (k0,i1): 1, (k1,i1): 1 }
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = { (i1,j0): 1, (i1,j1): 1 }
        v2 = { (j0,k0): 1 }
        w2 = { (k0,i1): 1, (k1,i1): 1 }
        terms.append((u2, v2, w2))
        
        # M3 = A00(B10-B11)
        u3 = { (i0,j0): 1 }
        v3 = { (j1,k0): 1, (j1,k1): -1 }
        w3 = { (k0,i0): 1, (k1,i0): -1 }
        terms.append((u3, v3, w3))
        
        # M4 = A11(B01-B00)
        u4 = { (i1,j1): 1 }
        v4 = { (j0,k1): 1, (j0,k0): -1 }
        w4 = { (k1,i1): 1, (k0,i1): -1 }
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = { (i0,j0): 1, (i0,j1): 1 }
        v5 = { (j1,k1): 1 }
        w5 = { (k1,i0): 1, (k1,i1): 1 }
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = { (i1,j0): 1, (i0,j0): -1 }
        v6 = { (j0,k0): 1, (j0,k1): 1 }
        w6 = { (k0,i1): 1, (k1,i1): 1, (k0,i0): -1, (k1,i0): -1 }
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = { (i0,j1): 1, (i1,j1): -1 }
        v7 = { (j1,k0): 1, (j1,k1): 1 }
        w7 = { (k0,i0): 1, (k1,i0): 1, (k0,i1): -1, (k1,i1): -1 }
        terms.append((u7, v7, w7))
        
        return terms

    # Accumulate all terms
    all_terms = []
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                i_size = len(i_blk)
                j_size = len(j_blk)
                k_size = len(k_blk)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    terms = strassen_222_terms(i0, i1, j0, j1, k0, k1)
                    all_terms.extend(terms)
                else:
                    # Schoolbook
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                u = { (i,j): 1 }
                                v = { (j,k): 1 }
                                w = { (k,i): 1 }
                                all_terms.append((u, v, w))
    
    # Now build U, V, W matrices
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verify
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if is_valid:
        return {"U": U, "V": V, "W": W}
    else:
        return best_result

# EVOLVE-BLOCK-END
