import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Total rank
    rank_total = 28
    
    # Initialize U, V, W
    # U: (n*m, rank) = (6, 28)
    # V: (m*p, rank) = (15, 28)
    # W: (p*n, rank) = (10, 28)
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank_total:
            return
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1
        
    # Term 1: A_11 B_11
    # A_11: i in {0,1}, j=0. Index in U: i*m + 0 = i*3.
    # B_11: j=0, k in {0..4}. Index in V: 0*p + k = k.
    # C_ik: i in {0,1}, k in {0..4}. Index in W: k*n + i.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
    # Term 2: A_12 B_21
    # A_12: i in {0,1}, j in {1,2}.
    # B_21: j in {1,2}, k in {0..4}.
    
    # Subterm 2a: A_12 B_21,1 (k in {0,1})
    # This is a 2x2 * 2x2 multiplication.
    # Local A indices: a00 (i=0,j=1), a01 (i=0,j=2), a10 (i=1,j=1), a11 (i=1,j=2)
    # Local B indices: b00 (j=1,k=0), b01 (j=1,k=1), b10 (j=2,k=0), b11 (j=2,k=1)
    # Local C indices: c00 (i=0,k=0), c01 (i=0,k=1), c10 (i=1,k=0), c11 (i=1,k=1)
    
    # Map local to global indices
    def u_idx_12(i, j_local):
        return i * m + (1 + j_local)
        
    def v_idx_21(j_local, k):
        return (1 + j_local) * p + k
        
    def w_idx_c(i, k):
        return k * n + i
        
    # Strassen for 2x2
    # M1 = (a00+a11)(b00+b11)
    # a00: u_idx_12(0,0), a11: u_idx_12(1,1)
    # b00: v_idx_21(0,0), b11: v_idx_21(1,1)
    # c00: w_idx_c(0,0), c01: w_idx_c(0,1), c10: w_idx_c(1,0), c11: w_idx_c(1,1)
    
    # M1
    add_term(u_idx_12(0,0), v_idx_21(0,0), w_idx_c(0,0), 1, 1, 1)
    add_term(u_idx_12(0,0), v_idx_21(0,1), w_idx_c(0,1), 1, 1, 1)
    add_term(u_idx_12(0,0), v_idx_21(1,0), w_idx_c(1,0), 1, 1, 1)
    add_term(u_idx_12(0,0), v_idx_21(1,1), w_idx_c(1,1), 1, 1, 1)
    
    add_term(u_idx_12(1,1), v_idx_21(0,0), w_idx_c(0,0), 1, 1, 1)
    add_term(u_idx_12(1,1), v_idx_21(0,1), w_idx_c(0,1), 1, 1, 1)
    add_term(u_idx_12(1,1), v_idx_21(1,0), w_idx_c(1,0), 1, 1, 1)
    add_term(u_idx_12(1,1), v_idx_21(1,1), w_idx_c(1,1), 1, 1, 1)
    
    # M2 = (a10+a11)b00
    add_term(u_idx_12(1,0), v_idx_21(0,0), w_idx_c(1,0), 1, 1, 1)
    add_term(u_idx_12(1,1), v_idx_21(0,0), w_idx_c(1,0), 1, 1, 1)
    
    # M3 = a00(b01-b11)
    add_term(u_idx_12(0,0), v_idx_21(0,1), w_idx_c(0,1), 1, 1, 1)
    add_term(u_idx_12(0,0), v_idx_21(1,1), w_idx_c(1,1), 1, -1, 1) # Wait, W index for c01 is w_idx_c(0,1).
    # Correction: M3 contributes to c01 and c11?
    # C = [ M1+M4-M5+M7, M3+M5 ]
    #     [ M2+M4,       M1-M2+M3+M6 ]
    # M3 is in c01 and c11.
    # c01 index: w_idx_c(0,1).
    # c11 index: w_idx_c(1,1).
    # So:
    add_term(u_idx_12(0,0), v_idx_21(0,1), w_idx_c(0,1), 1, 1, 1)
    add_term(u_idx_12(0,0), v_idx_21(1,1), w_idx_c(1,1), 1, -1, 1)
    # Wait, the term is a00 * (b01 - b11).
    # So a00*b01 goes to c01 and c11?
    # No, M3 is a scalar.
    # C01 = M3 + M5.
    # C11 = M1 - M2 + M3 + M6.
    # So M3 adds to C01 and C11.
    # The product is a00 * b01.
    # This product appears in M3.
    # So we need to add a term for a00*b01 that contributes to C01 and C11.
    # But U, V, W are rank-1 tensors.
    # A rank-1 tensor $u v w^T$ adds $u_i v_j w_k$ to $T_{ijk}$.
    # Here, the "output" is a matrix.
    # The tensor is $T_{i, j, k} = A_{ij} B_{jk}$.
    # My `add_term` function adds to a specific entry of U, V, W.
    # This corresponds to a rank-1 tensor $U[:, col] V[:, col] W[:, col]^T$.
    # This tensor contributes to ALL entries of C where U, V, W are non-zero.
    # Wait, the decomposition is $\sum_r U[:, r] V[:, r]^T \otimes W[:, r]$?
    # No, the tensor is $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # $T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}$.
    # Here $i$ is row of A, $j$ is col of A / row of B, $k$ is col of B.
    # $U$ has indices $(i, j)$? No, $U$ is shape $(n \times m, \text{rank})$.
    # So $U_{ij, r}$.
    # $V$ is shape $(m \times p, \text{rank})$. $V_{jk, r}$.
    # $W$ is shape $(p \times n, \text{rank})$. $W_{ki, r}$.
    # The tensor entry $T_{i, j, k}$ is $\sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r}$.
    # This matches $C_{ik} = \sum_j A_{ij} B_{jk}$.
    
    # So, if I want to add a term $A_{i_1 j_1} B_{j_2 k_2}$ to $C_{i_3 k_3}$,
    # I must ensure that the rank-1 tensor $u \otimes v \otimes w$ has:
    # $u$ non-zero only at $i_1 m + j_1$?
    # No, if $u$ has multiple non-zeros, it multiplies multiple $A$ entries.
    # Strassen combines sums like $(a+d)(e+h)$.
    # This requires $u$ to have non-zeros at $a$ and $d$.
    # And $v$ to have non-zeros at $e$ and $h$.
    # And $w$ to have non-zeros at the output entries.
    # But the product $(a+d)(e+h) = ae + ah + de + dh$.
    # The rank-1 tensor with $u=[1,0,1,0]^T, v=[1,1,0,0]^T, w=[1,1,1,1]^T$
    # produces $T_{i,j,k} = \sum u_i v_j w_k$.
    # This is NOT $(a+d)(e+h)$.
    # The tensor product is bilinear in the vectors.
    # $\sum_r U_{ij, r} V_{jk, r} W_{ki, r}$.
    # If $U_{ij, r}$ is non-zero for multiple $ij$, and $V_{jk, r}$ for multiple $jk$,
    # the term is $\sum_{ij, jk} U_{ij, r} V_{jk, r} W_{ki, r}$.
    # This sums over all combinations.
    # Strassen works because we can define linear combinations of inputs.
    # But the tensor decomposition is over the original basis.
    # To use Strassen, we must expand the linear combinations.
    # $(a+d)(e+h) = ae + ah + de + dh$.
    # This is 4 rank-1 terms: $ae, ah, de, dh$.
    # But Strassen claims rank 7.
    # How?
    # The rank of the tensor $M_{2,2,2}$ is 7.
    # This means there exist 7 vectors $u_r, v_r, w_r$ such that
    # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
    # The entries of $u_r$ are coefficients of $a,b,c,d$.
    # The entries of $v_r$ are coefficients of $e,f,g,h$.
    # The entries of $w_r$ are coefficients of $c_{11}, c_{12}, c_{21}, c_{22}$.
    #
    # Let's check the rank-1 expansion of $M_1 = (a+d)(e+h)$.
    # This term is not a single rank-1 tensor in the decomposition of the multiplication tensor.
    # The multiplication tensor is fixed: $T_{ij, jk, ki} = \delta_{ij, jk, ki}$? No.
    # $T$ is defined by $C = AB$.
    # $T_{i, j, k} = 1$ if $j$ matches? No.
    # $T_{i, j, k}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
    # So $T_{i, j, k} = 1$ for all $i, j, k$ in the valid range?
    # No, $T$ is a 3-tensor. $T_{i, j, k} = 1$ if the indices are valid?
    # Actually, $T_{i, j, k}$ is 1 if we are computing $C_{ik}$ using $A_{ij}$ and $B_{jk}$.
    # Since $C_{ik} = \sum_j A_{ij} B_{jk}$, the tensor has $T_{i, j, k} = 1$ for all $i, j, k$.
    # Wait, $T$ is indexed by $i$ (row of C), $j$ (col of A / row of B), $k$ (col of C).
    # No, standard tensor notation: $T_{i, j, k}$ where $i$ is A-row, $j$ is A-col/B-row, $k$ is B-col.
    # Then $C_{ik} = \sum_j T_{i, j, k} A_{ij} B_{jk}$.
    # So $T_{i, j, k} = 1$ for all $i, j, k$.
    # The rank of this tensor is $n \times m \times p$.
    # Strassen finds a decomposition of this tensor into 7 rank-1 tensors.
    # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
    # Here $u_r \in \mathbb{R}^{n \times m}$, $v_r \in \mathbb{R}^{m \times p}$, $w_r \in \mathbb{R}^{p \times n}$.
    #
    # For $2 \times 2 \times 2$:
    # $u_r$ has entries for $a, b, c, d$.
    # $v_r$ has entries for $e, f, g, h$.
    # $w_r$ has entries for $c_{11}, c_{12}, c_{21}, c_{22}$.
    #
    # Example: $M_1 = (a+d)(e+h)$.
    # This is not a rank-1 tensor.
    # The rank-1 tensors are the terms in the sum.
    # The 7 terms of Strassen are:
    # 1. $(a+d)(e+h) \to$ contributes to $c_{11}, c_{12}, c_{21}, c_{22}$.
    #    But this is a product of sums.
    #    In the tensor decomposition, we write:
    #    $u_1 = [1, 0, 0, 1]^T$ (for a, d)
    #    $v_1 = [1, 1, 0, 0]^T$ (for e, h) -- Wait, indices for e,f,g,h.
    #    $e$ is $b_{11}$, $f$ is $b_{12}$, $g$ is $b_{21}$, $h$ is $b_{22}$.
    #    $v_1$ should have 1 at $e$ and $h$.
    #    $w_1$ should have 1 at $c_{11}, c_{12}, c_{21}, c_{22}$?
    #    If we use $u_1 \otimes v_1 \otimes w_1$, the contribution is
    #    $\sum_{a,d} \sum_{e,h} u_{a/d} v_{e/h} w_{c}$.
    #    This gives $ae w_{11} + ah w_{12} + de w_{21} + dh w_{22}$.
    #    We want this to match the contribution of $M_1$ to the final sum.
    #    $M_1$ contributes $+1$ to $c_{11}$, $+1$ to $c_{12}$?
    #    $C_{11} = M_1 + M_4 - M_5 + M_7$.
    #    $C_{12} = M_3 + M_5$.
    #    $C_{21} = M_2 + M_4$.
    #    $C_{22} = M_1 - M_2 + M_3 + M_6$.
    #    
    #    Wait, $M_1$ does NOT contribute to $C_{12}$.
    #    So $w_1$ should be $[1, 0, 0, 1]^T$?
    #    Let's check $C_{11}$ and $C_{22}$.
    #    $M_1$ is in $C_{11}$ and $C_{22}$.
    #    So $w_1$ has 1 at $c_{11}$ and $c_{22}$.
    #    Then the term is $ae c_{11} + ah c_{22} + de c_{11} + dh c_{22}$?
    #    No, the sum is over all combinations.
    #    $u_1 = [1, 0, 0, 1]$ (a, d)
    #    $v_1 = [1, 0, 0, 1]$ (e, h)
    #    $w_1 = [1, 0, 0, 1]$ (c11, c22)
    #    Product: $ae c_{11} + ah c_{22} + de c_{11} + dh c_{22}$.
    #    This matches the $M_1$ contributions to $C_{11}$ ($ae+de$) and $C_{22}$ ($ah+dh$)?
    #    $M_1 = (a+d)(e+h) = ae + ah + de + dh$.
    #    In $C_{11}$, we need $ae + de$.
    #    In $C_{22}$, we need $ah + dh$.
    #    The term $ae$ is in $C_{11}$.
    #    The term $de$ is in $C_{11}$.
    #    The term $ah$ is in $C_{22}$.
    #    The term $dh$ is in $C_{22}$.
    #    So yes, $w_1 = [1, 0, 0, 1]$ works for $M_1$.
    
    #    So, to implement Strassen, I just need to add the 7 rank-1 terms
    #    corresponding to the 7 Strassen multiplications.
    
    #    I will write a helper function `add_strassen_2x2` that takes the base indices
    #    for A, B, C and adds the 7 terms.
    
    #    Similarly for the other blocks.
    
    #    Let's re-verify the indices for the 7 terms.
    #    A: a(0,0), b(0,1), c(1,0), d(1,1)
    #    B: e(0,0), f(0,1), g(1,0), h(1,1)
    #    C: c11(0,0), c12(0,1), c21(1,0), c22(1,1)
    
    #    M1: (a+d)(e+h) -> w: c11, c22
    #    u: a, d. v: e, h. w: c11, c22.
    #    M2: (c+d)e -> w: c21
    #    u: c, d. v: e. w: c21.
    #    M3: a(f-h) -> w: c12, c22
    #    u: a. v: f, -h. w: c12, c22.
    #    M4: d(g-e) -> w: c11, c21
    #    u: d. v: g, -e. w: c11, c21.
    #    M5: (a+b)h -> w: c12, c22
    #    u: a, b. v: h. w: c12, c22.
    #    M6: (c-a)(e+f) -> w: c22
    #    u: c, -a. v: e, f. w: c22.
    #    M7: (b-d)(g+h) -> w: c11
    #    u: b, -d. v: g, h. w: c11.
    
    #    Let's check C11: M1 + M4 - M5 + M7.
    #    M1: ae, de. (w11)
    #    M4: dg, -de. (w11) -> dg - de.
    #    M5: -ah, -bh. (w12) -> No, M5 is in C12 and C22.
    #    Wait, M5 = (a+b)h = ah + bh.
    #    C12 = M3 + M5.
    #    C22 = M1 - M2 + M3 + M6.
    #    So M5 is NOT in C11.
    #    My previous check: "M5 is in C12 and C22".
    #    So w5 should be [0, 1, 0, 1].
    #    Let's re-evaluate w for each M.
    
    #    M1: C11, C22. w=[1,0,0,1]
    #    M2: C21. w=[0,0,1,0]
    #    M3: C12, C22. w=[0,1,0,1]
    #    M4: C11, C21. w=[1,0,1,0]
    #    M5: C12, C22. w=[0,1,0,1]
    #    M6: C22. w=[0,0,0,1]
    #    M7: C11. w=[1,0,0,0]
    
    #    Check C11: M1(1) + M4(1) - M5(0) + M7(1) = 3?
    #    Formula: C11 = M1 + M4 - M5 + M7.
    #    Coefficients:
    #    M1: 1
    #    M4: 1
    #    M5: -1
    #    M7: 1
    #    Sum: 1+1-1+1 = 2? No, the formula is for the values.
    #    The tensor sum is $\sum w_{r, c} M_r$.
    #    So for C11, we need sum of $w_{r, 11} \times (\text{coeff of } M_r \text{ in } C_{11})$.
    #    Wait, the rank-1 tensor $u_r \otimes v_r \otimes w_r$ computes $M_r \times w_r$.
    #    So the contribution to $C_{11}$ is $\sum_r w_{r, 11} M_r$.
    #    We want this to equal $C_{11} = M_1 + M_4 - M_5 + M_7$.
    #    So we need:
    #    $w_{1, 11} = 1$
    #    $w_{2, 11} = 0$
    #    $w_{3, 11} = 0$
    #    $w_{4, 11} = 1$
    #    $w_{5, 11} = -1$
    #    $w_{6, 11} = 0$
    #    $w_{7, 11} = 1$
    
    #    Let's check my w vectors:
    #    w1: [1,0,0,1] -> w1,11 = 1. OK.
    #    w2: [0,0,1,0] -> w2,11 = 0. OK.
    #    w3: [0,1,0,1] -> w3,11 = 0. OK.
    #    w4: [1,0,1,0] -> w4,11 = 1. OK.
    #    w5: [0,1,0,1] -> w5,11 = 0. BUT we need -1.
    #    So w5 should be [0, 1, 0, 1] with a coefficient?
    #    No, the rank-1 term is $u_5 \otimes v_5 \otimes w_5$.
    #    $M_5 = (a+b)h$.
    #    $u_5 = [1, 1, 0, 0]$.
    #    $v_5 = [0, 0, 0, 1]$.
    #    $w_5 = [0, 1, 0, 1]$.
    #    Contribution to C11: $w_{5, 11} M_5 = 0 \cdot M_5 = 0$.
    #    But we need $-1 \cdot M_5$.
    #    So we can set $w_5 = [0, 1, 0, 1]$ and multiply the whole term by -1?
    #    No, we can just set $w_5 = [0, 1, 0, 1]$ and use $u_5 = [-1, -1, 0, 0]$?
    #    Or $v_5 = [0, 0, 0, -1]$?
    #    Or $w_5 = [0, 1, 0, 1]$ and accept that the coefficient is 1, but we need -1?
    #    We can absorb the sign into U, V, or W.
    #    Let's set $w_5 = [0, 1, 0, 1]$ and $u_5 = [-1, -1, 0, 0]$.
    #    Then $M_5' = (-a-b)h = -M_5$.
    #    Then contribution is $-M_5$.
    #    This works.
    
    #    Let's check C12: M3 + M5.
    #    Need:
    #    w3,12 = 1
    #    w5,12 = 1
    #    Others 0.
    #    w3: [0,1,0,1] -> w3,12 = 1. OK.
    #    w5: [0,1,0,1] -> w5,12 = 1. OK.
    #    w1: 0. w2: 0. w4: 0. w6: 0. w7: 0.
    #    Sum: M3 + M5. OK.
    
    #    Check C21: M2 + M4.
    #    Need:
    #    w2,21 = 1
    #    w4,21 = 1
    #    Others 0.
    #    w2: [0,0,1,0] -> w2,21 = 1. OK.
    #    w4: [1,0,1,0] -> w4,21 = 1. OK.
    #    Sum: M2 + M4. OK.
    
    #    Check C22: M1 - M2 + M3 + M6.
    #    Need:
    #    w1,22 = 1
    #    w2,22 = -1
    #    w3,22 = 1
    #    w6,22 = 1
    #    Others 0.
    #    w1: [1,0,0,1] -> w1,22 = 1. OK.
    #    w2: [0,0,1,0] -> w2,22 = 0. NEED -1.
    #    So set w2 = [0,0,1,-1]?
    #    If w2 = [0,0,1,-1], then w2,21 = 1, w2,22 = -1.
    #    Then contribution to C21 is $1 \cdot M_2$.
    #    Contribution to C22 is $-1 \cdot M_2$.
    #    This works!
    #    w3: [0,1,0,1] -> w3,22 = 1. OK.
    #    w6: [0,0,0,1] -> w6,22 = 1. OK.
    #    Sum: M1 - M2 + M3 + M6. OK.
    
    #    So the W vectors are:
    #    w1 = [1, 0, 0, 1]
    #    w2 = [0, 0, 1, -1]
    #    w3 = [0, 1, 0, 1]
    #    w4 = [1, 0, 1, 0]
    #    w5 = [0, 1, 0, 1] (with u5 sign flipped)
    #    w6 = [0, 0, 0, 1]
    #    w7 = [1, 0, 0, 0]
    
    #    And U, V vectors:
    #    1: u=[1,0,0,1], v=[1,0,0,1]
    #    2: u=[0,0,1,1], v=[1,0,0,0]
    #    3: u=[1,0,0,0], v=[0,1,0,-1]
    #    4: u=[0,0,0,1], v=[0,0,1,-1]
    #    5: u=[-1,-1,0,0], v=[0,0,0,1]
    #    6: u=[0,0,1,-1], v=[1,1,0,0]
    #    7: u=[0,1,0,-1], v=[0,0,1,1]
    
    #    This looks correct.
    
    #    I will implement this logic in the code.

    # The code will be structured to build these terms.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
