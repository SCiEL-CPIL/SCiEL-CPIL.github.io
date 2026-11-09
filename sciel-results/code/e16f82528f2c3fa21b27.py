import time
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, cu, cv, cw):
        nonlocal col
        U[u_idx][col] += cu
        V[v_idx][col] += cv
        W[w_idx][col] += cw
        col += 1

    # Helper for 2x2x2 Strassen
    # A rows: ar0, ar1; A cols: ac0, ac1
    # B rows: br0, br1; B cols: bc0, bc1
    # C rows: cr0, cr1; C cols: cc0, cc1
    # Note: A and B share the middle index.
    # In our global setup:
    # A is 2x5. Row i, Col j.
    # B is 5x5. Row j, Col k.
    # C is 2x5. Row i, Col k.
    
    # Block 1: A rows 0,1. B rows 0,1.
    # Sub-block 1.1: B cols 0,1. (2x2x2)
    # A indices: (0,0), (0,1), (1,0), (1,1) -> u_idx = i*5+j
    # B indices: (0,0), (0,1), (1,0), (1,1) -> v_idx = j*5+k
    # C indices: (0,0), (0,1), (1,0), (1,1) -> w_idx = k*2+i
    
    def strassen_block(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cc0, cc1):
        # A is 2x2 at rows ar0, ar1 and cols ac0, ac1
        # B is 2x2 at rows br0, br1 and cols bc0, bc1
        # Note: ac0/1 should correspond to br0/1 in the global tensor structure?
        # Yes, the inner dimension is shared.
        # So ac0 == br0 and ac1 == br1 in terms of the global 'j' index.
        
        # A elements
        a00 = (ar0, ac0)
        a01 = (ar0, ac1)
        a10 = (ar1, ac0)
        a11 = (ar1, ac1)
        
        # B elements
        b00 = (br0, bc0)
        b01 = (br0, bc1)
        b10 = (br1, bc0)
        b11 = (br1, bc1)
        
        # C elements
        c00 = (cc0, ar0) # w_idx = k*n + i. k=cc0, i=ar0
        c01 = (cc1, ar0)
        c10 = (cc0, ar1)
        c11 = (cc1, ar1)
        
        # u_idx for A: i*m + j
        u_a00 = ar0 * m + ac0
        u_a01 = ar0 * m + ac1
        u_a10 = ar1 * m + ac0
        u_a11 = ar1 * m + ac1
        
        # v_idx for B: j*p + k
        v_b00 = br0 * p + bc0
        v_b01 = br0 * p + bc1
        v_b10 = br1 * p + bc0
        v_b11 = br1 * p + bc1
        
        # w_idx for C: k*n + i
        w_c00 = cc0 * n + ar0
        w_c01 = cc1 * n + ar0
        w_c10 = cc0 * n + ar1
        w_c11 = cc1 * n + ar1
        
        # m1 = (a00+a11)(b00+b11)
        add_term(u_a00, v_b00, w_c00, 1, 1, 1)
        add_term(u_a00, v_b11, w_c00, 1, 1, 1)
        add_term(u_a11, v_b00, w_c00, 1, 1, 1)
        add_term(u_a11, v_b11, w_c00, 1, 1, 1)
        # m1 also contributes to c11 with -1
        add_term(u_a00, v_b00, w_c11, 1, 1, -1)
        add_term(u_a00, v_b11, w_c11, 1, 1, -1)
        add_term(u_a11, v_b00, w_c11, 1, 1, -1)
        add_term(u_a11, v_b11, w_c11, 1, 1, -1)
        
        # Wait, this is not how rank-1 decomposition works.
        # A rank-1 term is u * v * w.
        # m1 = (A00+A11)(B00+B11).
        # This is (1*A00 + 1*A11) * (1*B00 + 1*B11).
        # This requires 4 rank-1 terms if we expand? No.
        # The product of sums is not a single rank-1 tensor.
        # (x+y)(z+w) = xz + xw + yz + yw.
        # So m1 contributes 4 terms to the decomposition?
        # No, Strassen's algorithm says we compute 7 products.
        # Each product is a scalar multiplication of two linear combinations.
        # In the tensor decomposition context, a "term" is a rank-1 tensor.
        # If we use Strassen, we are effectively changing the basis.
        # The rank of the tensor is 7.
        # This means there exist 7 rank-1 tensors that sum to the matmul tensor.
        # We don't necessarily have to use the "expanded" form.
        # We can define the U, V, W vectors to be the linear combinations themselves.
        
        # So for m1:
        # U vector has 1 at a00, 1 at a11.
        # V vector has 1 at b00, 1 at b11.
        # W vector has 1 at c00, -1 at c11.
        # This single rank-1 term? No.
        # The tensor is $T = \sum_{s=1}^7 u^{(s)} \otimes v^{(s)} \otimes w^{(s)}$.
        # For m1, $u^{(1)}$ is the vector with 1s at a00, a11.
        # $v^{(1)}$ is the vector with 1s at b00, b11.
        # $w^{(1)}$ is the vector with 1 at c00, -1 at c11.
        # This corresponds to ONE rank in the decomposition.
        
        # Let's re-verify.
        # $C_{00} = m_1 + m_4 - m_5 + m_7$
        # $C_{11} = m_1 - m_2 + m_3 + m_6$
        # So $w^{(1)}$ should have 1 at c00 and 1 at c11?
        # No, $m_1$ adds to $C_{00}$ and $C_{11}$.
        # So $w^{(1)}$ has 1 at c00 and 1 at c11.
        
        # Let's implement Strassen correctly with 7 terms.
        
        # m1
        add_term(u_a00, v_b00, w_c00, 1, 1, 1) # This is wrong. add_term adds a single rank-1 component.
        # I need to add a term where U has multiple non-zeros.
        
        # Let's modify add_term to accept lists of indices and coeffs.
        
    # Redefine add_term
    def add_term_multi(u_idxs, u_coeffs, v_idxs, v_coeffs, w_idxs, w_coeffs):
        nonlocal col
        for idx, c in zip(u_idxs, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_idxs, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_idxs, w_coeffs):
            W[idx][col] += c
        col += 1

    # Re-implement Strassen
    def strassen_block(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cc0, cc1):
        u_a00 = ar0 * m + ac0
        u_a01 = ar0 * m + ac1
        u_a10 = ar1 * m + ac0
        u_a11 = ar1 * m + ac1
        
        v_b00 = br0 * p + bc0
        v_b01 = br0 * p + bc1
        v_b10 = br1 * p + bc0
        v_b11 = br1 * p + bc1
        
        w_c00 = cc0 * n + ar0
        w_c01 = cc1 * n + ar0
        w_c10 = cc0 * n + ar1
        w_c11 = cc1 * n + ar1
        
        # m1 = (a00+a11)(b00+b11) -> c00 + c11
        add_term_multi([u_a00, u_a11], [1, 1], [v_b00, v_b11], [1, 1], [w_c00, w_c11], [1, 1])
        
        # m2 = (a10+a11)b00 -> c10 + c11(-1)
        add_term_multi([u_a10, u_a11], [1, 1], [v_b00], [1], [w_c10, w_c11], [1, -1])
        
        # m3 = a00(b01-b11) -> c01 + c11
        add_term_multi([u_a00], [1], [v_b01, v_b11], [1, -1], [w_c01, w_c11], [1, 1])
        
        # m4 = a11(b10-b00) -> c00 + c10
        add_term_multi([u_a11], [1], [v_b10, v_b00], [1, -1], [w_c00, w_c10], [1, 1])
        
        # m5 = (a00+a01)b11 -> c00(-1) + c01
        add_term_multi([u_a00, u_a01], [1, 1], [v_b11], [1], [w_c00, w_c01], [-1, 1])
        
        # m6 = (a10-a00)(b00+b01) -> c11
        add_term_multi([u_a10, u_a00], [1, -1], [v_b00, v_b01], [1, 1], [w_c11], [1])
        
        # m7 = (a01-a11)(b10+b11) -> c00 + c01
        add_term_multi([u_a01, u_a11], [1, -1], [v_b10, v_b11], [1, 1], [w_c00, w_c01], [1, 1])

    # Block 1: A rows 0,1. B rows 0,1.
    # Sub 1.1: B cols 0,1
    strassen_block(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # Sub 1.2: B cols 2,3
    strassen_block(0, 1, 0, 1, 0, 1, 2, 3, 2, 3)
    # Sub 1.3: B col 4 (2x2x2x1)
    # A rows 0,1. B rows 0,1. B col 4.
    # C rows 0,1. C col 4.
    # Terms:
    # a00*b04 -> c04
    # a01*b14 -> c04
    # a10*b04 -> c14
    # a11*b14 -> c14
    u_a00 = 0*m + 0
    u_a01 = 0*m + 1
    u_a10 = 1*m + 0
    u_a11 = 1*m + 1
    v_b04 = 0*p + 4
    v_b14 = 1*p + 4
    w_c04 = 4*n + 0
    w_c14 = 4*n + 1
    
    add_term_multi([u_a00], [1], [v_b04], [1], [w_c04], [1])
    add_term_multi([u_a01], [1], [v_b14], [1], [w_c04], [1])
    add_term_multi([u_a10], [1], [v_b04], [1], [w_c14], [1])
    add_term_multi([u_a11], [1], [v_b14], [1], [w_c14], [1])

    # Block 2: A rows 0,1. B rows 2,3.
    # Sub 2.1: B cols 0,1
    strassen_block(0, 1, 2, 3, 2, 3, 0, 1, 0, 1)
    # Sub 2.2: B cols 2,3
    strassen_block(0, 1, 2, 3, 2, 3, 2, 3, 2, 3)
    # Sub 2.3: B col 4
    u_a02 = 0*m + 2
    u_a03 = 0*m + 3
    u_a12 = 1*m + 2
    u_a13 = 1*m + 3
    v_b24 = 2*p + 4
    v_b34 = 3*p + 4
    
    add_term_multi([u_a02], [1], [v_b24], [1], [w_c04], [1])
    add_term_multi([u_a03], [1], [v_b34], [1], [w_c04], [1])
    add_term_multi([u_a12], [1], [v_b24], [1], [w_c14], [1])
    add_term_multi([u_a13], [1], [v_b34], [1], [w_c14], [1])

    # Block 3: A rows 0,1. B row 4.
    # A col 4.
    # C all cols 0..4.
    # Terms:
    # a04*b4k -> c0k
    # a14*b4k -> c1k
    u_a04 = 0*m + 4
    u_a14 = 1*m + 4
    for k in range(5):
        v_b4k = 4*p + k
        w_c0k = k*n + 0
        w_c1k = k*n + 1
        add_term_multi([u_a04], [1], [v_b4k], [1], [w_c0k], [1])
        add_term_multi([u_a14], [1], [v_b4k], [1], [w_c1k], [1])

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    print(construct())
