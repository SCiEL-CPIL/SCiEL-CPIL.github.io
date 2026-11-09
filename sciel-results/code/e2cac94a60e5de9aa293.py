import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w
        col += 1
        
    # 1. Block 2: i=2, 1x4x4 (Schoolbook, rank 16)
    for j in range(4):
        for k in range(4):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
    # Strassen 2x2x2 template
    # Inputs: A[0][0]=a11, A[0][1]=a12, A[1][0]=a21, A[1][1]=a22
    #         B[0][0]=b11, B[0][1]=b12, B[1][0]=b21, B[1][1]=b22
    # Outputs: C[0][0]=c11, C[0][1]=c12, C[1][0]=c21, C[1][1]=c22
    
    # We define the 7 Strassen steps as lists of (u_offsets, v_offsets, w_offsets, coeff)
    # But since we are adding to global matrices, we need to map local indices to global.
    
    def add_strassen_block(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        """
        a_base_i, a_base_j: offset for A[i, j]
        b_base_j, b_base_k: offset for B[j, k]
        c_base_i, c_base_k: offset for C[i, k]
        """
        # Local A indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global A index: (a_base_i + li) * m + (a_base_j + lj)
        
        # Local B indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global B index: (b_base_j + lj) * p + (b_base_k + lk)
        
        # Local C indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global C index: (c_base_k + lk) * n + (c_base_i + li)
        
        def get_u(li, lj):
            return (a_base_i + li) * m + (a_base_j + lj)
            
        def get_v(lj, lk):
            return (b_base_j + lj) * p + (b_base_k + lk)
            
        def get_w(li, lk):
            return (c_base_k + lk) * n + (c_base_i + li)
            
        # M1 = (a11 + a22)(b11 + b22)
        # U: a11, a22. V: b11, b22. W: c11, c22
        # We need to distribute the sum.
        # The tensor decomposition adds rank-1 terms.
        # (a11+a22) is not a single variable.
        # We must express the bilinear form.
        # M1 = a11*b11 + a11*b22 + a22*b11 + a22*b22
        # This requires 4 rank-1 terms if done naively?
        # No, Strassen combines them.
        # The rank-1 term corresponds to a specific linear combination of inputs.
        # U vector has 1s at positions for a11 and a22.
        # V vector has 1s at positions for b11 and b22.
        # W vector has 1s at positions for c11 and c22 (with signs).
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1 = (a11 + a22)(b11 + b22)
        # U: 1 at a11, 1 at a22
        # V: 1 at b11, 1 at b22
        # W: 1 at c11, 1 at c22
        # Wait, W contributes to c11 and c22.
        # c11 += M1, c22 += M1
        # So W has 1 at c11, 1 at c22.
        
        u_11 = get_u(0, 0)
        u_22 = get_u(1, 1)
        v_11 = get_v(0, 0)
        v_22 = get_v(1, 1)
        w_11 = get_w(0, 0)
        w_22 = get_w(1, 1)
        
        # We can't add a "sum" of inputs in one rank-1 term unless we create a new variable.
        # But we are constructing the decomposition directly.
        # The term is $u \cdot v \cdot w^T$.
        # $u$ is a vector in input space A. $v$ is a vector in input space B.
        # $u = e_{a11} + e_{a22}$.
        # $v = e_{b11} + e_{b22}$.
        # $w = e_{c11} + e_{c22}$.
        # This single term contributes to $c11$ and $c22$.
        # Is this valid?
        # Yes, $T = \sum u_r \otimes v_r \otimes w_r$.
        # The contribution to $C_{ik}$ is $\sum_r u_r[A_{ik}] v_r[B_{jk}] w_r[C_{ik}]$.
        # So we add 1 to U at a11, 1 at a22.
        # Add 1 to V at b11, 1 at b22.
        # Add 1 to W at c11, 1 at c22.
        
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_11][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # Term 2: M2 = (a21 + a22)b11
        # U: a21, a22. V: b11. W: c21, c22 (with -1 for c22? No, M2 adds to c21, subtracts from c22)
        # c21 += M2, c22 -= M2
        u_21 = get_u(1, 0)
        u_22 = get_u(1, 1)
        v_11 = get_v(0, 0)
        w_21 = get_w(1, 0)
        w_22 = get_w(1, 1)
        
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_21][col] += 1
        W[w_22][col] -= 1
        col += 1
        
        # Term 3: M3 = a11(b12 - b22)
        # U: a11. V: b12, -b22. W: c11, c12
        # c11 += M3, c12 += M3
        u_11 = get_u(0, 0)
        v_12 = get_v(0, 1)
        v_22 = get_v(1, 1)
        w_11 = get_w(0, 0)
        w_12 = get_w(0, 1)
        
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] -= 1
        W[w_11][col] += 1
        W[w_12][col] += 1
        col += 1
        
        # Term 4: M4 = a22(b21 - b11)
        # U: a22. V: b21, -b11. W: c11, c21
        # c11 += M4, c21 += M4
        u_22 = get_u(1, 1)
        v_21 = get_v(1, 0)
        v_11 = get_v(0, 0)
        w_11 = get_w(0, 0)
        w_21 = get_w(1, 0)
        
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] -= 1
        W[w_11][col] += 1
        W[w_21][col] += 1
        col += 1
        
        # Term 5: M5 = (a11 + a12)b22
        # U: a11, a12. V: b22. W: c11(-1), c12
        # c11 -= M5, c12 += M5
        u_11 = get_u(0, 0)
        u_12 = get_u(0, 1)
        v_22 = get_v(1, 1)
        w_11 = get_w(0, 0)
        w_12 = get_w(0, 1)
        
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_11][col] -= 1
        W[w_12][col] += 1
        col += 1
        
        # Term 6: M6 = (a21 - a11)(b11 + b12)
        # U: a21, -a11. V: b11, b12. W: c22
        # c22 += M6
        u_21 = get_u(1, 0)
        u_11 = get_u(0, 0)
        v_11 = get_v(0, 0)
        v_12 = get_v(0, 1)
        w_22 = get_w(1, 1)
        
        U[u_21][col] += 1
        U[u_11][col] -= 1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # Term 7: M7 = (a12 - a22)(b21 + b22)
        # U: a12, -a22. V: b21, b22. W: c11
        # c11 += M7
        u_12 = get_u(0, 1)
        u_22 = get_u(1, 1)
        v_21 = get_v(1, 0)
        v_22 = get_v(1, 1)
        w_11 = get_w(0, 0)
        
        U[u_12][col] += 1
        U[u_22][col] -= 1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_11][col] += 1
        col += 1

    # 2. Block 1: i in {0,1}, 2x4x4 (Rank 28)
    
    # P1: A_left (i 0-1, j 0-1) * B_top (j 0-1, k 0-3)
    # Split B_top into B_TL (k 0-1) and B_TR (k 2-3)
    
    # Instance 1.1: A_left * B_TL
    # A: i 0-1, j 0-1
    # B: j 0-1, k 0-1
    # C: i 0-1, k 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Instance 1.2: A_left * B_TR
    # A: i 0-1, j 0-1
    # B: j 0-1, k 2-3
    # C: i 0-1, k 2-3
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # P2: A_right (i 0-1, j 2-3) * B_bottom (j 2-3, k 0-3)
    # Split B_bottom into B_BL (k 0-1) and B_BR (k 2-3)
    
    # Instance 2.1: A_right * B_BL
    # A: i 0-1, j 2-3
    # B: j 2-3, k 0-1
    # C: i 0-1, k 0-1
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Instance 2.2: A_right * B_BR
    # A: i 0-1, j 2-3
    # B: j 2-3, k 2-3
    # C: i 0-1, k 2-3
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = res["U"]
    V = res["V"]
    W = res["W"]
    
    n, m, p = 3, 4, 4
    rank = 44
    
    # Check dimensions
    assert len(U) == n*m
    assert len(U[0]) == rank
    assert len(V) == m*p
    assert len(V[0]) == rank
    assert len(W) == p*n
    assert len(W[0]) == rank
    
    # Verify decomposition
    # T[i, j, k] should be 1 if i, j, k match, 0 otherwise?
    # No, T is the tensor of matrix multiplication.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices i, j, k.
    # T[i, j, k] = 1.
    # Decomposition: T = sum_r U[:, r] V[:, r] W[:, r]^T ?
    # U is (nm, rank), V is (mp, rank), W is (pn, rank).
    # U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Sum over r should equal 1 for all i, j, k.
    
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Tensor shape (n, m, p)
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # We can check by contracting.
    # For each r, we have a rank-1 tensor.
    # Sum of U[:, r] V[:, r] W[:, r]
    
    # Let's check a few entries.
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    errors += 1
                    print(f"Error at {i},{j},{k}: {val}")
                    
    if errors == 0:
        print("Verification passed.")
    else:
        print(f"Verification failed with {errors} errors.")
