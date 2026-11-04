"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 24) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1
                
    # Verification helper
    def verify_decomp(U, V, W, rank):
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # Rank-22 Decomposition
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    def add_u(i, j, r, val):
        U22[i * m + j, r] += val
    def add_v(j, k, r, val):
        V22[j * p + k, r] += val
    def add_w(k, i, r, val):
        W22[k * n + i, r] += val
        
    # Part 1: A_bottom * B_bottom (Rank 8)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            r = i * p + k
            add_u(i, 2, r, 1)
            add_v(2, k, r, 1)
            add_w(k, i, r, 1)
            
    # Part 2: A_top * B_top (Rank 14) using Strassen for 2x2*2x2
    # We compute the correct Strassen coefficients via linear algebra
    # For 2x2 matrices A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]
    # C = A*B where C00=a00*b00+a01*b10, C01=a00*b01+a01*b11
    # C10=a10*b00+a11*b10, C11=a10*b01+a11*b11
    
    # Use the 7 Strassen products and find correct reconstruction
    # m1=(a00+a11)(b00+b11), m2=(a10+a11)b00, m3=a00(b01-b11)
    # m4=a11(b10-b00), m5=(a00+a01)b11, m6=(a10-a00)(b00+b10)
    # m7=(a01-a11)(b10+b11)
    
    # Solve for reconstruction coefficients using symbolic matching
    # C00 = m1 - m5 + m7 + m4
    # C01 = m3 + m5  
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    # Verify these are correct by checking against a test case
    test_A = np.array([[1,2],[3,4]], dtype=float)
    test_B = np.array([[5,6],[7,8]], dtype=float)
    test_C = test_A @ test_B
    
    a00,a01,a10,a11 = 1,2,3,4
    b00,b01,b10,b11 = 5,6,7,8
    m1 = (a00+a11)*(b00+b11)
    m2 = (a10+a11)*b00
    m3 = a00*(b01-b11)
    m4 = a11*(b10-b00)
    m5 = (a00+a01)*b11
    m6 = (a10-a00)*(b00+b10)
    m7 = (a01-a11)*(b10+b11)
    
    # Check C00
    assert abs((m1 - m5 + m7 + m4) - test_C[0,0]) < 1e-9, f"C00 mismatch: {m1-m5+m7+m4} vs {test_C[0,0]}"
    # Check C01
    assert abs((m3 + m5) - test_C[0,1]) < 1e-9, f"C01 mismatch: {m3+m5} vs {test_C[0,1]}"
    # Check C10
    assert abs((m2 + m4) - test_C[1,0]) < 1e-9, f"C10 mismatch: {m2+m4} vs {test_C[1,0]}"
    # Check C11 - this is where the bug was
    c11_formula = m1 - m2 + m3 + m6
    if abs(c11_formula - test_C[1,1]) > 1e-9:
        # The standard formula is wrong; find correct coefficients
        # We need to solve: c1*m1+c2*m2+c3*m3+c4*m4+c5*m5+c6*m6+c7*m7 = a10*b01+a11*b11
        # Use multiple test cases to determine coefficients
        from itertools import product
        samples = []
        for trial in range(20):
            ta = np.array([[random.uniform(-5,5) for _ in range(2)] for _ in range(2)])
            tb = np.array([[random.uniform(-5,5) for _ in range(2)] for _ in range(2)])
            tc = ta @ tb
            a00t,a01t,a10t,a11t = ta[0,0],ta[0,1],ta[1,0],ta[1,1]
            b00t,b01t,b10t,b11t = tb[0,0],tb[0,1],tb[1,0],tb[1,1]
            m1t = (a00t+a11t)*(b00t+b11t)
            m2t = (a10t+a11t)*b00t
            m3t = a00t*(b01t-b11t)
            m4t = a11t*(b10t-b00t)
            m5t = (a00t+a01t)*b11t
            m6t = (a10t-a00t)*(b00t+b10t)
            m7t = (a01t-a11t)*(b10t+b11t)
            samples.append(([m1t,m2t,m3t,m4t,m5t,m6t,m7t], tc[1,1]))
        
        # Solve least squares
        A_mat = np.array([s[0] for s in samples])
        b_vec = np.array([s[1] for s in samples])
        coeffs, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        # Round to nearest integers/half-integers
        coeffs_rounded = np.round(coeffs * 2) / 2
        # Verify
        max_err = max(abs(np.dot(s[0], coeffs_rounded) - s[1]) for s in samples)
        if max_err < 1e-9:
            c11_coeffs = coeffs_rounded
        else:
            c11_coeffs = np.array([1,-1,1,0,0,1,0])  # fallback to standard
    else:
        c11_coeffs = np.array([1,-1,1,0,0,1,0])
    
    def add_strassen_block(b_col_offset, c_col_offset, r_base):
        # C00 = m1 - m5 + m7 + m4
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = sum(c_i * m_i)
        
        # m1
        r = r_base + 0
        add_u(0, 0, r, 1)
        add_u(1, 1, r, 1)
        add_v(0, b_col_offset, r, 1)
        add_v(1, b_col_offset+1, r, 1)
        add_w(c_col_offset, 0, r, 1)    # C00
        add_w(c_col_offset+1, 1, r, c11_coeffs[0])  # C11
        
        # m2
        r = r_base + 1
        add_u(1, 0, r, 1)
        add_u(1, 1, r, 1)
        add_v(0, b_col_offset, r, 1)
        add_w(c_col_offset, 1, r, 1)    # C10
        add_w(c_col_offset+1, 1, r, c11_coeffs[1]) # C11
        
        # m3
        r = r_base + 2
        add_u(0, 0, r, 1)
        add_v(0, b_col_offset+1, r, 1)
        add_v(1, b_col_offset+1, r, -1)
        add_w(c_col_offset+1, 0, r, 1)  # C01
        add_w(c_col_offset+1, 1, r, c11_coeffs[2])  # C11
        
        # m4
        r = r_base + 3
        add_u(1, 1, r, 1)
        add_v(1, b_col_offset, r, 1)
        add_v(0, b_col_offset, r, -1)
        add_w(c_col_offset, 0, r, 1)    # C00
        add_w(c_col_offset, 1, r, 1)    # C10
        add_w(c_col_offset+1, 1, r, c11_coeffs[3])  # C11
        
        # m5
        r = r_base + 4
        add_u(0, 0, r, 1)
        add_u(0, 1, r, 1)
        add_v(1, b_col_offset+1, r, 1)
        add_w(c_col_offset, 0, r, -1)   # C00
        add_w(c_col_offset+1, 0, r, 1)  # C01
        add_w(c_col_offset+1, 1, r, c11_coeffs[4])  # C11
        
        # m6
        r = r_base + 5
        add_u(1, 0, r, 1)
        add_u(0, 0, r, -1)
        add_v(0, b_col_offset, r, 1)
        add_v(1, b_col_offset, r, 1)
        add_w(c_col_offset+1, 1, r, c11_coeffs[5])  # C11
        
        # m7
        r = r_base + 6
        add_u(0, 1, r, 1)
        add_u(1, 1, r, -1)
        add_v(1, b_col_offset, r, 1)
        add_v(1, b_col_offset+1, r, 1)
        add_w(c_col_offset, 0, r, 1)    # C00
        add_w(c_col_offset+1, 1, r, c11_coeffs[6])  # C11
    
    # Block 1: Left part (cols 0,1)
    add_strassen_block(0, 0, 8)
    
    # Block 2: Right part (cols 2,3)
    add_strassen_block(2, 2, 15)
    
    if verify_decomp(U22, V22, W22, rank_22):
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
        
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
