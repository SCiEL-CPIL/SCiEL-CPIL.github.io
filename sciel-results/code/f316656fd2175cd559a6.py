import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = 6, 21, 14
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # Part 1: M_{2,1,7} (Rank 1)
    # A_1 is col 0 of A. B_1 is row 0 of B.
    # u: A indices (0, 3)
    u1 = np.zeros(nm); u1[0] = 1; u1[3] = 1
    # v: B indices (0..6)
    v1 = np.zeros(mp); v1[0:7] = 1
    # w: C indices (0..13)
    w1 = np.zeros(pn); w1[:] = 1
    add_term(u1, v1, w1)
    
    # Part 2: M_{2,2,7}
    # A_2 is cols 1,2 of A. B_2 is rows 1,2 of B.
    
    # Sub-part 2a: M_{2,2,3}
    # B_{2,1} is cols 0,1,2 of B_2 (global cols 0,1,2)
    
    # 2a-i: M_{2,2,1} (Rank 4)
    # B_{2,1,1} is col 0 of B_{2,1} (global col 0)
    # A indices: 1, 2, 4, 5
    # B indices: 7, 14
    # C indices: 0, 1
    
    # Term 1: A_{0,1} B_{1,0} C_{0,0}
    u = np.zeros(nm); u[1] = 1
    v = np.zeros(mp); v[7] = 1
    w = np.zeros(pn); w[0] = 1
    add_term(u, v, w)
    
    # Term 2: A_{0,2} B_{2,0} C_{0,0}
    u = np.zeros(nm); u[2] = 1
    v = np.zeros(mp); v[14] = 1
    w = np.zeros(pn); w[0] = 1
    add_term(u, v, w)
    
    # Term 3: A_{1,1} B_{1,0} C_{1,0}
    u = np.zeros(nm); u[4] = 1
    v = np.zeros(mp); v[7] = 1
    w = np.zeros(pn); w[1] = 1
    add_term(u, v, w)
    
    # Term 4: A_{1,2} B_{2,0} C_{1,0}
    u = np.zeros(nm); u[5] = 1
    v = np.zeros(mp); v[14] = 1
    w = np.zeros(pn); w[1] = 1
    add_term(u, v, w)
    
    # 2a-ii: M_{2,2,2} (Rank 7, Strassen)
    # B_{2,1,2} is cols 1,2 of B_{2,1} (global cols 1,2)
    # A indices: 1, 2, 4, 5
    # B indices: 8, 9, 15, 16
    # C indices: 2, 3, 4, 5
    
    # Strassen formulas for A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # a=A[0,1]=1, b=A[0,2]=2, c=A[1,1]=4, d=A[1,2]=5
    # e=B[1,1]=8, f=B[1,2]=9, g=B[2,1]=15, h=B[2,2]=16
    
    # m1 = (a+d)(e+h)
    u = np.zeros(nm); u[1] = 1; u[5] = 1
    v = np.zeros(mp); v[8] = 1; v[16] = 1
    w = np.zeros(pn); w[2] = 1; w[5] = 1 # p=s=m1... wait.
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 - m6
    # Indices: p=2, q=3, r=4, s=5
    
    # We add terms for each m_i contribution to p,q,r,s
    
    # m1 = (a+d)(e+h)
    u = np.zeros(nm); u[1] = 1; u[5] = 1
    v = np.zeros(mp); v[8] = 1; v[16] = 1
    w = np.zeros(pn); w[2] = 1; w[5] = 1
    add_term(u, v, w)
    
    # m2 = (c+d)e
    u = np.zeros(nm); u[4] = 1; u[5] = 1
    v = np.zeros(mp); v[8] = 1
    w = np.zeros(pn); w[4] = 1; w[5] = -1 # r=m2, s=-m2
    add_term(u, v, w)
    
    # m3 = a(f-h)
    u = np.zeros(nm); u[1] = 1
    v = np.zeros(mp); v[9] = 1; v[16] = -1
    w = np.zeros(pn); w[3] = 1; w[5] = 1 # q=m3, s=m3
    add_term(u, v, w)
    
    # m4 = d(g-e)
    u = np.zeros(nm); u[5] = 1
    v = np.zeros(mp); v[15] = 1; v[8] = -1
    w = np.zeros(pn); w[2] = 1; w[4] = 1 # p=m4, r=m4
    add_term(u, v, w)
    
    # m5 = (a+b)h
    u = np.zeros(nm); u[1] = 1; u[2] = 1
    v = np.zeros(mp); v[16] = 1
    w = np.zeros(pn); w[2] = -1; w[3] = 1 # p=-m5, q=m5
    add_term(u, v, w)
    
    # m6 = (c-a)(e+f)
    u = np.zeros(nm); u[4] = 1; u[1] = -1
    v = np.zeros(mp); v[8] = 1; v[9] = 1
    w = np.zeros(pn); w[5] = -1 # s=-m6
    add_term(u, v, w)
    
    # m7 = (b-d)(g+h)
    u = np.zeros(nm); u[2] = 1; u[5] = -1
    v = np.zeros(mp); v[15] = 1; v[16] = 1
    w = np.zeros(pn); w[2] = 1 # p=m7
    add_term(u, v, w)
    
    # Sub-part 2b: M_{2,2,4}
    # B_{2,2} is cols 3,4,5,6 of B_2 (global cols 3,4,5,6)
    # Split into two 2x2 blocks
    
    # Block 1: cols 3,4
    # B indices: 10, 11, 17, 18
    # C indices: 6, 7, 8, 9
    
    # m1
    u = np.zeros(nm); u[1] = 1; u[5] = 1
    v = np.zeros(mp); v[10] = 1; v[18] = 1
    w = np.zeros(pn); w[6] = 1; w[9] = 1
    add_term(u, v, w)
    
    # m2
    u = np.zeros(nm); u[4] = 1; u[5] = 1
    v = np.zeros(mp); v[10] = 1
    w = np.zeros(pn); w[8] = 1; w[9] = -1
    add_term(u, v, w)
    
    # m3
    u = np.zeros(nm); u[1] = 1
    v = np.zeros(mp); v[11] = 1; v[18] = -1
    w = np.zeros(pn); w[7] = 1; w[9] = 1
    add_term(u, v, w)
    
    # m4
    u = np.zeros(nm); u[5] = 1
    v = np.zeros(mp); v[17] = 1; v[10] = -1
    w = np.zeros(pn); w[6] = 1; w[8] = 1
    add_term(u, v, w)
    
    # m5
    u = np.zeros(nm); u[1] = 1; u[2] = 1
    v = np.zeros(mp); v[18] = 1
    w = np.zeros(pn); w[6] = -1; w[7] = 1
    add_term(u, v, w)
    
    # m6
    u = np.zeros(nm); u[4] = 1; u[1] = -1
    v = np.zeros(mp); v[10] = 1; v[11] = 1
    w = np.zeros(pn); w[9] = -1
    add_term(u, v, w)
    
    # m7
    u = np.zeros(nm); u[2] = 1; u[5] = -1
    v = np.zeros(mp); v[17] = 1; v[18] = 1
    w = np.zeros(pn); w[6] = 1
    add_term(u, v, w)
    
    # Block 2: cols 5,6
    # B indices: 12, 13, 19, 20
    # C indices: 10, 11, 12, 13
    
    # m1
    u = np.zeros(nm); u[1] = 1; u[5] = 1
    v = np.zeros(mp); v[12] = 1; v[20] = 1
    w = np.zeros(pn); w[10] = 1; w[13] = 1
    add_term(u, v, w)
    
    # m2
    u = np.zeros(nm); u[4] = 1; u[5] = 1
    v = np.zeros(mp); v[12] = 1
    w = np.zeros(pn); w[12] = 1; w[13] = -1
    add_term(u, v, w)
    
    # m3
    u = np.zeros(nm); u[1] = 1
    v = np.zeros(mp); v[13] = 1; v[20] = -1
    w = np.zeros(pn); w[11] = 1; w[13] = 1
    add_term(u, v, w)
    
    # m4
    u = np.zeros(nm); u[5] = 1
    v = np.zeros(mp); v[19] = 1; v[12] = -1
    w = np.zeros(pn); w[10] = 1; w[12] = 1
    add_term(u, v, w)
    
    # m5
    u = np.zeros(nm); u[1] = 1; u[2] = 1
    v = np.zeros(mp); v[20] = 1
    w = np.zeros(pn); w[10] = -1; w[11] = 1
    add_term(u, v, w)
    
    # m6
    u = np.zeros(nm); u[4] = 1; u[1] = -1
    v = np.zeros(mp); v[12] = 1; v[13] = 1
    w = np.zeros(pn); w[13] = -1
    add_term(u, v, w)
    
    # m7
    u = np.zeros(nm); u[2] = 1; u[5] = -1
    v = np.zeros(mp); v[19] = 1; v[20] = 1
    w = np.zeros(pn); w[10] = 1
    add_term(u, v, w)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
