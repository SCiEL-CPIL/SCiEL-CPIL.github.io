import numpy as np

def winograd_2x2():
    # Returns U, V, W for 2x2x2, rank 7
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # i,j,k in {0,1}
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # W[k*2+i] -> k=0,i=0 is 0; k=1,i=1 is 3.
    # Wait, C00 gets +m1, C11 gets +m1.
    # C00 index in W is 0*2+0=0. C11 index is 1*2+1=3.
    # So W[0,0]=1, W[3,0]=1.
    
    # m2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 (k=0, i=1) -> 0*2+1=1? No.
    # C10 is row 1, col 0. i=1, k=0.
    # W index k*n+i = 0*2+1 = 1.
    # So W[1, 1] = 1.
    # C11 is row 1, col 1. i=1, k=1.
    # W index 1*2+1 = 3.
    # C11 gets -m2. So W[3, 1] = -1.
    W[1, 1] = 1
    W[3, 1] = -1
    
    # m3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01 (i=0, k=1) -> 1*2+0=2? No.
    # C01: i=0, k=1. W index 1*2+0=2.
    # C11: i=1, k=1. W index 3.
    # C01 gets +m3. C11 gets +m3.
    W[2, 2] = 1
    W[3, 2] = 1
    
    # m4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00 (i=0, k=0) -> 0.
    # C00 gets +m4. C10 gets +m4.
    # C10: i=1, k=0. W index 1.
    W[0, 3] = 1
    W[1, 3] = 1
    
    # m5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00 gets -m5.
    # C11 gets +m5.
    W[3, 4] = 1
    
    # m6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 gets +m6.
    
    # m7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 gets +m7.
    # C10 gets -m7?
    # C11 gets +m7.
    # C00 = m1+m4-m5+m7.
    # C01 = m3+m5.
    # C10 = m2+m4-m7?
    # Let's check C10.
    # C10 = a10b00 + a11b10.
    # m2 = (a10+a11)b00 = a10b00 + a11b00.
    # m4 = a11(b10-b00) = a11b10 - a11b00.
    # m2+m4 = a10b00 + a11b10. Correct.
    # So C10 = m2+m4.
    # My W for m2: W[1,1]=1 (C10). Correct.
    # My W for m4: W[1,3]=1 (C10). Correct.
    # So C10 = m2+m4.
    # What about m7?
    # m7 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11.
    # Does it appear in C10? No.
    # Does it appear in C00?
    # C00 = a00b00 + a01b10.
    # m1 = a00b00 + a00b11 + a11b00 + a11b11.
    # m4 = a11b10 - a11b00.
    # -m5 = -a00b11 - a01b11.
    # m7 = a01b10 + a01b11 - a11b10 - a11b11.
    # Sum:
    # a00b00: m1 (1)
    # a00b11: m1 (1) - m5 (1) = 0
    # a11b00: m1 (1) + m4 (-1) = 0
    # a11b11: m1 (1) - m5 (0? m5 has a01) - m7 (1) = 0?
    # m5 = (a00+a01)b11. No a11.
    # m7 has -a11b11.
    # So a11b11: 1 - 1 = 0.
    # a01b10: m7 (1).
    # a01b11: -m5 (1) + m7 (1) = 2? No.
    # -m5 = -a00b11 - a01b11.
    # m7 = ... + a01b11 ...
    # Sum a01b11: -1 + 1 = 0.
    # a11b10: m4 (1) + m7 (-1) = 0.
    # So C00 = a00b00 + a01b10. Correct.
    
    # So W[0,6] = 1 is correct for C00.
    
    return U, V, W
