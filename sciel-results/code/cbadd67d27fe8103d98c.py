"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    # 1 trivial block for column 14
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices: 0:A00(a), 1:A01(b), 2:A10(c), 3:A11(d)
    # B indices: j*p+k, j in {0,1}, k in {0..14}
    # C indices: k*n+i, k in {0..14}, i in {0,1}
    
    def add_strassen_block(base_r, k0, k1):
        r1, r2, r3, r4, r5, r6, r7 = (base_r + i for i in range(7))
        
        e_idx = k0      # B[0][k0]
        f_idx = k1      # B[0][k1]
        g_idx = p + k0  # B[1][k0]
        h_idx = p + k1  # B[1][k1]
        
        # P1 = (a+d)(e+h)
        U[0][r1] = 1; U[3][r1] = 1
        V[e_idx][r1] = 1; V[h_idx][r1] = 1
        
        # P2 = (c+d)e
        U[2][r2] = 1; U[3][r2] = 1
        V[e_idx][r2] = 1
        
        # P3 = a(f-h)
        U[0][r3] = 1
        V[f_idx][r3] = 1; V[h_idx][r3] = -1
        
        # P4 = d(g-e)
        U[3][r4] = 1
        V[g_idx][r4] = 1; V[e_idx][r4] = -1
        
        # P5 = (a+b)h
        U[0][r5] = 1; U[1][r5] = 1
        V[h_idx][r5] = 1
        
        # P6 = (c-a)(e+f)
        U[2][r6] = 1; U[0][r6] = -1
        V[e_idx][r6] = 1; V[f_idx][r6] = 1
        
        # P7 = (b-d)(g+h)
        U[1][r7] = 1; U[3][r7] = -1
        V[g_idx][r7] = 1; V[h_idx][r7] = 1
        
        # Reconstruction: C[k][i] = sum_r U[i*m+j...]*V[j*p+k...]*W[k*n+i...][r]
        # C0k0 (W index 2*k0): a*e+b*g = P1+P4-P5+P7
        W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
        # C1k0 (W index 2*k0+1): c*e+d*g = P2+P4
        W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
        # C0k1 (W index 2*k1): a*f+b*h = P3+P5
        W[2*k1][r3] = 1; W[2*k1][r5] = 1
        # C1k1 (W index 2*k1+1): c*f+d*h = P1-P2+P3+P6
        W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
    
    # Add 7 Strassen blocks
    for i in range(7):
        add_strassen_block(7 * i, 2 * i, 2 * i + 1)
    
    # Trivial block for column 14
    base_r = 49
    r1, r2, r3, r4 = (base_r + i for i in range(4))
    e14 = 14       # B[0][14]
    g14 = p + 14   # B[1][14]
    
    # Q1 = a * B014 -> C014
    U[0][r1] = 1; V[e14][r1] = 1; W[2*14][r1] = 1
    # Q2 = b * B114 -> C014
    U[1][r2] = 1; V[g14][r2] = 1; W[2*14][r2] = 1
    # Q3 = c * B014 -> C114
    U[2][r3] = 1; V[e14][r3] = 1; W[2*14+1][r3] = 1
    # Q4 = d * B114 -> C114
    U[3][r4] = 1; V[g14][r4] = 1; W[2*14+1][r4] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
