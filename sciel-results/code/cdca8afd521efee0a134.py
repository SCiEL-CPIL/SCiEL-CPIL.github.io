"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Block decomposition:
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices in U: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices in V: j*p + k  (row j, column k)
    # C indices in W: k*n + i  (row k, column i)
    
    def add_strassen_pair(base_r, k0, k1):
        """Add a rank-7 Strassen decomposition for a 2x2x2 block on columns k0, k1."""
        r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
        
        # P1 = (A00+A11) * (B0k0+B1k1)
        U[0][r1] = 1; U[3][r1] = 1
        V[0*p+k0][r1] = 1; V[1*p+k1][r1] = 1
        # P2 = (A10+A11) * B0k0
        U[2][r2] = 1; U[3][r2] = 1
        V[0*p+k0][r2] = 1
        # P3 = A00 * (B0k1 - B1k1)
        U[0][r3] = 1
        V[0*p+k1][r3] = 1; V[1*p+k1][r3] = -1
        # P4 = A11 * (B1k0 - B0k0)
        U[3][r4] = 1
        V[1*p+k0][r4] = 1; V[0*p+k0][r4] = -1
        # P5 = (A00+A01) * B1k1
        U[0][r5] = 1; U[1][r5] = 1
        V[1*p+k1][r5] = 1
        # P6 = (A10-A00) * (B0k0+B0k1)
        U[2][r6] = 1; U[0][r6] = -1
        V[0*p+k0][r6] = 1; V[0*p+k1][r6] = 1
        # P7 = (A01-A11) * (B1k0+B1k1)
        U[1][r7] = 1; U[3][r7] = -1
        V[1*p+k0][r7] = 1; V[1*p+k1][r7] = 1
        
        # Reconstruction: C[k][i] = sum_r W[k*n+i][r] * (U row dot V col)
        # C[0][k0] = P1 + P4 - P5 + P7
        W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
        # C[1][k0] = P2 + P4
        W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
        # C[0][k1] = P3 + P5
        W[2*k1][r3] = 1; W[2*k1][r5] = 1
        # C[1][k1] = P1 - P2 + P3 + P6
        W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        add_strassen_pair(7*i, 2*i, 2*i+1)
    
    # Trivial schoolbook block for column 14 (rank 4)
    base_r = 49
    U[0][base_r]     = 1; V[0*p+14][base_r]     = 1; W[2*14][base_r]     = 1
    U[1][base_r + 1] = 1; V[1*p+14][base_r + 1] = 1; W[2*14][base_r+1] = 1
    U[2][base_r + 2] = 1; V[0*p+14][base_r + 2] = 1; W[2*14+1][base_r+2] = 1
    U[3][base_r + 3] = 1; V[1*p+14][base_r + 3] = 1; W[2*14+1][base_r+3] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
