"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,10) sub-block."""
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Split j into {0,1} (Strassen per k-pair) and j=2 (schoolbook)
    # (2,2,10): 5 k-pairs × 7 Strassen = 35
    # (2,1,10): 2×10 = 20
    # Total: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    
    # Strassen coefficients:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # W coefficients per Strassen term (local i, local k → global k = pair*2 + local_k)
    # W_idx = (global_k) * n + i = (pair*2 + lk) * 2 + li
    
    strassen_terms = [
        # (M_idx, U_entries, V_entries, W_entries)
        # M1: (A00+A11)(B00+B11)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        # W: C[0,0]+=1, C[1,1]+=1
        {
            'U': [(0, 1), (4, 1)],       # i*m+j: 0*3+0=0, 1*3+1=4
            'V': [(0, 1), (10, 1)],      # j*p+k: 0*10+0=0, 1*10+1=11? No wait...
        },
    ]
    
    # Let me be more explicit. For pair with global k0, k1:
    # Local (i,j,k) → global: i stays, j stays (0 or 1), k: local 0→k0, local 1→k1
    # U row: i*3+j
    # V row: j*10+k_global
    # W row: k_global*2+i
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: row 0*3+0=0 → 1, row 1*3+1=4 → 1
        # V: row 0*10+k0=k0 → 1, row 1*10+k1=10+k1 → 1
        # W: row k0*2+0=2*k0 → +1, row k1*2+1=2*k1+1 → +1
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[10+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # U: row 1*3+0=3 → 1, row 1*3+1=4 → 1
        # V: row 0*10+k0=k0 → 1
        # W: row k0*2+1=2*k0+1 → +1, row k1*2+1=2*k1+1 → -1
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        # U: row 0*3+0=0 → 1
        # V: row 0*10+k1=10+k1 → 1, row 1*10+k1=10+k1 → -1
        # W: row k1*2+0=2*k1 → +1, row k1*2+1=2*k1+1 → +1
        U[0][col] = 1
        V[10+k1][col] = 1; V[10+k1][col] += -1  # Wait, both are row 10+k1!
        
        # Oops! B[0,k1] has row 0*10+k1 = k1, B[1,k1] has row 1*10+k1 = 10+k1
        # These are different rows! Let me redo.
        # V row for B[j,k]: j*p + k = j*10 + k
        # B[0,k1]: row 0*10+k1 = k1
        # B[1,k1]: row 1*10+k1 = 10+k1
        
        # I made an error above. Let me restart more carefully.
    
    # Restart with correct indexing
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1   # A[0,0], row 0*3+0
        U[4][col] = 1   # A[1,1], row 1*3+1
        V[k0][col] = 1       # B[0,k0], row 0*10+k0
        V[10+k1][col] = 1    # B[1,k1], row 1*10+k1
        W[2*k0][col] = 1     # C[0,k0], row k0*2+0
        W[2*k1+1][col] = 1   # C[1,k1], row k1*2+1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3][col] = 1   # A[1,0], row 1*3+0
        U[4][col] = 1   # A[1,1], row 1*3+1
        V[k0][col] = 1  # B[0,k0], row 0*10+k0
        W[2*k0+1][col] = 1    # C[1,k0], row k0*2+1
        W[2*k1+1][col] = -1   # C[1,k1], row k1*2+1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0][col] = 1   # A[0,0], row 0*3+0
        V[k1][col] = 1       # B[0,k1], row 0*10+k1
        V[10+k1][col] = -1   # B[1,k1], row 1*10+k1
        W[2*k1][col] = 1     # C[0,k1], row k1*2+0
        W[2*k1+1][col] = 1   # C[1,k1], row k1*2+1
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4][col] = 1   # A[1,1], row 1*3+1
        V[10+k0][col] = 1   # B[1,k0], row 1*10+k0
        V[k0][col] = -1     # B[0,k0], row 0*10+k0
        W[2*k0][col] = 1    # C[0,k0], row k0*2+0
        W[2*k0+1][col] = 1  # C[1,k0], row k0*2+1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1   # A[0,0], row 0*3+0
        U[1][col] = 1   # A[0,1], row 0*3+1
        V[10+k1][col] = 1  # B[1,k1], row 1*10+k1
        W[2*k0][col] = -1  # C[0,k0], row k0*2+0
        W[2*k1][col] = 1   # C[0,k1], row k1*2+0
        col += 1
        
        # M6 = (A[1,1]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[4][col] = 1   # A[1,1], row 1*3+1
        U[0][col] = -1  # A[0,0], row 0*3+0
        V[k0][col] = 1     # B[0,k0], row 0*10+k0
        V[k1][col] = 1     # B[0,k1], row 0*10+k1
        W[2*k1+1][col] = 1  # C[1,k1], row k1*2+1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1   # A[0,1], row 0*3+1
        U[4][col] = -1  # A[1,1], row 1*3+1
        V[10+k0][col] = 1  # B[1,k0], row 1*10+k0
        V[10+k1][col] = 1  # B[1,k1], row 1*10+k1
        W[2*k0][col] = 1    # C[0,k0], row k0*2+0
        W[2*k1+1][col] = -1 # C[1,k1], row k1*2+1
        col += 1
    
    # Part 2: Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1   # A[i,2], row i*3+2
            V[20+k][col] = 1    # B[2,k], row 2*10+k
            W[k*2+i][col] = 1   # C[i,k], row k*2+i
            col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify
    import random
    random.seed(42)
    def verify(U, V, W, n, m, p, max_checks=10):
        rank = len(U[0])
        for _ in range(max_checks):
            A = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
            B = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
            C = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C[i][k] += A[i][j] * B[j][k]
            Cc = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                        vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                        acc += W[k*n+i][r] * ua * vb
                    Cc[i][k] = acc
            for i in range(n):
                for k in range(p):
                    if abs(C[i][k] - Cc[i][k]) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
