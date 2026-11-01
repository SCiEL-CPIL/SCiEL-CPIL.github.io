"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    The method uses a block-diagonal structure where the 2x10 matrix B is split
    into five 2x2 blocks. Each block multiplication uses the rank-7 Strassen algorithm.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2   # 5 blocks
    
    # Strassen decomposition for 2x2 matrices
    # We define the 7 terms (u, v, w) such that:
    # sum_t u[i,j,t] * v[j,k,t] * w[k,i,t] = (A*B)[i,k]
    # Indices i,j,k are in {0,1}
    
    # Term definitions:
    # 0: (a+d)(e+h) -> u:(0,0)=1,(1,1)=1; v:(0,0)=1,(1,1)=1; w:(0,0)=1,(1,1)=1
    # 1: (c+d)e     -> u:(1,0)=1,(1,1)=1; v:(0,0)=1;       w:(0,1)=1,(1,1)=-1
    # 2: a(f-h)     -> u:(0,0)=1;         v:(0,1)=1,(1,1)=-1; w:(1,0)=1,(1,1)=1
    # 3: d(g-e)     -> u:(1,1)=1;         v:(1,0)=1,(0,0)=-1; w:(0,0)=1,(0,1)=1
    # 4: (a+b)h     -> u:(0,0)=1,(0,1)=1; v:(1,1)=1;        w:(0,0)=-1,(1,0)=1
    # 5: (c-a)(e+f) -> u:(1,0)=1,(0,0)=-1; v:(0,0)=1,(0,1)=1; w:(1,1)=1
    # 6: (b-d)(g+h) -> u:(0,1)=1,(1,1)=-1; v:(1,0)=1,(1,1)=1; w:(0,0)=1
    
    # Stored as lists of (index, value) pairs for efficiency
    # U index: i*m + j
    # V index: j*p + k (local k) -> global k = 2*r + k_local
    # W index: k*n + i (local k) -> global k = 2*r + k_local
    
    strassen_u = [
        [(0, 1), (3, 1)],       # (0,0), (1,1)
        [(2, 1), (3, 1)],       # (1,0), (1,1)
        [(0, 1)],               # (0,0)
        [(3, 1)],               # (1,1)
        [(0, 1), (1, 1)],       # (0,0), (0,1)
        [(2, 1), (0, -1)],      # (1,0), (0,0)
        [(1, 1), (3, -1)]       # (0,1), (1,1)
    ]
    
    strassen_v = [
        [(0, 1), (3, 1)],       # (0,0), (1,1)
        [(0, 1)],               # (0,0)
        [(1, 1), (3, -1)],      # (0,1), (1,1)
        [(2, 1), (0, -1)],      # (1,0), (0,0)
        [(3, 1)],               # (1,1)
        [(0, 1), (1, 1)],       # (0,0), (0,1)
        [(2, 1), (3, 1)]        # (1,0), (1,1)
    ]
    
    strassen_w = [
        [(0, 1), (3, 1)],       # (0,0), (1,1)
        [(1, 1), (3, -1)],      # (0,1), (1,1)
        [(2, 1), (3, 1)],       # (1,0), (1,1)
        [(0, 1), (1, 1)],       # (0,0), (0,1)
        [(0, -1), (2, 1)],      # (0,0), (1,0)
        [(3, 1)],               # (1,1)
        [(0, 1)]                # (0,0)
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            
            # Fill U
            for idx, val in strassen_u[t]:
                U[idx][col] = val
                
            # Fill V
            # V index is j*p + k_global. k_global = 2*r + k_local
            # The stored idx in strassen_v is j*2 + k_local (since m=2)
            for idx, val in strassen_v[t]:
                j = idx // 2
                k_local = idx % 2
                k_global = 2 * r + k_local
                V[j * p + k_global][col] = val
                
            # Fill W
            # W index is k_global*n + i. k_global = 2*r + k_local
            # The stored idx in strassen_w is k_local*2 + i (since n=2)
            for idx, val in strassen_w[t]:
                k_local = idx // 2
                i = idx % 2
                k_global = 2 * r + k_local
                W[k_global * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
