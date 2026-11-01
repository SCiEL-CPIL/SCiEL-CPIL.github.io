import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Strategy: Split B's 15 columns into 7 pairs + 1 single.
    # For each pair of columns (k1, k2), use Strassen's algorithm (rank 7) for 2x2 * 2x2.
    # For the single column, use schoolbook (rank 4).
    # Total rank: 7*7 + 4 = 53
    
    # This exploits the fact that tensor rank is additive over disjoint index sets.
    # The 2x2*2x2 multiplication tensor has rank 7 (Strassen), not 4 (schoolbook).
    # By grouping columns in pairs, we apply Strassen to each 2x2*2x2 sub-problem.
    
    rank = 53
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(30)]
    
    col = 0
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
    
    for k1, k2 in pairs:
        # Strassen 7 terms for columns k1, k2
        # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        # B indices: k1=B[0,k1], k2=B[0,k2], 15+k1=B[1,k1], 15+k2=B[1,k2]
        # C indices: k1=C[0,k1], k2=C[0,k2], 15+k1=C[1,k1], 15+k2=C[1,k2]
        
        # Standard Strassen for C = A*B where:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a-d)*(g+h)
        # M2 = (a+b)*g
        # M3 = d*(f-h)
        # M4 = a*e
        # M5 = (b+d)*(e+h)
        # M6 = (a-b)*f
        # M7 = (c+a)*(h-e)
        # C11 = M4 + M5 - M1 - M7
        # C12 = M2 + M6
        # C21 = M3 + M5
        # C22 = M1 + M4 - M2 + M7
        
        # In our notation:
        # a=A[0,0] (idx 0), b=A[0,1] (idx 1), c=A[1,0] (idx 2), d=A[1,1] (idx 3)
        # e=B[0,k1] (idx k1), f=B[0,k2] (idx k2)
        # g=B[1,k1] (idx 15+k1), h=B[1,k2] (idx 15+k2)
        # C[0,k1] (idx k1), C[0,k2] (idx k2)
        # C[1,k1] (idx 15+k1), C[1,k2] (idx 15+k2)
        
        # M1 = (a-d)*(g+h) = (A[0,0]-A[1,1])*(B[1,k1]+B[1,k2])
        # U: A[0,0] +1, A[1,1] -1
        # V: B[1,k1] +1, B[1,k2] +1
        # W: contributes to C[0,k1] with -1, C[1,k2] with +1
        U[0][col] += 1; U[3][col] += -1
        V[15+k1][col] += 1; V[15+k2][col] += 1
        W[k1][col] += -1; W[15+k2][col] += 1
        col += 1
        
        # M2 = (a+b)*g = (A[0,0]+A[0,1])*B[1,k1]
        # U: A[0,0] +1, A[0,1] +1
        # V: B[1,k1] +1
        # W: contributes to C[0,k2] with +1, C[1,k2] with -1
        U[0][col] += 1; U[1][col] += 1
        V[15+k1][col] += 1
        W[k2][col] += 1; W[15+k2][col] += -1
        col += 1
        
        # M3 = d*(f-h) = A[1,1]*(B[0,k2]-B[1,k2])
        # U: A[1,1] +1
        # V: B[0,k2] +1, B[1,k2] -1
        # W: contributes to C[1,k1] with +1
        U[3][col] += 1
        V[k2][col] += 1; V[15+k2][col] += -1
        W[15+k1][col] += 1
        col += 1
        
        # M4 = a*e = A[0,0]*B[0,k1]
        # U: A[0,0] +1
        # V: B[0,k1] +1
        # W: contributes to C[0,k1] with +1, C[1,k2] with +1
        U[0][col] += 1
        V[k1][col] += 1
        W[k1][col] += 1; W[15+k2][col] += 1
        col += 1
        
        # M5 = (b+d)*(e+h) = (A[0,1]+A[1,1])*(B[0,k1]+B[1,k2])
        # U: A[0,1] +1, A[1,1] +1
        # V: B[0,k1] +1, B[1,k2] +1
        # W: contributes to C[0,k1] with +1, C[1,k1] with +1
        U[1][col] += 1; U[3][col] += 1
        V[k1][col] += 1; V[15+k2][col] += 1
        W[k1][col] += 1; W[15+k1][col] += 1
        col += 1
        
        # M6 = (a-b)*f = (A[0,0]-A[0,1])*B[0,k2]
        # U: A[0,0] +1, A[0,1] -1
        # V: B[0,k2] +1
        # W: contributes to C[0,k2] with +1
        U[0][col] += 1; U[1][col] += -1
        V[k2][col] += 1
        W[k2][col] += 1
        col += 1
        
        # M7 = (c+a)*(h-e) = (A[1,0]+A[0,0])*(B[1,k2]-B[0,k1])
        # U: A[1,0] +1, A[0,0] +1
        # V: B[1,k2] +1, B[0,k1] -1
        # W: contributes to C[0,k1] with -1, C[1,k2] with +1
        U[2][col] += 1; U[0][col] += 1
        V[15+k2][col] += 1; V[k1][col] += -1
        W[k1][col] += -1; W[15+k2][col] += 1
        col += 1
    
    # Schoolbook for column 14
    k = 14
    # C[0,14] = A[0,0]*B[0,14] + A[0,1]*B[1,14]
    # C[1,14] = A[1,0]*B[0,14] + A[1,1]*B[1,14]
    
    # Term: A[0,0]*B[0,14] -> C[0,14]
    U[0][col] += 1; V[k][col] += 1; W[k][col] += 1
    col += 1
    
    # Term: A[0,1]*B[1,14] -> C[0,14]
    U[1][col] += 1; V[15+k][col] += 1; W[k][col] += 1
    col += 1
    
    # Term: A[1,0]*B[0,14] -> C[1,14]
    U[2][col] += 1; V[k][col] += 1; W[15+k][col] += 1
    col += 1
    
    # Term: A[1,1]*B[1,14] -> C[1,14]
    U[3][col] += 1; V[15+k][col] += 1; W[15+k][col] += 1
    col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Verify numerically
    if time.time() < deadline - 5:
        T = np.zeros((4, 30, 30), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    T[i*2+j, j*15+k, i*15+k] = 1
        
        # Reconstruct
        T_recon = np.zeros((4, 30, 30), dtype=float)
        for r in range(rank):
            ua = np.array([U[a][r] for a in range(4)], dtype=float)
            vb = np.array([V[b][r] for b in range(30)], dtype=float)
            wc = np.array([W[c][r] for c in range(30)], dtype=float)
            T_recon += np.einsum('a,b,c->abc', ua, vb, wc)
        
        max_err = np.max(np.abs(T - T_recon))
        if max_err > 1e-10:
            # Fallback to schoolbook rank 60
            rank = 60
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(30)]
            W = [[0]*rank for _ in range(30)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(15):
                        U[i*2+j][col] = 1
                        V[j*15+k][col] = 1
                        W[i*15+k][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
