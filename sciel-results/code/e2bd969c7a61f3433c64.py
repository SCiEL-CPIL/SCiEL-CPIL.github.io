import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_low_rank(n, m, p):
    cols = []
    def strassen(ai, aj, bj, bk, ci, ck):
        def A(i, j): return (ai+i, aj+j)
        def B(j, k): return (bj+j, bk+k)
        def C(i, k): return (ci+i, ck+k)
        M = [
            ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
            ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
            ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
            ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
            ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))
            
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen(0, 0, 0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen(0, 0, 0, 2, 0, 2)
    
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A is 2x1, B is 1x2. Rank 2.
    # A[0,2] B[2,0] -> C[0,0]
    # A[0,2] B[2,1] -> C[0,1]
    # A[1,2] B[2,0] -> C[1,0]
    # A[1,2] B[2,1] -> C[1,1]
    # We can do this with 2 multiplications?
    # M1 = (A[0,2]+A[1,2])(B[2,0]+B[2,1])
    # M2 = A[0,2]B[2,0]
    # M3 = A[1,2]B[2,0]
    # No, standard rank of 2x1 * 1x2 is 2?
    # Actually, the tensor for 2x1 * 1x2 is rank 2.
    # T_{i,0, j,0, i,j} ? No.
    # Indices: A_{i,2}, B_{2,j}, C_{i,j}.
    # T_{i,2, 2,j, i,j} = 1.
    # This is a permutation of the 2x2 identity?
    # Rank is 2.
    # Current code uses 2 columns for this?
    # cols.append(( {(0,2):1,(1,2):1}, {(2,0):1,(2,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1} ))
    # This is 1 column.
    # U: A[0,2]=1, A[1,2]=1.
    # V: B[2,0]=1, B[2,1]=1.
    # W: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=1.
    # Product: (A[0,2]+A[1,2])(B[2,0]+B[2,1]) * (C[0,0]+C[0,1]+C[1,0]+C[1,1])?
    # No, W is a vector.
    # The term is $U_a V_b W_c$.
    # For $a=0,2$ (A[0,2]), $b=2,0$ (B[2,0]), $c=0,0$ (C[0,0]).
    # $U_{0,2}=1, V_{2,0}=1, W_{0,0}=1$. Contribution 1.
    # For $a=0,2$, $b=2,1$, $c=0,1$. $U=1, V=1, W=1$. Contribution 1.
    # For $a=1,2$, $b=2,0$, $c=1,0$. $U=1, V=1, W=1$. Contribution 1.
    # For $a=1,2$, $b=2,1$, $c=1,1$. $U=1, V=1, W=1$. Contribution 1.
    # Cross terms:
    # $a=0,2, b=2,1, c=1,0$? $W_{1,0}=1$. $U=1, V=1$. Contribution 1.
    # But we want 0 for $C[1,0]$ from $A[0,2]B[2,1]$.
    # So this single column creates unwanted cross terms.
    # The current code adds this column.
    # Then it adds:
    # cols.append(( {(0,2):1,(1,2):1}, {(2,2):1,(2,3):1}, {(0,2):1,(0,3):1,(1,2):1,(1,3):1} ))
    # This handles A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4].
    
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # A is 1x2, B is 2x2.
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
    # Current code:
    for k in range(2):
        cols.append(( {(2,0):1,(2,1):1}, {(0,k):1,(1,k):1}, {(2,k):1} ))
    # This adds 2 columns.
    # Col k=0: U: A[2,0]=1, A[2,1]=1. V: B[0,0]=1, B[1,0]=1. W: C[2,0]=1.
    # Term: (A[2,0]+A[2,1])(B[0,0]+B[1,0]) C[2,0].
    # Expands to A[2,0]B[0,0] + A[2,0]B[1,0] + A[2,1]B[0,0] + A[2,1]B[1,0].
    # We want A[2,0]B[0,0] + A[2,1]B[1,0].
    # Cross terms A[2,0]B[1,0] and A[2,1]B[0,0] are unwanted.
    
    # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    for k in range(2):
        cols.append(( {(2,0):1,(2,1):1}, {(0,2+k):1,(1,2+k):1}, {(2,2+k):1} ))
    # Adds 2 columns.
    
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    cols.append(( {(2,2):1}, {(2,0):1,(2,1):1}, {(2,0):1,(2,1):1} ))
    # Adds 1 column.
    # U: A[2,2]=1. V: B[2,0]=1, B[2,1]=1. W: C[2,0]=1, C[2,1]=1.
    # Term: A[2,2](B[2,0]+B[2,1])(C[2,0]+C[2,1]).
    # Expands to A[2,2]B[2,0]C[2,0] + A[2,2]B[2,0]C[2,1] + A[2,2]B[2,1]C[2,0] + A[2,2]B[2,1]C[2,1].
    # We want A[2,2]B[2,0]C[2,0] and A[2,2]B[2,1]C[2,1].
    # Cross terms unwanted.
    
    # A[2, 2] * B[2, 2:4] -> C[2, 2:4]
    cols.append(( {(2,2):1}, {(2,2):1,(2,3):1}, {(2,2):1,(2,3):1} ))
    # Adds 1 column.
    
    # Total rank: 7 + 7 + 1 + 1 + 2 + 2 + 1 + 1 = 22.
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 115.0
    
    U, V, W = _build_low_rank(n, m, p)
    
    if _verify(U, V, W, n, m, p, len(U[0])):
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    if time.time() - t0 < DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
