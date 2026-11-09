"""Optimized algorithm: exact rank minimization for 2x3x5 matrix multiplication tensor.
Uses integer arithmetic for speed. Attempts rank 28 via random search, falls back to rank 29."""
import time
import random

DIMENSIONS = [2, 3, 5]

def _check_int(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(rank):
                    a = uc[c]
                    if a:
                        s += a * vc[c] * wc[c]
                if s != 1:
                    return False
    return True

def _to_half(M):
    """Converts list of ints to list of ints or halves (floats)."""
    out = []
    for row in M:
        r = []
        for v in row:
            if v % 2 == 0:
                r.append(v // 2)
            else:
                r.append(v / 2.0)
        out.append(r)
    return out

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100 # Safety margin

    # 1. Attempt rank 28 with random search (limited time)
    # This is a long shot, but if it works, score > 1.0
    # We use a very small search space: entries in {-1, 0, 1}
    # To make it slightly more likely, we can try to fix some structures.
    # But for now, pure random.
    found_28 = None
    start_time = time.time()
    max_iters = 500
    for _ in range(max_iters):
        if time.time() - start_time > 10: # 10s budget for rank 28 search
            break
        rank = 28
        U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        if _check_int(U, V, W, n, m, p):
            found_28 = (U, V, W)
            break

    if found_28:
        U, V, W = found_28
        # Scale by 1/2 if needed? No, check_int expects 1.
        # The output requires integer/half-integer.
        # If entries are -1,0,1, they are integers.
        # So we can return them directly, or pass through _to_half if we want to be safe.
        # _to_half on int v: if v%2==0, v//2. If v=1, 1/2=0.5.
        # But the decomposition must sum to 1.
        # If we found a valid int decomposition, we can just return the ints.
        # The contract says "integer/half-integer entries". Ints are allowed.
        return {"U": U, "V": V, "W": W}

    # 2. Fallback to known rank 29 decomposition
    # Bini 2x2x2 rank 7
    bini = [
        ( 1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0),
        ( 0, 1, 0, 0,  0, 1, 0, 0,  0, 1, 0, 0),
        ( 1, 0, 0, 1,  1, 0, 0, 0,  0, 0, 1, 0),
        ( 0, 1, 1, 0,  0, 1, 0, 0,  0, 0, 0, 1),
        ( 1, 1, 0, 0,  1, 1, 0, 0,  1, 0, 0, 0),
        ( 0, 0, 1, 1,  0, 0, 1, 1,  0, 1, 0, 0),
        ( 1, 0, 1, 1,  0, 1, 1, 0,  1, 1, 0, 0),
    ]
    
    rank = 29
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    
    # Block A: 2x2x2 via Bini
    for (u00,u01,u10,u11,v00,v01,v10,v11,w00,w01,w10,w11) in bini:
        for i in range(2):
            U[i*m+0][col] = (u00,u01,u10,u11)[i*2+0]
            U[i*m+1][col] = (u00,u01,u10,u11)[i*2+1]
        for j in range(2):
            V[j*p+0][col] = (v00,v01,v10,v11)[j*2+0]
            V[j*p+1][col] = (v00,v01,v10,v11)[j*2+1]
        for k in range(2):
            W[k*n+0][col] = (w00,w01,w10,w11)[k*2+0]
            W[k*n+1][col] = (w00,w01,w10,w11)[k*2+1]
        col += 1
        
    # Block B: j=2, all k
    for i in range(n):
        for k in range(p):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # Block C: j in {0,1}, k in {2,3,4}
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    assert col == rank
    
    # Verify
    if _check_int(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
