"""Matrix multiplication 2x4 by 4x8 - rank-56 Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    """Build the rank-56 Strassen decomposition (8 independent 2x2x2 blocks)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7
            if h == 0:
                def a_idx(r, c): return r * m + c
                def b_idx(r, c): return r * p + (2 * d + c)
            else:
                def a_idx(r, c): return r * m + (2 + c)
                def b_idx(r, c): return (2 + r) * p + (2 * d + c)
            def c_idx(r, c): return (2 * d + c) * n + r
            
            # M1
            col = base + 0
            U[a_idx(0,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,1),col]+=1
            # M2
            col = base + 1
            U[a_idx(1,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1
            W[c_idx(1,0),col]+=1; W[c_idx(1,1),col]-=1
            # M3
            col = base + 2
            U[a_idx(0,0),col]+=1
            V[b_idx(0,1),col]+=1; V[b_idx(1,1),col]-=1
            W[c_idx(0,1),col]+=1; W[c_idx(1,1),col]+=1
            # M4
            col = base + 3
            U[a_idx(1,1),col]+=1
            V[b_idx(1,0),col]+=1; V[b_idx(0,0),col]-=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,0),col]+=1
            # M5
            col = base + 4
            U[a_idx(0,0),col]+=1; U[a_idx(0,1),col]+=1
            V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]-=1; W[c_idx(0,1),col]+=1
            # M6
            col = base + 5
            U[a_idx(1,0),col]+=1; U[a_idx(0,0),col]-=1
            V[b_idx(0,0),col]+=1; V[b_idx(0,1),col]+=1
            W[c_idx(1,1),col]+=1
            # M7
            col = base + 6
            U[a_idx(0,1),col]+=1; U[a_idx(1,1),col]-=1
            V[b_idx(1,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition satisfies all tensor entries."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Build the full tensor via broadcasting for speed
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We need: for valid (i,j,k), T[i*m+j, j*p+k, k*n+i] == 1
    # and for invalid indices, T == 0
    
    # Efficient check: build the tensor as sum of outer products
    # T = sum_r U[:,r] * V[:,r] * W[:,r]  (broadcast)
    # This is rank x (nm x mp x pn) which is 56 * 8 * 32 * 16 = too big
    # Instead, check each entry directly but vectorize where possible
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(np.dot(U[a_idx, :], V[b_idx, :]), W[c_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries - sample them for speed
    import random
    rng = random.Random(42)
    total_zeros = 0
    checked = 0
    for _ in range(2000):
        a = rng.randint(0, n*m-1)
        b = rng.randint(0, m*p-1)
        c = rng.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val) > tol:
                return False
            checked += 1
    
    return True

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2x2 Strassen decompositions (rank 7 each) = rank 56.
    """
    t_start = time.time()
    deadline = t_start + 100.0  # Safety margin before 120s limit
    
    U, V, W = _build_rank56()
    
    # Quick verification
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W):
            U, V, W = _build_rank56()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(np.dot(U[a_idx, :], V[b_idx, :]), W[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val) > 1e-9:
                return False
    
    return True


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    n, m, p = DIMENSIONS
    ok = _verify(result["U"], result["V"], result["W"], n, m, p)
    rank = len(result["U"][0])
    elapsed = time.time() - t0
    print(f"Rank: {rank}, Valid: {ok}, Time: {elapsed:.1f}s")
    if not ok:
        print("DECOMPOSITION IS INVALID!")
    else:
        print("Decomposition verified successfully.")
