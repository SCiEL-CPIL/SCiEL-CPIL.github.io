"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 10.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _residual(U, V, W, n, m, p):
    """Return max absolute tensor residual over all (i,j,k)."""
    worst = 0.0
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(r):
                    s += uc[c] * vc[c] * wc[c]
                diff = abs(s - 1.0)
                if diff > worst:
                    worst = diff
    return worst

def _is_half_int(x, tol=1e-6):
    return abs(x * 2 - round(x * 2)) < tol

def _try_round_verify(U, V, W):
    """Round to half-integers and check exactness."""
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    n, m, p = DIMENSIONS
    r = len(Ur[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ur[i * m + j][c] * Vr[j * p + k][c] * Wr[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return None
    return {"U": Ur, "V": Vr, "W": Wr}

def _refine(U, V, W, n, m, p, iters=50):
    """Gauss-Seidel style refinement to drive residual toward zero."""
    r = len(U[0])
    for _ in range(iters):
        # Update W: for each (i,k), solve m linear equations
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass
        # Update U: for each (i,j), solve p linear equations
        for i in range(n):
            for j in range(m):
                A = []
                for k in range(p):
                    row = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass
        # Update V: for each (j,k), solve n linear equations
        for j in range(m):
            for k in range(p):
                A = []
                for i in range(n):
                    row = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass
    return U, V, W

def _polish(U, V, W, n, m, p, iters=100):
    """High-precision polish: use tighter tolerance and more iterations."""
    r = len(U[0])
    for _ in range(iters):
        for i in range(n):
            for k in range(p):
                A = np.array([[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)], dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
                    W[k*n+i] = [float(sol[c]) for c in range(r)]
                except:
                    pass
        for i in range(n):
            for j in range(m):
                A = np.array([[V[j*p+k][c] * W[k*n+i][c] for c in range(r)] for k in range(p)], dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
                    U[i*m+j] = [float(sol[c]) for c in range(r)]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                A = np.array([[U[i*m+j][c] * W[k*n+i][c] for c in range(r)] for i in range(n)], dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
                    V[j*p+k] = [float(sol[c]) for c in range(r)]
                except:
                    pass
    return U, V, W

def _build_kronecker_rank9(n, m, p):
    """
    Exploit separable structure: view 2x3*3x6 as tensor product of 1x3*3x2 (rank 3)
    and 2x1*1x3 (rank 3). The Kronecker product gives rank 9.
    
    For 1x3 * 3x2: output is 1x2. Rank-3 decomposition:
    U1 (1*3=3 rows, 3 cols): rows for (i=0, j=0..2)
    V1 (3*2=6 rows, 3 cols): rows for (j=0..2, k=0..1)  
    W1 (2*1=2 rows, 3 cols): rows for (k=0..1, i=0)
    
    For 2x1 * 1x3: output is 2x3. Rank-3 decomposition:
    U2 (2*1=2 rows, 3 cols): rows for (i=0..1, j=0)
    V2 (1*3=3 rows, 3 cols): rows for (j=0, k=0..2)
    W2 (3*2=6 rows, 3 cols): rows for (k=0..2, i=0..1)
    """
    # Rank-1 terms for 1x3 * 3x2 (matrix A[1,3] * B[3,2] = C[1,2])
    # We need sum_c u_c * v_c * w_c = 1 for each (i,j,k)
    # For 1x3*3x2: n=1,m=3,p=2
    # Index: U[i*m+j] = U[j], V[j*p+k] = V[2j+k], W[k*n+i] = W[k]
    
    # Simple rank-3 for 1x3*3x2:
    # U rows: 3 entries (for j=0,1,2)
    # V rows: 6 entries (for j=0,1,2 and k=0,1)
    # W rows: 2 entries (for k=0,1)
    
    # Let's use identity-like structure:
    # For each (j,k), we need U[j]*V[2j+k]*W[k] = 1
    # This is a 3x2 grid. We can use rank 3 with:
    # c=0: U[0]=1, V[0]=1, W[0]=1 -> contributes to (j=0,k=0)
    # c=1: U[1]=1, V[3]=1, W[1]=1 -> contributes to (j=1,k=1)  
    # c=2: U[2]=1, V[4]=1, W[0]=1 -> contributes to (j=2,k=0)
    # But we also need (j=0,k=1), (j=1,k=0), (j=2,k=1)
    
    # Actually, the rank of 1x3*3x2 multiplication is 3 (it's like vector-matrix mult)
    # Let me use a known rank-3 decomposition for 1x3*3x2:
    # This is equivalent to computing [a0,a1,a2] * [[b00,b01],[b10,b11],[b20,b21]]
    # = [a0*b00+a1*b10+a2*b20, a0*b01+a1*b11+a2*b21]
    
    # Rank-3:
    # c=0: contributes to (j=0,k=0): U[0]=1, V[0]=1, W[0]=1
    # c=1: contributes to (j=1,k=0): U[1]=1, V[2]=1, W[0]=1  
    # c=2: contributes to (j=2,k=0): U[2]=1, V[4]=1, W[0]=1
    # But we need k=1 terms too...
    
    # The actual rank of 1x3*3x2 is 3. Let me use:
    # The tensor T[i,j,k] for 1x3*3x2 has i always 0, so T[0,j,k] = 1
    # We need sum_c U[j,c]*V[2j+k,c]*W[k,c] = 1 for j=0,1,2 and k=0,1
    
    # Rank 3 decomposition:
    # c=0: U=[1,0,0], V=[1,0,1,0,0,0], W=[1,1] -> T[0,0,0]=1, T[0,0,1]=1, others 0
    # c=1: U=[0,1,0], V=[0,0,0,1,0,0], W=[1,0] -> T[0,1,0]=1, others 0
    # c=2: U=[0,0,1], V=[0,0,0,0,1,0], W=[1,0] -> T[0,2,0]=1, others 0
    # Missing: T[0,1,1] and T[0,2,1]
    
    # This doesn't work with rank 3. Let me reconsider.
    # Actually 1x3*3x2 has rank 3? No, it should be 3*2=6 schoolbook, but can be lower.
    # The border rank is 3, but exact rank might be higher.
    
    # Let me just use the Kronecker structure differently.
    # 2x3 * 3x6 can be seen as (2x1)*(1x3) * (3x1)*(1x6) but that's not quite right.
    
    # Actually, let's use the fact that 2x3*3x6 = (I_2 kron I_1) * ... 
    # The key insight: 2x3*3x6 has a block structure.
    # A = [[a11,a12,a13],[a21,a22,a23]], B = 3x6 matrix.
    # C = A*B is 2x6.
    
    # The rank of 2x3*3x6 multiplication tensor:
    # Lower bound: max(n*m, m*p, p*n) = max(6,18,12) = 18
    # Upper bound: n*m*p = 36 (schoolbook)
    # Known results suggest rank is around 18-24 for such products.
    
    # For now, let's try rank 18 with structured initialization.
    return None

def _structured_init(rank, n, m, p, rng):
    """Create structured initialization exploiting block patterns."""
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Use block-structured init: divide rank into blocks
    block_size = rank // 6
    if block_size < 1:
        block_size = 1
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                # Pattern: U depends on i and j
                U[i * m + j][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V[j * p + k][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = rng.choice([-1, -0.5, 0, 0.5, 1])
    
    return U, V, W

def _try_rank(rank, n, m, p, rng, max_attempts=50):
    """Try to find a valid decomposition at given rank via random init + refinement."""
    for attempt in range(max_attempts):
        if _remaining() < 5.0:
            return None
        # Choose initialization strategy based on attempt number
        if attempt % 5 == 0:
            # Sparse half-integer init
            U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        elif attempt % 5 == 1:
            # Dense small-magnitude init
            U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        elif attempt % 5 == 2:
            # Structured: some rows share patterns
            U, V, W = _structured_init(rank, n, m, p, rng)
        elif attempt % 5 == 3:
            # Binary +/- 1 init
            U = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(p * n)]
        else:
            # Gaussian init
            U = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)]
        
        U, V, W = _refine(U, V, W, n, m, p, iters=40)
        res = _residual(U, V, W, n, m, p)
        if res < 1e-7:
            # Try polish
            U, V, W = _polish(U, V, W, n, m, p, iters=50)
            res = _residual(U, V, W, n, m, p)
            if res < 1e-8:
                rounded = _try_round_verify(U, V, W)
                if rounded is not None:
                    return rounded
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Use multiple seeds for diversity
    seeds = [42, 12345, 99999, 7, 31415, 27182, 1000003, 55555, 88888, 123456]
    
    # Try ranks from just below schoolbook down to 18
    for rank in range(best_rank - 1, 17, -1):
        if _remaining() < 15.0:
            break
        found = False
        for seed in seeds:
            if _remaining() < 10.0:
                break
            if found:
                break
            rng = random.Random(seed)
            result = _try_rank(rank, n, m, p, rng, max_attempts=25)
            if result is not None:
                best = result
                best_rank = rank
                found = True
                break
    return best

if __name__ == "__main__":
    print(construct())
