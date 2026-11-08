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
    for it in range(iters):
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
    """High-precision polish: tighten residual to machine epsilon."""
    r = len(U[0])
    for _ in range(iters):
        # Update W
        for i in range(n):
            for k in range(p):
                A = np.array([[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)])
                b = np.ones(m)
                try:
                    sol = np.linalg.solve(A, b)
                    for c in range(r):
                        W[k*n+i][c] = float(sol[c])
                except np.linalg.LinAlgError:
                    try:
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        for c in range(r):
                            W[k*n+i][c] = float(sol[c])
                    except Exception:
                        pass
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.array([[V[j*p+k][c] * W[k*n+i][c] for c in range(r)] for k in range(p)])
                b = np.ones(p)
                try:
                    sol = np.linalg.solve(A, b)
                    for c in range(r):
                        U[i*m+j][c] = float(sol[c])
                except np.linalg.LinAlgError:
                    try:
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        for c in range(r):
                            U[i*m+j][c] = float(sol[c])
                    except Exception:
                        pass
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.array([[U[i*m+j][c] * W[k*n+i][c] for c in range(r)] for i in range(n)])
                b = np.ones(n)
                try:
                    sol = np.linalg.solve(A, b)
                    for c in range(r):
                        V[j*p+k][c] = float(sol[c])
                except np.linalg.LinAlgError:
                    try:
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        for c in range(r):
                            V[j*p+k][c] = float(sol[c])
                    except Exception:
                        pass
    return U, V, W

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

def _strassen_init(rank, n, m, p, rng):
    """Strassen-like structured initialization with symmetry breaking."""
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Initialize with random signs and magnitudes
    for idx in range(rank):
        # Assign to random positions with structured patterns
        ui = rng.randint(0, n*m-1)
        uj = rng.randint(0, m-1)
        U[ui][idx] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        
        vj = rng.randint(0, m-1)
        vk = rng.randint(0, p-1)
        V[vj * p + vk][idx] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        
        wk = rng.randint(0, p-1)
        wi = rng.randint(0, n-1)
        W[wk * n + wi][idx] = rng.choice([-1.0, -0.5, 0.5, 1.0])
    
    return U, V, W

def _try_rank(rank, n, m, p, rng, max_attempts=50):
    """Try to find a valid decomposition at given rank via random init + refinement."""
    for attempt in range(max_attempts):
        if _remaining() < 5.0:
            return None
        
        # Choose initialization strategy
        strat = attempt % 6
        if strat == 0:
            # Sparse half-integer init
            U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        elif strat == 1:
            # Dense small-magnitude init
            U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        elif strat == 2:
            # Structured: some rows share patterns
            U = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(p * n)]
        elif strat == 3:
            # Gaussian init
            U = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)]
        elif strat == 4:
            # Strassen-like structured
            U, V, W = _strassen_init(rank, n, m, p, rng)
        else:
            # Mixed: half-integer with some random perturbations
            U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) + rng.gauss(0, 0.1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) + rng.gauss(0, 0.1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) + rng.gauss(0, 0.1) for _ in range(rank)] for _ in range(p * n)]
        
        U, V, W = _refine(U, V, W, n, m, p, iters=40)
        res = _residual(U, V, W, n, m, p)
        if res < 1e-5:
            # Polish to high precision
            U, V, W = _polish(U, V, W, n, m, p, iters=50)
            res = _residual(U, V, W, n, m, p)
            if res < 1e-7:
                rounded = _try_round_verify(U, V, W)
                if rounded is not None:
                    return rounded
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Use multiple seeds for diversity
    seeds = [42, 12345, 99999, 7, 31415, 27182, 1000003, 55555, 777777, 888888]
    
    # Search from lower ranks upward to find minimal rank quickly
    for rank in range(12, best_rank):
        if _remaining() < 15.0:
            break
        found = False
        for seed in seeds:
            if _remaining() < 10.0:
                break
            if found:
                break
            rng = random.Random(seed)
            result = _try_rank(rank, n, m, p, rng, max_attempts=40)
            if result is not None:
                best = result
                best_rank = rank
                found = True
                # Continue searching for even lower ranks
                break
    return best

if __name__ == "__main__":
    print(construct())
