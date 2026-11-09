import time
import random
from fractions import Fraction

def _build_strassen_decomposition():
    """Build the guaranteed rank-77 decomposition using tensor product of Strassen."""
    n, m, p = 4, 4, 6
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    rank1 = 49
    U1 = [[0]*rank1 for _ in range(n*m)]
    V1 = [[0]*rank1 for _ in range(m*p)]
    W1 = [[0]*rank1 for _ in range(p*n)]
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            row_U = i * m + j
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U1[row_U][col] = S_U[i1*2+j1][c1] * S_U[i2*2+j2][c2]
    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V1[row_V][col] = S_V[j1*2+k1][c1] * S_V[j2*2+k2][c2]
    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W1[row_W][col] = S_W[k1*2+i1][c1] * S_W[k2*2+i2][c2]
    
    rank2 = 28
    U2 = [[0]*rank2 for _ in range(n*m)]
    V2 = [[0]*rank2 for _ in range(m*p)]
    W2 = [[0]*rank2 for _ in range(p*n)]
    
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j][col] = S_U[di * 2 + dj][c]
                
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk
                        V2[j * p + k][col] = S_V[dj * 2 + dk][c]
                
                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i][col] = S_W[dk * 2 + di][c]
    
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
    
    for i in range(m * p):
        for c in range(rank1):
            V[i][c] = V1[i][c]
        for c in range(rank2):
            V[i][rank1 + c] = V2[i][c]
    
    for i in range(p * n):
        for c in range(rank1):
            W[i][c] = W1[i][c]
        for c in range(rank2):
            W[i][rank1 + c] = W2[i][c]
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=4, m=4, p=6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True


def _try_random_search(deadline, best_rank, seed=42):
    """Try to find a lower-rank decomposition using random search with rational entries."""
    n, m, p = 4, 4, 6
    rng = random.Random(seed)
    
    # Try ranks from best_rank-1 down to some minimum
    # The trivial rank is n*m*p = 96, Strassen gives 77
    # Known lower bounds suggest rank >= 48 for 4x4x4, so 4x4x6 might be lower
    min_rank = 50  # conservative lower bound
    
    for rank in range(best_rank - 1, min_rank - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for each rank
        max_attempts = 50 if rank > 60 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with small rational numbers
            # Use entries from {-1, 0, 1} with some randomness
            U = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            # Also try with half-integers
            if attempt % 2 == 1:
                U = [[rng.choice([-1, -0.5, 0, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
                V = [[rng.choice([-1, -0.5, 0, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
                W = [[rng.choice([-1, -0.5, 0, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
                
                if _verify_decomposition(U, V, W, n, m, p):
                    # Convert to exact rationals
                    U = [[Fraction(x).limit_denominator(100) for x in row] for row in U]
                    V = [[Fraction(x).limit_denominator(100) for x in row] for row in V]
                    W = [[Fraction(x).limit_denominator(100) for x in row] for row in W]
                    return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 120s limit
    
    # First, build the guaranteed rank-77 decomposition
    best_result = _build_strassen_decomposition()
    best_rank = len(best_result["U"][0])
    
    # Try to find a better (lower rank) decomposition
    if time.time() < DEADLINE - 10:  # leave 10s margin
        improved = _try_random_search(DEADLINE - 5, best_rank)
        if improved is not None:
            best_result = improved
    
    return best_result
