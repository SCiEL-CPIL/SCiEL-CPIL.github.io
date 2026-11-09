import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
N, M, P = DIMENSIONS
TRIVIAL_RANK = N * M * P  # 48

# Build target tensor T[i,j,k] = 1 for all (i,j,k) in [N]x[M]x[P]
# Decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all (i,j,k)

def verify(U, V, W, rank):
    # U: N*M x rank, V: M*P x rank, W: P*N x rank
    # Check sum_c U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c] == 1
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(rank):
                    s += U[i*M+j][c] * V[j*P+k][c] * W[k*N+i][c]
                if s != 1:
                    return False
    return True

def construct():
    deadline = time.time() + 120
    best_rank = TRIVIAL_RANK
    best_result = None

    # Trivial decomposition as fallback
    rank = TRIVIAL_RANK
    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][col] = 1
                V[j*P+k][col] = 1
                W[k*N+i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}

    # Candidate values: integers and half-integers in a small range
    # Try values from {-2,-1,-1/2,0,1/2,1,2}
    vals = [-2, -1, Fraction(-1,2), 0, Fraction(1,2), 1, 2]

    random.seed(42)

    # Try various ranks below the trivial rank
    # Known: rank(2,3,p) — for p=8, we can try to find rank < 48
    # Lower bound: rank >= max(N*M, M*P, P*N) = max(6, 24, 16) = 24
    # Also rank >= (N*M*P) / max possible... 
    # Let's search from lower ranks upward

    for rank in range(24, TRIVIAL_RANK):
        if time.time() > deadline - 5:
            break

        # Random search: try to find a decomposition of this rank
        # Use a structured approach: each column c has a "pattern"
        # U[i,j,c], V[j,k,c], W[k,i,c]
        
        # For a rank-r decomposition, we need r "gates" that together
        # sum to 1 for each (i,j,k)
        
        # Simple approach: random integer/half-integer entries
        max_iter = 2000
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Random entries from small set
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(N*M)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(M*P)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(P*N)]
            
            # Quick check: for each (i,j,k), the sum must be exactly 1
            ok = True
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        s = 0
                        for c in range(rank):
                            s += U[i*M+j][c] * V[j*P+k][c] * W[k*N+i][c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert Fraction to int/float for output
                def convert(mat):
                    out = []
                    for row in mat:
                        newrow = []
                        for v in row:
                            if isinstance(v, Fraction):
                                if v.denominator == 1:
                                    newrow.append(int(v.numerator))
                                else:
                                    newrow.append(v.numerator / v.denominator)
                            else:
                                newrow.append(v)
                        out.append(newrow)
                    return out
                best_rank = rank
                best_result = {"U": convert(U), "V": convert(V), "W": convert(W)}
                break
        if best_rank < TRIVIAL_RANK:
            break

    return best_result
