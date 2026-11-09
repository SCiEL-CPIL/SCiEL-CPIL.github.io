import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # DP table for minimum rank
    # rank_dp[n][m][p] = minimum rank for n x m times m x p
    # We'll compute this bottom-up
    max_n, max_m, max_p = n, m, p
    
    # Initialize with schoolbook
    rank_dp = {}
    for i in range(1, max_n + 1):
        for j in range(1, max_m + 1):
            for k in range(1, max_p + 1):
                rank_dp[(i, j, k)] = i * j * k
    
    # Fill in DP table
    for i in range(1, max_n + 1):
        for j in range(1, max_m + 1):
            for k in range(1, max_p + 1):
                if time.time() > deadline:
                    break
                # Try splitting i
                for ii in range(1, i):
                    r = rank_dp[(ii, j, k)] + rank_dp[(i - ii, j, k)]
                    if r < rank_dp[(i, j, k)]:
                        rank_dp[(i, j, k)] = r
                # Try splitting j
                for jj in range(1, j):
                    r = rank_dp[(i, jj, k)] + rank_dp[(i, j - jj, k)]
                    if r < rank_dp[(i, j, k)]:
                        rank_dp[(i, j, k)] = r
                # Try splitting k
                for kk in range(1, k):
                    r = rank_dp[(i, j, kk)] + rank_dp[(i, j, k - kk)]
                    if r < rank_dp[(i, j, k)]:
                        rank_dp[(i, j, k)] = r
    
    best_rank = rank_dp[(n, m, p)]
    
    # Now construct the decomposition using the same splitting strategy
    def get_split(i, j, k):
        """Find the best split for (i, j, k)"""
        if time.time() > deadline:
            return None
        best = rank_dp[(i, j, k)]
        # Try splitting i
        for ii in range(1, i):
            r = rank_dp[(ii, j, k)] + rank_dp[(i - ii, j, k)]
            if r == best:
                return ('n', ii)
        # Try splitting j
        for jj in range(1, j):
            r = rank_dp[(i, jj, k)] + rank_dp[(i, j - jj, k)]
            if r == best:
                return ('m', jj)
        # Try splitting k
        for kk in range(1, k):
            r = rank_dp[(i, j, kk)] + rank_dp[(i, j, k - kk)]
            if r == best:
                return ('p', kk)
        return None  # No beneficial split, use schoolbook
    
    def build(i, j, k):
        """Build U, V, W for i x j times j x k"""
        if time.time() > deadline:
            # Fallback to schoolbook
            rank = i * j * k
            U = [[0] * rank for _ in range(i * j)]
            V = [[0] * rank for _ in range(j * k)]
            W = [[0] * rank for _ in range(k * i)]
            col = 0
            for a in range(i):
                for b in range(j):
                    for c in range(k):
                        U[a * j + b][col] = 1
                        V[b * k + c][col] = 1
                        W[c * i + a][col] = 1
                        col += 1
            return U, V, W, rank
        
        split = get_split(i, j, k)
        
        if split is None:
            # Schoolbook
            rank = i * j * k
            U = [[0] * rank for _ in range(i * j)]
            V = [[0] * rank for _ in range(j * k)]
            W = [[0] * rank for _ in range(k * i)]
            col = 0
            for a in range(i):
                for b in range(j):
                    for c in range(k):
                        U[a * j + b][col] = 1
                        V[b * k + c][col] = 1
                        W[c * i + a][col] = 1
                        col += 1
            return U, V, W, rank
        
        dim, idx = split
        if dim == 'n':
            i1, i2 = idx, i - idx
            U1, V1, W1, r1 = build(i1, j, k)
            U2, V2, W2, r2 = build(i2, j, k)
            
            U = [[0] * (r1 + r2) for _ in range(i * j)]
            V = [[0] * (r1 + r2) for _ in range(j * k)]
            W = [[0] * (r1 + r2) for _ in range(k * i)]
            
            for a in range(i1):
                for b in range(j):
                    for c in range(r1):
                        U[a * j + b][c] = U1[a * j + b][c]
            for a in range(i2):
                for b in range(j):
                    for c in range(r2):
                        U[(i1 + a) * j + b][r1 + c] = U2[a * j + b][c]
                        
            for b in range(j):
                for c in range(k):
                    for cc in range(r1):
                        V[b * k + c][cc] = V1[b * k + c][cc]
                    for cc in range(r2):
                        V[b * k + c][r1 + cc] = V2[b * k + c][cc]
                        
            for c in range(k):
                for a in range(i1):
                    for cc in range(r1):
                        W[c * i + a][cc] = W1[c * i1 + a][cc]
                for a in range(i2):
                    for cc in range(r2):
                        W[c * i + (i1 + a)][r1 + cc] = W2[c * i2 + a][cc]
                        
            return U, V, W, r1 + r2
            
        elif dim == 'm':
            j1, j2 = idx, j - idx
            U1, V1, W1, r1 = build(i, j1, k)
            U2, V2, W2, r2 = build(i, j2, k)
            
            U = [[0] * (r1 + r2) for _ in range(i * j)]
            V = [[0] * (r1 + r2) for _ in range(j * k)]
            W = [[0] * (r1 + r2) for _ in range(k * i)]
            
            for a in range(i):
                for b in range(j1):
                    for c in range(r1):
                        U[a * j + b][c] = U1[a * j1 + b][c]
                for b in range(j2):
                    for c in range(r2):
                        U[a * j + (j1 + b)][r1 + c] = U2[a * j2 + b][c]
                        
            for b in range(j1):
                for c in range(k):
                    for cc in range(r1):
                        V[b * k + c][cc] = V1[b * k + c][cc]
            for b in range(j2):
                for c in range(k):
                    for cc in range(r2):
                        V[(j1 + b) * k + c][r1 + cc] = V2[b * k + c][cc]
                        
            for c in range(k):
                for a in range(i):
                    for cc in range(r1):
                        W[c * i + a][cc] = W1[c * i + a][cc]
                    for cc in range(r2):
                        W[c * i + a][r1 + cc] = W2[c * i + a][cc]
                        
            return U, V, W, r1 + r2
            
        else:  # dim == 'p'
            k1, k2 = idx, k - idx
            U1, V1, W1, r1 = build(i, j, k1)
            U2, V2, W2, r2 = build(i, j, k2)
            
            U = [[0] * (r1 + r2) for _ in range(i * j)]
            V = [[0] * (r1 + r2) for _ in range(j * k)]
            W = [[0] * (r1 + r2) for _ in range(k * i)]
            
            for a in range(i):
                for b in range(j):
                    for c in range(r1):
                        U[a * j + b][c] = U1[a * j + b][c]
                    for c in range(r2):
                        U[a * j + b][r1 + c] = U2[a * j + b][c]
                        
            for b in range(j):
                for c in range(k1):
                    for cc in range(r1):
                        V[b * k + c][cc] = V1[b * k1 + c][cc]
                for c in range(k2):
                    for cc in range(r2):
                        V[b * k + (k1 + c)][r1 + cc] = V2[b * k2 + c][cc]
                        
            for c in range(k1):
                for a in range(i):
                    for cc in range(r1):
                        W[c * i + a][cc] = W1[c * i + a][cc]
            for c in range(k2):
                for a in range(i):
                    for cc in range(r2):
                        W[(k1 + c) * i + a][r1 + cc] = W2[c * i + a][cc]
                        
            return U, V, W, r1 + r2
    
    U, V, W, rank = build(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
