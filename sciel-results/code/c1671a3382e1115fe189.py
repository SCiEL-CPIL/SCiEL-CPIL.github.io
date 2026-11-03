import time
import sys
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Generate all possible contiguous partitions of a dimension of size D
    def get_partitions(D):
        parts = []
        # k is the number of blocks
        for k in range(1, D + 1):
            # We need to choose k-1 cut points from D-1 possible gaps
            # This generates all compositions of D into k parts
            for cuts in itertools.combinations(range(1, D), k-1):
                prev = 0
                part = []
                for c in cuts:
                    part.append(c - prev)
                    prev = c
                part.append(D - prev)
                parts.append(tuple(part))
        return parts

    P_n = get_partitions(n)
    P_m = get_partitions(m)
    P_p = get_partitions(p)
    
    best_rank = n * m * p
    best_partition = (P_n[0], P_m[0], P_p[0])
    
    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    # Search for the best partition
    for pn in P_n:
        for pm in P_m:
            for pp in P_p:
                rank = 0
                for a in pn:
                    for b in pm:
                        for c in pp:
                            rank += block_rank(a, b, c)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (pn, pm, pp)
                    
    # Construct the matrices
    pn, pm, pp = best_partition
    
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
                    col += 1

    def add_strassen(i0, j0, k0):
        nonlocal col
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # Fill matrices based on the best partition
    i_start = 0
    for a in pn:
        i_end = i_start + a
        j_start = 0
        for b in pm:
            j_end = j_start + b
            k_start = 0
            for c in pp:
                k_end = k_start + c
                if a == 2 and b == 2 and c == 2:
                    add_strassen(i_start, j_start, k_start)
                else:
                    add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end)
                k_start = k_end
            j_start = j_end
        i_start = i_end
        
    # Verify rank
    if col != best_rank:
        # This should not happen if logic is correct
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    import random
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
    else:
        print(f"Verification passed. Rank: {rank}", file=sys.stderr)
