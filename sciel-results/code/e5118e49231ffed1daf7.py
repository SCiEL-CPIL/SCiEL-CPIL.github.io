import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - START_TIME) > TIME_LIMIT

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n=3, m=5, p=7, trials=4):
    """
    Verify the tensor decomposition:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Efficient verification using einsum
    # U_reshaped: (n, m, rank)
    # V_reshaped: (m, p, rank)
    # W_reshaped: (p, n, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: W index is (k, i) corresponding to W[k*n+i, r]
    try:
        # einsum notation:
        # U_r: i, j, r
        # V_r: j, k, r
        # W_r: k, i, r
        # Result: i, j, k
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    except Exception:
        return False
        
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = 3, 5, 7
    
    # Predefined block decompositions
    # Returns (rank, U_local, V_local, W_local)
    # U_local shape: (a*b, rank)
    # V_local shape: (b*c, rank)
    # W_local shape: (c*a, rank)
    
    def schoolbook_block(a, b, c):
        rank = a * b * c
        U = [[0]*rank for _ in range(a*b)]
        V = [[0]*rank for _ in range(b*c)]
        W = [[0]*rank for _ in range(c*a)]
        idx = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i*b+j][idx] = 1
                    V[j*c+k][idx] = 1
                    W[k*a+i][idx] = 1
                    idx += 1
        return rank, U, V, W

    def strassen_block():
        # 2x2x2 Strassen, rank 7.
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0], A11: U[3]
        # B00: V[0], B11: V[3]
        # C00: W[0], C11: W[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        # A10: U[2], A11: U[3]
        # B00: V[0]
        # C10: W[1], C11: W[3]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3 = A00(B01-B11)
        # A00: U[0]
        # B01: V[1], B11: V[3]
        # C01: W[2], C11: W[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        # A11: U[3]
        # B10: V[2], B00: V[0]
        # C00: W[0], C10: W[1]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (A00+A01)B11
        # A00: U[0], A01: U[1]
        # B11: V[3]
        # C00: W[0], C01: W[2]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10: U[2], A00: U[0]
        # B00: V[0], B01: V[1]
        # C11: W[3]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01: U[1], A11: U[3]
        # B10: V[2], B11: V[3]
        # C00: W[0]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return 7, U, V, W

    def block_decomp(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return strassen_block()
        return schoolbook_block(a, b, c)

    # Enumerate all partitions of n, m, p into blocks of size 1 or 2
    def partitions(total, max_part=2):
        res = []
        def rec(rem, cur):
            if rem == 0:
                res.append(tuple(cur))
                return
            for s in (1, 2):
                if s <= rem:
                    cur.append(s)
                    rec(rem - s, cur)
                    cur.pop()
        rec(total, [])
        return res

    pn = partitions(n)
    pm = partitions(m)
    pp = partitions(p)

    best_rank = None
    best_partition = None
    best_blocks = None # Store the precomputed blocks for the best partition to avoid recomputing

    # We will iterate through partitions and calculate the rank.
    # To optimize, we can precompute block ranks.
    block_rank_cache = {}
    def get_block_rank(a, b, c):
        key = (a, b, c)
        if key not in block_rank_cache:
            r, _, _, _ = block_decomp(a, b, c)
            block_rank_cache[key] = r
        return block_rank_cache[key]

    for part_n in pn:
        for part_m in pm:
            for part_p in pp:
                total_rank = 0
                for a in part_n:
                    for b in part_m:
                        for c in part_p:
                            total_rank += get_block_rank(a, b, c)
                
                if best_rank is None or total_rank < best_rank:
                    best_rank = total_rank
                    best_partition = (part_n, part_m, part_p)
                
                if _time_exceeded():
                    break
            if _time_exceeded():
                break
        if _time_exceeded():
            break

    # If no partition found (shouldn't happen), fall back to schoolbook
    if best_partition is None:
        best_partition = ((n,), (m,), (p,))
        best_rank = n * m * p

    part_n, part_m, part_p = best_partition
    rank_total = best_rank

    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]

    def place(rank_offset, i0, j0, k0, a, b, c, U_loc, V_loc, W_loc):
        # Map local indices to global indices
        # Local U index: i_l * b + j_l -> Global: (i0+i_l)*m + (j0+j_l)
        # Local V index: j_l * c + k_l -> Global: (j0+j_l)*p + (k0+k_l)
        # Local W index: k_l * a + i_l -> Global: (k0+k_l)*n + (i0+i_l)
        
        for i_l in range(a):
            for j_l in range(b):
                u_g = (i0+i_l)*m + (j0+j_l)
                u_l = i_l*b + j_l
                for r in range(len(U_loc[0])):
                    U[u_g][rank_offset+r] += U_loc[u_l][r]
        
        for j_l in range(b):
            for k_l in range(c):
                v_g = (j0+j_l)*p + (k0+k_l)
                v_l = j_l*c + k_l
                for r in range(len(V_loc[0])):
                    V[v_g][rank_offset+r] += V_loc[v_l][r]
        
        for k_l in range(c):
            for i_l in range(a):
                w_g = (k0+k_l)*n + (i0+i_l)
                w_l = k_l*a + i_l
                for r in range(len(W_loc[0])):
                    W[w_g][rank_offset+r] += W_loc[w_l][r]

    rank_offset = 0
    i0 = 0
    for a in part_n:
        j0 = 0
        for b in part_m:
            k0 = 0
            for c in part_p:
                r, U_loc, V_loc, W_loc = block_decomp(a, b, c)
                place(rank_offset, i0, j0, k0, a, b, c, U_loc, V_loc, W_loc)
                rank_offset += r
                k0 += c
            j0 += b
        i0 += a

    # Safety check
    if not _verify(U, V, W):
        # Fall back to pure schoolbook
        rank_total = n * m * p
        U = [[0]*rank_total for _ in range(n*m)]
        V = [[0]*rank_total for _ in range(m*p)]
        W = [[0]*rank_total for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", _verify(res['U'], res['V'], res['W']))
