import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: A[i*2+j], B[j*2+k], C[k*2+i]
    """
    # 7 Strassen terms
    # u[i*2+j], v[j*2+k], w[k*2+i]
    terms = [
        # (u, v, w)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),   # M3
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),   # M4
        ([1,0,1,0], [0,0,0,1], [-1,0,1,0]),   # M5
        ([-1,0,0,1], [1,0,1,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),   # M7
    ]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for t, (u, v, w) in enumerate(terms):
        for idx in range(4):
            U[idx][t] = u[idx]
            V[idx][t] = v[idx]
            W[idx][t] = w[idx]
    return U, V, W

def construct():
    n, m, p = 2, 2, 15
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Partition B's 15 columns: 7 groups of 2, 1 group of 1
    # Groups: columns 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13 (Strassen), 14 (schoolbook)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 53)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, 53)
    W = [[0] * total_rank for _ in range(p * n)]  # (30, 53)
    
    col = 0
    # Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for block in range(7):
        k_start = block * 2
        for t in range(7):
            # Map local indices to global
            for i in range(n):
                for j in range(m):
                    U[i*m + j][col] = U_s[i*m + j][t]
            for j in range(m):
                for k in range(2):
                    V[j*p + (k_start + k)][col] = V_s[j*2 + k][t]
            for k in range(2):
                for i in range(n):
                    W[(k_start + k)*n + i][col] = W_s[k*2 + i][t]
            col += 1
    
    # Schoolbook for column 14
    k = 14
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    # Check that sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    # where T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i with j=j' and k=k'... 
    # Actually T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    
    # Build expected tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Compute decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute sum_t U[:,t] * V[:,t] * W[:,t] for each (a,b,c)
    # This is: for each t, outer product of U[:,t], V[:,t], W[:,t]
    # Sum over t
    computed = np.zeros((n*m, m*p, p*n))
    for t in range(total_rank):
        computed += np.einsum('a,b,c->abc', U_arr[:,t], V_arr[:,t], W_arr[:,t])
    
    if not np.allclose(computed, T, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_fb = [[0] * rank_school for _ in range(n * m)]
        V_fb = [[0] * rank_school for _ in range(m * p)]
        W_fb = [[0] * rank_school for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
