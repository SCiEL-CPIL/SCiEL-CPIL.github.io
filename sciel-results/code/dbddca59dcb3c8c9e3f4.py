import numpy as np

def construct():
    n, m, p = 3, 5, 6
    rank = 31
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(u_base, v_base, w_base):
        """
        u_base: dict mapping local (i,j) to global U index
        v_base: dict mapping local (j,k) to global V index  
        w_base: dict mapping local (k,i) to global W index
        """
        nonlocal col
        # m0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[u_base[(0,0)], col] = 1; U[u_base[(1,1)], col] = 1
        V[v_base[(0,0)], col] = 1; V[v_base[(1,1)], col] = 1
        W[w_base[(0,0)], col] = 1; W[w_base[(1,1)], col] = 1
        col += 1
        # m1 = (A[1,0]+A[1,1])*B[0,0]
        U[u_base[(1,0)], col] = 1; U[u_base[(1,1)], col] = 1
        V[v_base[(0,0)], col] = 1
        W[w_base[(0,1)], col] = 1; W[w_base[(1,1)], col] = -1
        col += 1
        # m2 = A[0,0]*(B[0,1]-B[1,1])
        U[u_base[(0,0)], col] = 1
        V[v_base[(0,1)], col] = 1; V[v_base[(1,1)], col] = -1
        W[w_base[(1,0)], col] = 1; W[w_base[(1,1)], col] = 1
        col += 1
        # m3 = A[1,1]*(B[1,0]-B[0,0])
        U[u_base[(1,1)], col] = 1
        V[v_base[(1,0)], col] = 1; V[v_base[(0,0)], col] = -1
        W[w_base[(0,0)], col] = 1; W[w_base[(0,1)], col] = 1
        col += 1
        # m4 = (A[0,0]+A[0,1])*B[1,1]
        U[u_base[(0,0)], col] = 1; U[u_base[(0,1)], col] = 1
        V[v_base[(1,1)], col] = 1
        W[w_base[(0,0)], col] = -1; W[w_base[(1,0)], col] = 1
        col += 1
        # m5 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[u_base[(1,0)], col] = 1; U[u_base[(0,0)], col] = -1
        V[v_base[(0,0)], col] = 1; V[v_base[(0,1)], col] = 1
        W[w_base[(1,1)], col] = 1
        col += 1
        # m6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[u_base[(0,1)], col] = 1; U[u_base[(1,1)], col] = -1
        V[v_base[(1,0)], col] = 1; V[v_base[(1,1)], col] = 1
        W[w_base[(0,0)], col] = 1
        col += 1
    
    # Strassen block 1: i in {0,1}, j in {0,1}, k in {0,1}
    u1 = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    v1 = {(0,0): 0, (0,1): 1, (1,0): 6, (1,1): 7}
    w1 = {(0,0): 0, (1,0): 3, (0,1): 1, (1,1): 4}
    add_strassen(u1, v1, w1)
    
    # Strassen block 2: i in {0,1}, j in {0,1}, k in {2,3}
    u2 = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    v2 = {(0,0): 2, (0,1): 3, (1,0): 8, (1,1): 9}
    w2 = {(0,0): 6, (1,0): 9, (0,1): 7, (1,1): 10}
    add_strassen(u2, v2, w2)
    
    # Strassen block 3: i in {0,1}, j in {0,1}, k in {4,5}
    u3 = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    v3 = {(0,0): 4, (0,1): 5, (1,0): 10, (1,1): 11}
    w3 = {(0,0): 12, (1,0): 15, (0,1): 13, (1,1): 16}
    add_strassen(u3, v3, w3)
    
    # 2x1x2 blocks: i in {0,1}, j in {2,3,4}, k blocks {0,1},{2,3},{4,5}
    for j in [2, 3, 4]:
        for k_start in [0, 2, 4]:
            U[0 * 5 + j, col] = 1
            U[1 * 5 + j, col] = 1
            V[j * 6 + k_start, col] = 1
            V[j * 6 + k_start + 1, col] = 1
            for i in [0, 1]:
                for k in [k_start, k_start + 1]:
                    W[k * 3 + i, col] = 1
            col += 1
    
    # 1x5x6 block: i=2, j in {0..4}, k in {0..5}
    for j in range(5):
        U[2 * 5 + j, col] = 1
    for j in range(5):
        for k in range(6):
            V[j * 6 + k, col] = 1
    for k in range(6):
        W[k * 3 + 2, col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1) > 1e-10:
                    raise ValueError(f"Failed at ({i},{j},{k}): {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
