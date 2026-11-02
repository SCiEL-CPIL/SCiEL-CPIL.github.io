"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # WFTA (Winograd Fourier Transform Algorithm) decomposition using 2-point DFT
    # delta_{j,j'} = (1/2) sum_{s=0}^{1} (-1)^{s(j%2)} (-1)^{s(j'%2)} * delta_{j//2, j'//2}
    # Rank = 2 (for a) * 2 (for s) = 4
    def build_wfta():
        rank = 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                a = j // 2
                b = j % 2
                a_idx = i * m + j
                for s in range(2):
                    t = a * 2 + s
                    U[a_idx, t] = float((-1) ** (s * b))
            for j in range(m):
                a = j // 2
                b = j % 2
                for k in range(p):
                    b_idx = j * p + k
                    for s in range(2):
                        t = a * 2 + s
                        V[b_idx, t] = float((-1) ** (s * b))
            for i in range(n):
                for k in range(p):
                    c_idx = k * n + i
                    for t in range(rank):
                        W[c_idx, t] = 0.5
        return U, V, W

    U4, V4, W4 = build_wfta()
    if verify(U4, V4, W4, tol=1e-8):
        return to_lists(U4, V4, W4, 4)

    # Fallback: schoolbook rank 32
    schoolbook_rank = n * m * p
    U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
    return to_lists(U, V, W, schoolbook_rank)
# EVOLVE-BLOCK-END
