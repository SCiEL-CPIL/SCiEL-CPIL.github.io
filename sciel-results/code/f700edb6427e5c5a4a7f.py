"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_factors():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = sum(W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c] for c in range(rank))
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _generate_partitions(size):
    if size == 0: return [[]]
    if size == 1: return [[(0, 1)]]
    if size == 2: return [[(0, 2)], [(0, 1), (1, 1)]]
    results = []
    for bs in [1, 2]:
        if size >= bs:
            for rest in _generate_partitions(size - bs):
                results.append([(0, bs)] + [(s + bs, sz) for s, sz in rest])
    return results


def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _strassen_2x2x2_factors()
    I_parts = _generate_partitions(n)
    J_parts = _generate_partitions(m)
    K_parts = _generate_partitions(p)
    best_rank = n * m * p
    best_U = best_V = best_W = None
    for I_b in I_parts:
        if time.time() > deadline: break
        for J_b in J_parts:
            if time.time() > deadline: break
            for K_b in K_parts:
                if time.time() > deadline: break
                total_rank = 0
                U = np.zeros((n*m, 200))
                V = np.zeros((m*p, 200))
                W = np.zeros((p*n, 200))
                col = 0
                for (i_s, i_sz) in I_b:
                    for (j_s, j_sz) in J_b:
                        for (k_s, k_sz) in K_b:
                            if i_sz==2 and j_sz==2 and k_sz==2:
                                for li in range(2):
                                    for la in range(2):
                                        U[(i_s+li)*m+(j_s+la), col:col+7]=U_str[li*2+la]
                                for lb in range(2):
                                    for lk in range(2):
                                        V[(j_s+lb)*p+(k_s+lk), col:col+7]=V_str[lb*2+lk]
                                for lk in range(2):
                                    for li in range(2):
                                        W[(k_s+lk)*n+(i_s+li), col:col+7]=W_str[lk*2+li]
                                col += 7
                                total_rank += 7
                            else:
                                for li in range(i_sz):
                                    for la in range(j_sz):
                                        for lk in range(k_sz):
                                            U[(i_s+li)*m+(j_s+la), col]=1
                                            V[(j_s+la)*p+(k_s+lk), col]=1
                                            W[(k_s+lk)*n+(i_s+li), col]=1
                                            col += 1
                                total_rank += i_sz*j_sz*k_sz
                if total_rank < best_rank and col == total_rank:
                    U=U[:,:total_rank]; V=V[:,:total_rank]; W=W[:,:total_rank]
                    if _verify_decomposition(U,V,W,n,m,p):
                        best_rank=total_rank
                        best_U,best_V,best_W=U,V,W
    if best_U is None:
        rank=n*m*p
        best_U=np.zeros((n*m,rank));best_V=np.zeros((m*p,rank));best_W=np.zeros((p*n,rank))
        col=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j,col]=1;best_V[j*p+k,col]=1;best_W[k*n+i,col]=1;col+=1
    def to_list(a):
        r=[]
        for row in a:
            rl=[]
            for v in row:
                rn=round(v)
                if abs(v-rn)<1e-9: rl.append(int(rn))
                else: rl.append(round(v*2)/2)
            r.append(rl)
        return r
    return {"U":to_list(best_U),"V":to_list(best_V),"W":to_list(best_W)}
# EVOLVE-BLOCK-END
