import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 140

    def make_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def numeric_check(U, V, W, A, B, tol=1e-6):
        """Verify U, V, W decompose the multiplication: for all i,j,k, sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = C[i,k] where C = A@B."""
        C = A @ B
        # Build full tensor check via the rank decomposition
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for t in range(U.shape[1]):
                        total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(total - C[i,k]) > tol:
                    return False
        return True

    def make_strassen_block():
        """2x2*2x2 Strassen, rank 7. U indexed [i*2+j], V [j*2+k], W [k*2+i] (n=2)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # a=A[0,0]->U0, b=A[0,1]->U1, c=A[1,0]->U2, d=A[1,1]->U3
        # e=B[0,0]->V0, f=B[0,1]->V1, g=B[1,0]->V2, h=B[1,1]->V3
        # C: C00=W0, C01=W1, C10=W2, C11=W3  (W[k*2+i]: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
        # M1 = (a+d)(e+h)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        # M2 = (c+d)e
        U[2,1]=1; U[3,1]=1; V[0,1]=1
        # M3 = a(f-h)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1
        # M4 = d(g-e)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        # M5 = (a+c)h
        U[0,4]=1; U[2,4]=1; V[3,4]=1
        # M6 = (b-d)(e+g)
        U[1,5]=1; U[3,5]=-1; V[0,5]=1; V[2,5]=1
        # M7 = (c-a)(f+h)
        U[2,6]=1; U[0,6]=-1; V[1,6]=1; V[3,6]=1
        # C00 = M1 - M5 + M4 + M7
        W[0,0]=1; W[0,4]=-1; W[0,3]=1; W[0,6]=1
        # C01 = M3 + M5
        W[1,2]=1; W[1,4]=1
        # C10 = M2 + M4
        W[2,1]=1; W[2,3]=1
        # C11 = M1 - M3 + M6 - M2
        W[3,0]=1; W[3,2]=-1; W[3,5]=1; W[3,1]=-1
        return U, V, W

    # Build full 2x4 * 4x8 via 8 Strassen blocks
    U_block, V_block, W_block = make_strassen_block()
    rank_total = 8 * 7  # 56
    U_full = np.zeros((U_dim, rank_total))
    V_full = np.zeros((V_dim, rank_total))
    W_full = np.zeros((W_dim, rank_total))

    col_offset = 0
    for a_block in range(2):  # which 2x2 block of A (columns)
        a_col_base = a_block * 2
        b_row_base = a_block * 2  # corresponding rows of B
        for c_block in range(4):  # which 2 columns of output
            c_col_base = c_block * 2
            for t in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[i*4 + a_col_base + j, col_offset] += U_block[i*2+j, t]
                for j in range(2):
                    for k in range(2):
                        V_full[(b_row_base+j)*8 + c_col_base + k, col_offset] += V_block[j*2+k, t]
                for i in range(2):
                    for k in range(2):
                        W_full[(c_col_base+k)*2 + i, col_offset] += W_block[k*2+i, t]
                col_offset += 1

    # Verify with random matrix
    rng = np.random.RandomState(0)
    A = rng.randint(-5, 6, size=(n, m)).astype(float)
    B = rng.randint(-5, 6, size=(m, p)).astype(float)
    if numeric_check(U_full, V_full, W_full, A, B, tol=1e-6):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback schoolbook
    U0, V0, W0, r0 = make_schoolbook()
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
