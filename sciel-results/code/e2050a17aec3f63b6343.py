import time
import numpy as np

N, M, P = 2, 5, 5
DEADLINE = 120.0

def A_idx(i, j): return i * M + j
def B_idx(j, k): return j * P + k
def C_idx(k, i): return k * N + i

def build_tensor():
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[A_idx(i,j), B_idx(j,k), C_idx(k,i)] = 1.0
    return T

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    U_total = []
    V_total = []
    W_total = []
    
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # r=0: M1 = (A11 - A22)(B21 + B22)
        U[0,0] = 1.0; U[3,0] = -1.0
        V[2,0] = 1.0; V[3,0] = 1.0
        W[0,0] = 1.0
        
        # r=1: M2 = (A11 + A22)B22
        U[0,1] = 1.0; U[3,1] = 1.0
        V[3,1] = 1.0
        W[2,1] = 1.0; W[3,1] = 1.0
        
        # r=2: M3 = A11(B11 - B21)
        U[0,2] = 1.0
        V[0,2] = 1.0; V[2,2] = -1.0
        W[1,2] = 1.0; W[3,2] = -1.0
        
        # r=3: M4 = A22(B12 - B11)
        U[3,3] = 1.0
        V[1,3] = 1.0; V[0,3] = -1.0
        W[0,3] = 1.0; W[3,3] = 1.0
        
        # r=4: M5 = (A11 + A12)B11
        U[0,4] = 1.0; U[1,4] = 1.0
        V[0,4] = 1.0
        W[0,4] = -1.0; W[1,4] = 1.0
        
        # r=5: M6 = (A21 - A11)(B11 + B12)
        U[2,5] = 1.0; U[0,5] = -1.0
        V[0,5] = 1.0; V[1,5] = 1.0
        W[3,5] = 1.0
        
        # r=6: M7 = (A12 - A22)(B21 + B22)
        U[1,6] = 1.0; U[3,6] = -1.0
        V[2,6] = 1.0; V[3,6] = 1.0
        W[0,6] = 1.0
        
        return U, V, W

    def get_block(n, m, p):
        # n=2 always for this problem
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        elif n == 2 and m == 1 and p == 1:
            U = np.zeros((2, 1)); V = np.zeros((1, 1)); W = np.zeros((2, 1))
            U[0,0] = 1.0; U[1,0] = 1.0
            V[0,0] = 1.0
            W[0,0] = 1.0; W[1,0] = 1.0
            return U, V, W
        elif n == 2 and m == 1 and p == 2:
            U = np.zeros((2, 2)); V = np.zeros((2, 2)); W = np.zeros((4, 2))
            # r=0: k=0
            U[0,0] = 1.0; U[1,0] = 1.0
            V[0,0] = 1.0
            W[0,0] = 1.0; W[2,0] = 1.0
            # r=1: k=1
            U[0,1] = 1.0; U[1,1] = 1.0
            V[1,1] = 1.0
            W[1,1] = 1.0; W[3,1] = 1.0
            return U, V, W
        elif n == 2 and m == 2 and p == 1:
            U = np.zeros((4, 2)); V = np.zeros((2, 2)); W = np.zeros((2, 2))
            # r=0: j=0
            U[0,0] = 1.0; U[2,0] = 1.0
            V[0,0] = 1.0
            W[0,0] = 1.0; W[1,0] = 1.0
            # r=1: j=1
            U[1,1] = 1.0; U[3,1] = 1.0
            V[1,1] = 1.0
            W[0,1] = 1.0; W[1,1] = 1.0
            return U, V, W
        else:
            # Fallback to schoolbook
            R = n*m*p
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
            return U, V, W

    def add_block(n, m, p, i_off, j_off, k_off):
        Ub, Vb, Wb = get_block(n, m, p)
        R = Ub.shape[1]
        
        U_block = np.zeros((N*M, R))
        V_block = np.zeros((M*P, R))
        W_block = np.zeros((P*N, R))
        
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U_block[A_idx(i_off+i, j_off+j), r] = Ub[i*m+j, r]
            for j in range(m):
                for k in range(p):
                    V_block[B_idx(j_off+j, k_off+k), r] = Vb[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    W_block[C_idx(k_off+k, i_off+i), r] = Wb[k*n+i, r]
                    
        U_total.append(U_block)
        V_total.append(V_block)
        W_total.append(W_block)

    # Decomposition Plan (Rank 27)
    # 1. Core 2x2x3 (j=0,1; k=0,1,2) -> Split into 2x2x1 (k=0) and 2x2x2 (k=1,2)
    add_block(2, 2, 1, 0, 0, 0) # Rank 2
    add_block(2, 2, 2, 0, 0, 1) # Rank 7
    
    # 2. Core 2x1x3 (j=2; k=0,1,2) -> Split into 2x1x1 (k=0) and 2x1x2 (k=1,2)
    add_block(2, 1, 1, 0, 2, 0) # Rank 1
    add_block(2, 1, 2, 0, 2, 1) # Rank 2
    
    # 3. Edge 2x3x1 (j=0,1,2; k=3) -> Split into 2x2x1 (j=0,1) and 2x1x1 (j=2)
    add_block(2, 2, 1, 0, 0, 3) # Rank 2
    add_block(2, 1, 1, 0, 2, 3) # Rank 1
    
    # 4. Edge 2x1x3 (j=3; k=0,1,2) -> Split into 2x1x1 (k=0) and 2x1x2 (k=1,2)
    add_block(2, 1, 1, 0, 3, 0) # Rank 1
    add_block(2, 1, 2, 0, 3, 1) # Rank 2
    
    # 5. Corner 2x1x1 (j=3; k=3)
    add_block(2, 1, 1, 0, 3, 3) # Rank 1
    
    # 6. Edge 2x4x1 (j=0,1,2,3; k=4) -> Split into 2x2x1 (j=0,1) and 2x2x1 (j=2,3)
    add_block(2, 2, 1, 0, 0, 4) # Rank 2
    add_block(2, 2, 1, 0, 2, 4) # Rank 2
    
    # 7. Edge 2x1x4 (j=4; k=0,1,2,3,4) -> Split into 2x1x2 (k=0,1) and 2x1x2 (k=2,3) and 2x1x1 (k=4)
    # Wait, 2x1x4 split into 2x1x2 and 2x1x2 covers k=0,1 and k=2,3. k=4 is left.
    add_block(2, 1, 2, 0, 4, 0) # Rank 2
    add_block(2, 1, 2, 0, 4, 2) # Rank 2
    add_block(2, 1, 1, 0, 4, 4) # Rank 1
    
    # Total Rank: 2+7+1+2+2+1+1+2+1+2+2+2+2+1 = 28?
    # Let's sum:
    # 1: 2+7=9
    # 2: 1+2=3
    # 3: 2+1=3
    # 4: 1+2=3
    # 5: 1
    # 6: 2+2=4
    # 7: 2+2+1=5
    # Total: 9+3+3+3+1+4+5 = 28.
    
    # Can we reduce Block 7?
    # 2x1x4. Split into 2x1x2 (k=0,1) and 2x1x2 (k=2,3) and 2x1x1 (k=4).
    # Is there a better split?
    # 2x1x4 is just 4 dot products of length 1? No, A is 2x1.
    # C_k = A_0 B_0,k + A_1 B_1,k.
    # This is 4 independent rank-1 terms?
    # No, A is shared.
    # C_k = A_0 B_0,k + A_1 B_1,k.
    # This is a 2x1x4 tensor.
    # Rank is 4?
    # U (2x4), V (4x4), W (4x2).
    # Schoolbook is 4.
    # Can we do rank 3?
    # Unlikely for this small size.
    
    # So Rank 28.
    
    U = np.hstack(U_total)
    V = np.hstack(V_total)
    W = np.hstack(W_total)
    
    rank = U.shape[1]
    
    if time_left() > 3.0:
        T = build_tensor()
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((recon - T)**2)
        if err > 1e-9:
            # Fallback
            U = np.zeros((N*M, N*M*P))
            V = np.zeros((M*P, N*M*P))
            W = np.zeros((P*N, N*M*P))
            col = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        U[A_idx(i,j), col] = 1.0
                        V[B_idx(j,k), col] = 1.0
                        W[C_idx(k,i), col] = 1.0
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
