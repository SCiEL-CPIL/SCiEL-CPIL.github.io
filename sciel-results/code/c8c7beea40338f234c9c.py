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

    # We will build the decomposition by combining smaller blocks.
    # Target Rank ~ 23
    
    # Helper to create a block decomposition
    # n, m, p are dimensions of A (n x m), B (m x p), C (p x n)
    # Returns U, V, W for this block
    def get_block(n, m, p):
        # Schoolbook for small sizes if no better known
        # But we have specific constructions for 2xn x n
        
        if n == 2 and m == 2 and p == 2:
            # Strassen 7
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            # ... Strassen code ...
            # Re-using strassen_222 from previous thought
            U, V, W = strassen_222()
            return U, V, W
            
        elif n == 2 and m == 1 and p == 1:
            # Rank 1
            U = np.zeros((2, 1))
            V = np.zeros((1, 1))
            W = np.zeros((2, 1))
            U[0,0] = 1.0
            U[1,0] = 1.0
            V[0,0] = 1.0
            W[0,0] = 1.0
            W[1,0] = 1.0
            return U, V, W
            
        elif n == 2 and m == 1 and p == 2:
            # Rank 1
            U = np.zeros((2, 1))
            V = np.zeros((2, 1))
            W = np.zeros((2, 1))
            U[0,0] = 1.0
            U[1,0] = 1.0
            V[0,0] = 1.0
            V[1,0] = 1.0
            W[0,0] = 1.0
            W[1,0] = 1.0
            return U, V, W
            
        elif n == 2 and m == 2 and p == 1:
            # Rank 2
            U = np.zeros((4, 2))
            V = np.zeros((2, 2))
            W = np.zeros((2, 2))
            # r=0: A_i0 B_00
            U[0,0] = 1.0 # i=0, j=0
            U[2,0] = 1.0 # i=1, j=0
            V[0,0] = 1.0 # j=0, k=0
            W[0,0] = 1.0
            W[1,0] = 1.0
            # r=1: A_i1 B_10
            U[1,1] = 1.0 # i=0, j=1
            U[3,1] = 1.0 # i=1, j=1
            V[1,1] = 1.0 # j=1, k=0
            W[0,1] = 1.0
            W[1,1] = 1.0
            return U, V, W
            
        elif n == 2 and m == 3 and p == 1:
            # Rank 3
            U = np.zeros((6, 3))
            V = np.zeros((3, 3))
            W = np.zeros((2, 3))
            for j in range(3):
                U[j, j] = 1.0 # i=0? No, U is (n*m, R). Index i*m+j.
                # Wait, U index is i*m+j.
                # For n=2, m=3.
                # i=0: 0,1,2. i=1: 3,4,5.
                # We need U[i, j] to pick A[i,j].
                # r=j corresponds to j.
                # U[i*3+j, j] = 1?
                # If r=j, U[i*3+j, j] should be 1 for all i?
                # No, U is constant.
                # C_{0,i} = sum_j A[i,j] B[j,0].
                # Term r=j: U[i*3+j, j] * V[j,0,j] * W[0,i,j].
                # We want A[i,j] B[j,0].
                # So U[i*3+j, j] = 1.
                # V[j,0,j] = 1.
                # W[0,i,j] = 1.
                pass
            # Construct properly
            U = np.zeros((6, 3))
            V = np.zeros((3, 3))
            W = np.zeros((2, 3))
            for j in range(3):
                for i in range(2):
                    U[i*3+j, j] = 1.0
                V[j, j] = 1.0 # V[j*1+0, j] = V[j, j]
                for i in range(2):
                    W[i, j] = 1.0 # W[0*2+i, j] = W[i, j]
            return U, V, W
            
        elif n == 2 and m == 1 and p == 3:
            # Rank 3
            U = np.zeros((2, 3))
            V = np.zeros((3, 3))
            W = np.zeros((6, 3))
            for k in range(3):
                for i in range(2):
                    U[i, k] = 1.0
                V[k, k] = 1.0 # V[0*3+k, k] = V[k, k]
                for i in range(2):
                    W[k*2+i, k] = 1.0
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

    # Combine blocks
    # 2x5x5
    # Split B (5x5) into rows 0-3 and 4.
    # Split A (2x5) into cols 0-3 and 4.
    # Core: A(2x4) B(4x4)
    # Edge1: A(2x4) B(4x1) [col 4 of B]
    # Edge2: A(2x1) B(1x4) [row 4 of B]
    
    # Core 2x4x4
    # Split B(4x4) into rows 0-2 and 3.
    # Split A(2x4) into cols 0-2 and 3.
    # Core1: A(2x3) B(3x3)
    # Edge1: A(2x3) B(3x1)
    # Edge2: A(2x1) B(1x3)
    # Edge3: A(2x1) B(1x1)
    
    # We will accumulate U, V, W
    U_total = []
    V_total = []
    W_total = []
    
    def add_block(n, m, p, i_off, j_off, k_off):
        # i_off: offset in A rows (always 0 for N=2)
        # j_off: offset in A cols / B rows
        # k_off: offset in B cols / C rows
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

    # 1. Core 2x3x3 (i=0, j=0..2, k=0..2)
    add_block(2, 3, 3, 0, 0, 0)
    
    # 2. Edge 2x3x1 (i=0, j=0..2, k=3)
    add_block(2, 3, 1, 0, 0, 3)
    
    # 3. Edge 2x1x3 (i=0, j=3, k=0..2)
    add_block(2, 1, 3, 0, 3, 0)
    
    # 4. Edge 2x1x1 (i=0, j=3, k=3)
    add_block(2, 1, 1, 0, 3, 3)
    
    # 5. Edge 2x4x1 (i=0, j=0..3, k=4)
    add_block(2, 4, 1, 0, 0, 4)
    
    # 6. Edge 2x1x4 (i=0, j=4, k=0..4)
    add_block(2, 1, 4, 0, 4, 0)

    U = np.hstack(U_total)
    V = np.hstack(V_total)
    W = np.hstack(W_total)
    
    rank = U.shape[1]
    
    # Verify
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
