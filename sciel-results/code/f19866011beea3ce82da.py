"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use Kronecker product structure
    # 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Factor 2: 2x2x3 matrix multiplication
    
    # First, let's build the Strassen decomposition for 2x2x2
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3, let's try to find a good decomposition
    # We'll use a systematic approach with random initialization and optimization
    
    # Let's first try a rank-6 decomposition for 2x2x3
    # If that doesn't work, we'll fall back to rank 11
    
    def build_2x2x3_decomposition(rank):
        """Build a decomposition for 2x2x3 matrix multiplication."""
        # A is 2x2, B is 2x3, C is 2x3
        # U: (2*2, rank) = (4, rank)
        # V: (2*3, rank) = (6, rank)  
        # W: (3*2, rank) = (6, rank)
        
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(4, rank)
        V = np.random.randn(6, rank)
        W = np.random.randn(6, rank)
        
        # The tensor for 2x2x3 matrix multiplication
        # T[i,j,k] = 1 if i,j,k are valid indices for A[i,j]*B[j,k]=C[i,k]
        # A: i in 0..1, j in 0..1
        # B: j in 0..1, k in 0..2
        # C: i in 0..1, k in 0..2
        
        # We need: sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = delta(i,j,k valid)
        
        # Let's use gradient descent to optimize
        lr = 0.01
        for step in range(10000):
            if time.time() > deadline - 10:
                break
            
            # Compute the current tensor
            T_current = np.zeros((4, 6, 6))
            for r in range(rank):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            T_current[i*2+j, j*3+k, k*2+i] += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
            
            # Target tensor
            T_target = np.zeros((4, 6, 6))
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        T_target[i*2+j, j*3+k, k*2+i] = 1.0
            
            # Loss
            loss = np.sum((T_current - T_target)**2)
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(rank):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            idx_u = i*2+j
                            idx_v = j*3+k
                            idx_w = k*2+i
                            err = T_current[idx_u, idx_v, idx_w] - T_target[idx_u, idx_v, idx_w]
                            dU[idx_u, r] += 2 * err * V[idx_v, r] * W[idx_w, r]
                            dV[idx_v, r] += 2 * err * U[idx_u, r] * W[idx_w, r]
                            dW[idx_w, r] += 2 * err * U[idx_u, r] * V[idx_v, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Round to nearest integer/half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
        
        # Check if we got a valid decomposition
        T_final = np.zeros((4, 6, 6))
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        T_final[i*2+j, j*3+k, k*2+i] += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
        
        error = np.max(np.abs(T_final - T_target))
        if error < 1e-6:
            return U, V, W, True
        return U, V, W, False
    
    # Try to find a low-rank decomposition for 2x2x3
    best_rank = None
    best_U2, best_V2, best_W2 = None, None, None
    
    for rank in [5, 6, 7, 8, 9, 10, 11]:
        if time.time() > deadline - 20:
            break
        
        U2, V2, W2, valid = build_2x2x3_decomposition(rank)
        if valid:
            best_rank = rank
            best_U2, best_V2, best_W2 = U2, V2, W2
            break
    
    # If we didn't find a valid decomposition, fall back to rank 11
    if best_rank is None:
        best_rank = 11
        U2 = np.zeros((4, 11))
        V2 = np.zeros((6, 11))
        W2 = np.zeros((6, 11))
        
        # Strassen part (columns 0-1), terms 0-6
        for c in range(7):
            U2[0][c] = U1[0][c]
            U2[1][c] = U1[1][c]
            U2[2][c] = U1[2][c]
            U2[3][c] = U1[3][c]
            
            V2[0][c] = V1[0][c]
            V2[1][c] = V1[1][c]
            V2[3][c] = V1[2][c]
            V2[4][c] = V1[3][c]
            V2[2][c] = 0
            V2[5][c] = 0
            
            W2[0][c] = W1[0][c]
            W2[1][c] = W1[1][c]
            W2[2][c] = W1[2][c]
            W2[3][c] = W1[3][c]
            W2[4][c] = 0
            W2[5][c] = 0
        
        # Trivial part (column 2), terms 7-10
        U2[0][7] = 1
        V2[2][7] = 1
        W2[4][7] = 1
        
        U2[1][8] = 1
        V2[5][8] = 1
        W2[4][8] = 1
        
        U2[2][9] = 1
        V2[2][9] = 1
        W2[5][9] = 1
        
        U2[3][10] = 1
        V2[5][10] = 1
        W2[5][10] = 1
    
    # Now build the full 4x4x6 decomposition using Kronecker product
    rank = 7 * best_rank
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    U1_arr = np.array(U1)
    V1_arr = np.array(V1)
    W1_arr = np.array(W1)
    U2_arr = np.array(U2)
    V2_arr = np.array(V2)
    W2_arr = np.array(W2)
    
    for c1 in range(7):
        for c2 in range(best_rank):
            c = c1 * best_rank + c2
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1_arr[i1*2+j1, c1]
                            u2 = U2_arr[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row, c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1_arr[j1*2+k1, c1]
                            v2 = V2_arr[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row, c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1_arr[k1*2+i1, c1]
                            w2 = W2_arr[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row, c] = w1 * w2
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
