import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def validate(U, V, W, atol=1e-5):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
        
    def snap_to_int(arr):
        return np.round(arr)
        
    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        U_i = snap_to_int(U)
        V_i = snap_to_int(V)
        W_i = snap_to_int(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None, None

    def strassen_2x2x2():
        """Return U, V, W factors (2x2x2) with rank 7, integer entries."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # P1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1; V[0,0] = 1; V[3,0] = 1; W[0,0] = 1; W[3,0] = 1
        # P2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1; V[0,1] = 1; W[3,1] = 1; W[0,1] = -1
        # P3 = A00(B01-B11)
        U[0,2] = 1; V[1,2] = 1; V[3,2] = -1; W[1,2] = 1; W[3,2] = 1
        # P4 = A11(B10-B00)
        U[3,3] = 1; V[2,3] = 1; V[0,3] = -1; W[0,3] = 1; W[3,3] = 1
        # P5 = (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1; V[3,4] = 1; W[0,4] = -1; W[1,4] = 1
        # P6 = (A10-A00)(B00+B01)
        U[2,5] = 1; U[0,5] = -1; V[0,5] = 1; V[1,5] = 1; W[3,5] = 1
        # P7 = (A01-A11)(B10+B11)
        U[1,6] = 1; U[3,6] = -1; V[2,6] = 1; V[3,6] = 1; W[0,6] = 1
        return U, V, W

    def optimize_subtensor(T_sub, rank, U_rows, V_rows, W_rows, max_iter=500, lr=0.02, seed=42, atol=1e-6):
        """Optimize factors for a sub-tensor."""
        np.random.seed(seed)
        U = np.random.randn(U_rows, rank) * 0.1
        V = np.random.randn(V_rows, rank) * 0.1
        W = np.random.randn(W_rows, rank) * 0.1
        
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T_sub
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if loss < atol:
                    break
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            t = it + 1
            mU = beta1*mU + (1-beta1)*gradU; vU = beta2*vU + (1-beta2)*(gradU**2)
            U -= lr * (mU/(1-beta1**t)) / (np.sqrt(vU/(1-beta2**t)) + eps)
            
            mV = beta1*mV + (1-beta1)*gradV; vV = beta2*vV + (1-beta2)*(gradV**2)
            V -= lr * (mV/(1-beta1**t)) / (np.sqrt(vV/(1-beta2**t)) + eps)
            
            mW = beta1*mW + (1-beta1)*gradW; vW = beta2*vW + (1-beta2)*(gradW**2)
            W -= lr * (mW/(1-beta1**t)) / (np.sqrt(vW/(1-beta2**t)) + eps)
        
        valid, Us, Vs, Ws = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return Us, Vs, Ws
        return None, None, None

    def build_rank41_2x3():
        """Build 2x5x5 using 2+3 partition: ranks 7+10+10+14=41."""
        U = np.zeros((U_dim, 41))
        V = np.zeros((V_dim, 41))
        W = np.zeros((W_dim, 41))
        
        # Block 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2], rank 7
        U1, V1, W1 = strassen_2x2x2()
        # Map indices: U[0:4, 0:4] for i=0,1; j=0,1
        U[0:4, 0:7] = U1
        V[0:4, 0:7] = V1
        W[0:4, 0:7] = W1
        col = 7
        
        # Block 2: A[0:2,0:2] x B[2:5,0:2] -> C[2:5,0:2], rank 10 (2x2x3)
        # T_sub: u=i*2+j (i=0,1; j=0,1), v=j*3+k (j=0,1; k=0,1,2), w=k*2+i (k=0,1,2; i=0,1)
        T2 = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T2[i*2+j, j*3+k, k*2+i] = 1.0
        U2, V2, W2 = optimize_subtensor(T2, 10, 4, 6, 6, max_iter=400, seed=100)
        if U2 is None:
            return None
        U[0:4, col:col+10] = U2
        V[6:12, col:col+10] = V2  # j=0,1 maps to v indices 0*3+0..1*3+2 = 0..5, but B[2:5,0:2] means j_global=2,3,4 -> v = 2*5+k, 3*5+k, 4*5+k
        # Wait, I need to be careful with index mapping
        # For block 2: A[i, j] where i=0,1; j=0,1 (local) -> j_global = 0,1
        # B[j_global, k] where j_global=0,1; k=2,3,4 -> v = j_global*5 + k
        # C[k, i] where k=2,3,4; i=0,1 -> w = k*2 + i
        # So V local index (j_local, k_local) -> v_global = j_local*5 + (k_local+2)
        # W local index (k_local, i) -> w_global = (k_local+2)*2 + i
        V[0:5, col:col+10] = 0  # Clear
        for j_local in range(2):
            for k_local in range(3):
                v_global = j_local * 5 + (k_local + 2)
                V[v_global, col:col+10] = V2[j_local*3+k_local, :]
        for k_local in range(3):
            for i in range(2):
                w_global = (k_local + 2) * 2 + i
                W[w_global, col:col+10] = W2[k_local*2+i, :]
        col += 10
        
        # Block 3: A[0:2,2:5] x B[0:2,2:5] -> C[0:2,2:5], rank 10 (2x3x2)
        # T_sub: u=i*3+j (i=0,1; j=0,1,2), v=j*2+k (j=0,1; k=0,1), w=k*2+i (k=0,1; i=0,1)
        T3 = np.zeros((6, 4, 4))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    T3[i*3+j, j*2+k, k*2+i] = 1.0
        U3, V3, W3 = optimize_subtensor(T3, 10, 6, 4, 4, max_iter=400, seed=200)
        if U3 is None:
            return None
        # A[i, j_global] where i=0,1; j_global=2,3,4 -> u = i*5 + j_global
        # B[j_global, k_global] where j_global=0,1; k_global=2,3,4 -> v = j_global*5 + k_global
        # C[k_global, i] where k_global=0,1; i=0,1 -> w = k_global*2 + i
        for i in range(2):
            for j_local in range(3):
                u_global = i * 5 + (j_local + 2)
                U[u_global, col:col+10] = U3[i*3+j_local, :]
        for j_local in range(2):
            for k_local in range(2):
                v_global = j_local * 5 + (k_local + 2)
                V[v_global, col:col+10] = V3[j_local*2+k_local, :]
        for k_local in range(2):
            for i in range(2):
                w_global = k_local * 2 + i
                W[w_global, col:col+10] = W3[k_local*2+i, :]
        col += 10
        
        # Block 4: A[0:2,2:5] x B[2:5,2:5] -> C[2:5,2:5], rank 14 (2x3x3)
        # T_sub: u=i*3+j (i=0,1; j=0,1,2), v=j*3+k (j=0,1,2; k=0,1,2), w=k*2+i (k=0,1,2; i=0,1)
        T4 = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    T4[i*3+j, j*3+k, k*2+i] = 1.0
        U4, V4, W4 = optimize_subtensor(T4, 14, 6, 9, 6, max_iter=600, seed=300)
        if U4 is None:
            return None
        for i in range(2):
            for j_local in range(3):
                u_global = i * 5 + (j_local + 2)
                U[u_global, col:col+14] = U4[i*3+j_local, :]
        for j_local in range(3):
            for k_local in range(3):
                v_global = (j_local + 2) * 5 + (k_local + 2)
                V[v_global, col:col+14] = V4[j_local*3+k_local, :]
        for k_local in range(3):
            for i in range(2):
                w_global = (k_local + 2) * 2 + i
                W[w_global, col:col+14] = W4[k_local*2+i, :]
        col += 14
        
        return U[:, :col], V[:, :col], W[:, :col]

    def build_rank46_fallback():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            U[0*m+j_off][c] += 1; U[1*m+(j_off+1)][c] += 1
            V[j_off*p+k_off][c] += 1; V[(j_off+1)*p+(k_off+1)][c] += 1
            W[k_off*n+0][c] += 1; W[(k_off+1)*n+1][c] += 1
            c += 1
            U[1*m+j_off][c] += 1; U[1*m+(j_off+1)][c] += 1
            V[j_off*p+k_off][c] += 1
            W[k_off*n+1][c] += 1; W[(k_off+1)*n+1][c] -= 1
            c += 1
            U[0*m+j_off][c] += 1
            V[j_off*p+(k_off+1)][c] += 1; V[(j_off+1)*p+(k_off+1)][c] -= 1
            W[(k_off+1)*n+0][c] += 1; W[(k_off+1)*n+1][c] += 1
            c += 1
            U[1*m+(j_off+1)][c] += 1
            V[(j_off+1)*p+k_off][c] += 1; V[j_off*p+k_off][c] -= 1
            W[k_off*n+0][c] += 1; W[k_off*n+1][c] += 1
            c += 1
            U[0*m+j_off][c] += 1; U[0*m+(j_off+1)][c] += 1
            V[(j_off+1)*p+(k_off+1)][c] += 1
            W[k_off*n+0][c] -= 1; W[(k_off+1)*n+0][c] += 1
            c += 1
            U[1*m+j_off][c] += 1; U[0*m+j_off][c] -= 1
            V[j_off*p+k_off][c] += 1; V[j_off*p+(k_off+1)][c] += 1
            W[(k_off+1)*n+1][c] += 1
            c += 1
            U[0*m+(j_off+1)][c] += 1; U[1*m+(j_off+1)][c] -= 1
            V[(j_off+1)*p+k_off][c] += 1; V[(j_off+1)*p+(k_off+1)][c] += 1
            W[k_off*n+0][c] += 1
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        col = 28
        for i in range(2):
            for k in range(4):
                U[i*m+4][col] = 1; V[4*p+k][col] = 1; W[k*n+i][col] = 1; col += 1
        for i in range(2):
            for j in range(4):
                U[i*m+j][col] = 1; V[j*p+4][col] = 1; W[4*n+i][col] = 1; col += 1
        for i in range(2):
            U[i*m+4][col] = 1; V[4*p+4][col] = 1; W[4*n+i][col] = 1; col += 1
        return U, V, W

    # Try rank 41 first
    result_41 = build_rank41_2x3()
    if result_41 is not None:
        U_r, V_r, W_r = result_41
        valid, Us, Vs, Ws = try_snap_validate(U_r, V_r, W_r)
        if valid:
            return {"U": [[float(x) for x in row] for row in Us],
                    "V": [[float(x) for x in row] for row in Vs],
                    "W": [[float(x) for x in row] for row in Ws]}
    
    # Fallback to rank 46
    U46, V46, W46 = build_rank46_fallback()
    return {"U": [[float(x) for x in row] for row in U46],
            "V": [[float(x) for x in row] for row in V46],
            "W": [[float(x) for x in row] for row in W46]}
# EVOLVE-BLOCK-END
