import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Shapes
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6

    # We construct the rank-17 decomposition explicitly.
    # The 2x3x3 multiplication can be decomposed into:
    # 1. A 2x2x2 multiplication (rows 0,1 of A, cols 0,1 of B, rows 0,1 of B) -> Rank 7 (Strassen)
    # 2. A 2x3x1 multiplication (rows 0,1 of A, cols 2 of B, rows 0,1 of B) -> Rank 6 (Schoolbook)
    # 3. A 2x1x2 multiplication (rows 0,1 of A, cols 0,1 of B, row 2 of B) -> Rank 4 (Schoolbook)
    # Total rank = 7 + 6 + 4 = 17.
    
    # We will build U, V, W as horizontal concatenations of the decompositions for each part.
    # U shape: (6, 17)
    # V shape: (9, 17)
    # W shape: (6, 17)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Part 1: 2x2x2 using Strassen ---
    # Indices: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen formulas for C = A B:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map global indices to local indices for this subproblem.
    # A local (i, j) -> global (i, j)
    # B local (j, k) -> global (j, k)
    # C local (i, k) -> global (i, k)
    
    U1 = np.zeros((6, 7))
    V1 = np.zeros((9, 7))
    W1 = np.zeros((6, 7))
    
    # M1: (A00 + A11) * (B00 + B11)
    # U1[0, 0] += 1 (A00)
    # U1[3, 0] += 1 (A11)
    # V1[0, 0] += 1 (B00)
    # V1[3, 0] += 1 (B11)
    # W1[0, 0] += 1 (C00)
    # W1[3, 0] += 1 (C11)
    U1[0, 0] = 1
    U1[3, 0] = 1
    V1[0, 0] = 1
    V1[3, 0] = 1
    W1[0, 0] = 1
    W1[3, 0] = 1
    
    # M2: (A10 + A11) * B00
    # U1[1, 1] += 1 (A10)
    # U1[3, 1] += 1 (A11)
    # V1[0, 1] += 1 (B00)
    # W1[2, 1] += 1 (C10)
    U1[1, 1] = 1
    U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1
    
    # M3: A00 * (B01 - B11)
    # U1[0, 2] += 1 (A00)
    # V1[1, 2] += 1 (B01)
    # V1[3, 2] -= 1 (B11)
    # W1[1, 2] += 1 (C01)
    # W1[3, 2] += 1 (C11)
    U1[0, 2] = 1
    V1[1, 2] = 1
    V1[3, 2] = -1
    W1[1, 2] = 1
    W1[3, 2] = 1
    
    # M4: A11 * (B10 - B00)
    # U1[3, 3] += 1 (A11)
    # V1[2, 3] += 1 (B10)
    # V1[0, 3] -= 1 (B00)
    # W1[0, 3] += 1 (C00)
    # W1[2, 3] += 1 (C10)
    U1[3, 3] = 1
    V1[2, 3] = 1
    V1[0, 3] = -1
    W1[0, 3] = 1
    W1[2, 3] = 1
    
    # M5: (A00 + A01) * B11
    # U1[0, 4] += 1 (A00)
    # U1[1, 4] += 1 (A01)
    # V1[3, 4] += 1 (B11)
    # W1[0, 4] -= 1 (C00)
    # W1[1, 4] += 1 (C01)
    U1[0, 4] = 1
    U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1
    W1[1, 4] = 1
    
    # M6: (A10 - A00) * (B00 + B01)
    # U1[1, 5] += 1 (A10)
    # U1[0, 5] -= 1 (A00)
    # V1[0, 5] += 1 (B00)
    # V1[1, 5] += 1 (B01)
    # W1[3, 5] += 1 (C11)
    U1[1, 5] = 1
    U1[0, 5] = -1
    V1[0, 5] = 1
    V1[1, 5] = 1
    W1[3, 5] = 1
    
    # M7: (A01 - A11) * (B10 + B11)
    # U1[1, 6] += 1 (A01)
    # U1[3, 6] -= 1 (A11)
    # V1[2, 6] += 1 (B10)
    # V1[3, 6] += 1 (B11)
    # W1[0, 6] += 1 (C00)
    U1[1, 6] = 1
    U1[3, 6] = -1
    V1[2, 6] = 1
    V1[3, 6] = 1
    W1[0, 6] = 1
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # --- Part 2: 2x3x1 Schoolbook ---
    # Indices: A[0:2, 0:3], B[0:3, 2] -> C[0:2, 2]
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    U2 = np.zeros((6, 6))
    V2 = np.zeros((9, 6))
    W2 = np.zeros((6, 6))
    
    # Term 1: A00 * B02 -> C02
    U2[0, 0] = 1
    V2[2, 0] = 1 # B[0, 2] is index 2
    W2[2, 0] = 1 # C[0, 2] is index 2
    
    # Term 2: A01 * B12 -> C02
    U2[1, 1] = 1
    V2[5, 1] = 1 # B[1, 2] is index 5
    W2[2, 1] = 1
    
    # Term 3: A02 * B22 -> C02
    U2[2, 2] = 1
    V2[8, 2] = 1 # B[2, 2] is index 8
    W2[2, 2] = 1
    
    # Term 4: A10 * B02 -> C12
    U2[3, 3] = 1
    V2[2, 3] = 1
    W2[5, 3] = 1 # C[1, 2] is index 5
    
    # Term 5: A11 * B12 -> C12
    U2[4, 4] = 1
    V2[5, 4] = 1
    W2[5, 4] = 1
    
    # Term 6: A12 * B22 -> C12
    U2[5, 5] = 1
    V2[8, 5] = 1
    W2[5, 5] = 1
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # --- Part 3: 2x1x2 Schoolbook ---
    # Indices: A[0:2, 0:2], B[2, 0:2] -> C[0:2, 0:2]
    # C00 += A00*B20 + A01*B21
    # C01 += A00*B20? No.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Here j is fixed to 2.
    # C00 += A02 * B20
    # C01 += A02 * B21
    # C10 += A12 * B20
    # C11 += A12 * B21
    
    U3 = np.zeros((6, 4))
    V3 = np.zeros((9, 4))
    W3 = np.zeros((6, 4))
    
    # Term 1: A02 * B20 -> C00
    U3[2, 0] = 1
    V3[6, 0] = 1 # B[2, 0] is index 6
    W3[0, 0] = 1 # C[0, 0] is index 0
    
    # Term 2: A02 * B21 -> C01
    U3[2, 1] = 1
    V3[7, 1] = 1 # B[2, 1] is index 7
    W3[1, 1] = 1 # C[0, 1] is index 1
    
    # Term 3: A12 * B20 -> C10
    U3[5, 2] = 1
    V3[6, 2] = 1
    W3[2, 2] = 1 # C[1, 0] is index 2
    
    # Term 4: A12 * B21 -> C11
    U3[5, 3] = 1
    V3[7, 3] = 1
    W3[3, 3] = 1 # C[1, 1] is index 3
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    # Verification
    def check_solution(U, V, W, R):
        np.random.seed(123)
        for _ in range(20):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_val = np.dot(U[:, r], A_flat)
                v_val = np.dot(V[:, r], B_flat)
                w_vec = W[:, r]
                for i in range(n):
                    for k in range(p):
                        C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
                        
            if not np.allclose(C_approx, C_true, atol=1e-4):
                return False
        return True
        
    if not check_solution(U, V, W, U.shape[1]):
        # Fallback to schoolbook if something went wrong (should not happen)
        best_rank = 18
        best_U = np.zeros((dim_U, best_rank))
        best_V = np.zeros((dim_V, best_rank))
        best_W = np.zeros((dim_W, best_rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, column] = 1
                    best_V[j * p + k, column] = 1
                    best_W[k * n + i, column] = 1
                    column += 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
