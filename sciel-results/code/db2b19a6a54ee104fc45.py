import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):
        for j in range(3):
            a_rows = [2*i, 2*i+1]
            a_cols = [0, 1]
            b_rows = [0, 1]
            b_cols = [2*j, 2*j+1]
            c_rows = [2*i, 2*i+1]
            c_cols = [2*j, 2*j+1]
            
            a00 = a_rows[0] * 4 + a_cols[0]
            a01 = a_rows[0] * 4 + a_cols[1]
            a10 = a_rows[1] * 4 + a_cols[0]
            a11 = a_rows[1] * 4 + a_cols[1]
            
            b00 = b_rows[0] * 6 + b_cols[0]
            b01 = b_rows[0] * 6 + b_cols[1]
            b10 = b_rows[1] * 6 + b_cols[0]
            b11 = b_rows[1] * 6 + b_cols[1]
            
            c00 = c_cols[0] * 4 + c_rows[0]
            c01 = c_cols[1] * 4 + c_rows[0]
            c10 = c_cols[0] * 4 + c_rows[1]
            c11 = c_cols[1] * 4 + c_rows[1]
            
            # Strassen for A[i,0] * B[0,j]
            U[a00][col] += 1; U[a11][col] += 1
            V[b00][col] += 1; V[b11][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1
            
            U[a10][col] += 1; U[a11][col] += 1
            V[b00][col] += 1
            W[c10][col] += 1; W[c11][col] += 1
            col += 1
            
            U[a00][col] += 1
            V[b01][col] += 1; V[b11][col] += -1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1
            
            U[a11][col] += 1
            V[b10][col] += 1; V[b00][col] += -1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1
            
            U[a00][col] += 1; U[a01][col] += 1
            V[b11][col] += 1
            W[c00][col] += -1; W[c01][col] += 1
            col += 1
            
            U[a10][col] += 1; U[a00][col] += -1
            V[b00][col] += 1; V[b01][col] += 1
            W[c11][col] += 1
            col += 1
            
            U[a01][col] += 1; U[a11][col] += -1
            V[b10][col] += 1; V[b11][col] += 1
            W[c00][col] += 1; W[c01][col] += -1
            col += 1
            
            # Strassen for A[i,1] * B[1,j]
            a_cols2 = [2, 3]
            b_rows2 = [2, 3]
            
            a00_2 = a_rows[0] * 4 + a_cols2[0]
            a01_2 = a_rows[0] * 4 + a_cols2[1]
            a10_2 = a_rows[1] * 4 + a_cols2[0]
            a11_2 = a_rows[1] * 4 + a_cols2[1]
            
            b00_2 = b_rows2[0] * 6 + b_cols[0]
            b01_2 = b_rows2[0] * 6 + b_cols[1]
            b10_2 = b_rows2[1] * 6 + b_cols[0]
            b11_2 = b_rows2[1] * 6 + b_cols[1]
            
            U[a00_2][col] += 1; U[a11_2][col] += 1
            V[b00_2][col] += 1; V[b11_2][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1
            
            U[a10_2][col] += 1; U[a11_2][col] += 1
            V[b00_2][col] += 1
            W[c10][col] += 1; W[c11][col] += 1
            col += 1
            
            U[a00_2][col] += 1
            V[b01_2][col] += 1; V[b11_2][col] += -1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1
            
            U[a11_2][col] += 1
            V[b10_2][col] += 1; V[b00_2][col] += -1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1
            
            U[a00_2][col] += 1; U[a01_2][col] += 1
            V[b11_2][col] += 1
            W[c00][col] += -1; W[c01][col] += 1
            col += 1
            
            U[a10_2][col] += 1; U[a00_2][col] += -1
            V[b00_2][col] += 1; V[b01_2][col] += 1
            W[c11][col] += 1
            col += 1
            
            U[a01_2][col] += 1; U[a11_2][col] += -1
            V[b10_2][col] += 1; V[b11_2][col] += 1
            W[c00][col] += 1; W[c01][col] += -1
            col += 1
    
    assert col == 84, f"Expected 84, got {col}"
    return {"U": U, "V": V, "W": W}
