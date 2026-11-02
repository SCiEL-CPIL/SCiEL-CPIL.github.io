import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 120.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Minimal rank for 2x2x3 matrix multiplication is known to be 11.
    # We provide a rank-11 decomposition.
    # U shape: (n*m, rank) = (4, 11)
    # V shape: (m*p, rank) = (6, 11)
    # W shape: (p*n, rank) = (6, 11)
    
    rank = 11
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2): a=0, b=1, c=2, d=3
    # Indices for B (2x3): b00=0, b01=1, b02=2, b10=3, b11=4, b12=5
    # Indices for C (2x3): c00=0, c01=1, c02=2, c10=3, c11=4, c12=5
    # Note: The task specifies W[k*n+i, r]. 
    # C is 2x3. k is column index (0..2), i is row index (0..1).
    # So W index = k*2 + i.
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[0,1] -> k=1, i=0 -> idx 2
    # C[0,2] -> k=2, i=0 -> idx 4
    # C[1,0] -> k=0, i=1 -> idx 1
    # C[1,1] -> k=1, i=1 -> idx 3
    # C[1,2] -> k=2, i=1 -> idx 5
    
    # We use a hybrid approach: Strassen for the first 2 columns of B (2x2x2 part) and schoolbook for the 3rd column.
    # This gives rank 7 + 4 = 11.
    
    # --- Strassen part (columns 0 and 1 of B) ---
    # Let B2 be the 2x2 matrix [[b00, b01], [b10, b11]].
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # --- Schoolbook part (column 2 of B) ---
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # --- W matrix (reconstruction) ---
    # C00 = a*b00 + b*b10
    # C01 = a*b01 + b*b11
    # C02 = a*b02 + b*b12
    # C10 = c*b00 + d*b10
    # C11 = c*b01 + d*b11
    # C12 = c*b02 + d*b12
    
    # Using Strassen identities:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C01 = M3 + M5
    
    # Verify C00:
    # M1 = (a+d)(b00+b11) = ab00 + ab11 + db00 + db11
    # M4 = d(b10-b00) = db10 - db00
    # M5 = (a+b)b11 = ab11 + bb11
    # M7 = (b-d)(b10+b11) = bb10 + bb11 - db10 - db11
    # Sum: ab00 + ab11 + db00 + db11 + db10 - db00 - ab11 - bb11 + bb10 + bb11 - db10 - db11
    # = ab00 + db10 + bb10 = a*b00 + b*b10. Correct.
    
    # Verify C01:
    # M3 + M5 = a(b01-b11) + (a+b)b11 = ab01 - ab11 + ab11 + bb11 = ab01 + bb11. Correct.
    
    # Verify C10:
    # M2 + M4 = (c+d)b00 + d(b10-b00) = cb00 + db00 + db10 - db00 = cb00 + db10. Correct.
    
    # Verify C11:
    # M1 - M2 + M3 + M6
    # M1 = ab00 + ab11 + db00 + db11
    # M2 = cb00 + db00
    # M3 = ab01 - ab11
    # M6 = (c-a)(b00+b01) = cb00 + cb01 - ab00 - ab01
    # Sum: (ab00 + ab11 + db00 + db11) - (cb00 + db00) + (ab01 - ab11) + (cb00 + cb01 - ab00 - ab01)
    # = ab00 + ab11 + db00 + db11 - cb00 - db00 + ab01 - ab11 + cb00 + cb01 - ab00 - ab01
    # = db11 + cb01 = c*b01 + d*b11. Correct.
    
    # W indices:
    # C00 -> idx 0
    # C01 -> idx 2
    # C02 -> idx 4
    # C10 -> idx 1
    # C11 -> idx 3
    # C12 -> idx 5
    
    # C00 (idx 0): M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    
    # C10 (idx 1): M2 + M4
    W[1][1] = 1; W[1][3] = 1
    
    # C11 (idx 3): M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # C01 (idx 2): M3 + M5
    W[2][2] = 1; W[2][4] = 1
    
    # C02 (idx 4): M8 + M9
    W[4][7] = 1; W[4][8] = 1
    
    # C12 (idx 5): M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Explicit deadline guard
    if time.time() - start_time > TIME_LIMIT - 10:
        pass # Return best result found (which is this static result)
        
    return {"U": U, "V": V, "W": W}

# Verification function to ensure correctness during development
def _verify():
    n, m, p = DIMENSIONS
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Check shapes
    assert U.shape == (n*m, 11)
    assert V.shape == (m*p, 11)
    assert W.shape == (p*n, 11)
    
    # Reconstruct tensor
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*m+k, r] * W[k*n+i, r]
    # Note: The task says "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the standard decomposition:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor form T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*m+k, r} W_{k*n+i, r}
    # Let's verify with random matrices
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(11):
                    val += U[i*m+j, r] * V[j*m+k, r] * W[k*n+i, r]
            C_comp[i, k] = val
    
    if not np.allclose(C_true, C_comp):
        print("Verification Failed!")
        print(C_true)
        print(C_comp)
    else:
        print("Verification Passed")

if __name__ == "__main__":
    _verify()
