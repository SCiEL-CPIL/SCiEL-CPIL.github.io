import numpy as np
import time

# Define the Strassen algorithm factors for 2x2 matrix multiplication
# U, V, W are 4x7 matrices corresponding to the 7 products
def get_strassen_2x2():
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices: row = i*2+j (for U/V) or k*2+i (for W)
    # U acts on A (indices i,j), V acts on B (indices j,k), W acts on C (indices k,i)
    
    # Product 0: M0 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # Product 1: M1 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1                # B00
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, -C11
    
    # Product 2: M2 = A00(B01-B11)
    U2[0, 2] = 1                # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, -B11
    W2[2, 2] = 1; W2[3, 2] = 1  # C01, C11
    
    # Product 3: M3 = A11(B10-B00)
    U2[3, 3] = 1                # A11
    V2[0, 3] = -1; V2[2, 3] = 1 # -B00, B10
    W2[0, 3] = 1; W2[1, 3] = 1  # C00, C10
    
    # Product 4: M4 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1  # A00, A01
    V2[3, 4] = 1                # B11
    W2[0, 4] = -1; W2[2, 4] = 1 # -C00, C01
    
    # Product 5: M5 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1 # -A00, A10
    V2[0, 5] = 1; V2[1, 5] = 1  # B00, B01
    W2[3, 5] = 1                # C11
    
    # Product 6: M6 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, -A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10, B11
    W2[0, 6] = 1                # C00
    
    return U2, V2, W2

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # 1. Get 2x2x2 Strassen factors
    U2, V2, W2 = get_strassen_2x2()
    
    # 2. Construct 4x4x4 factors using Kronecker product
    # The decomposition for (n*m) x (m*p) is built from (n/2 * m/2) x (m/2 * p/2)
    # U4 = U2 \u2295 U2, V4 = V2 \u2295 V2, W4 = W2 \u2295 W2
    # Shapes:
    # U2 is (4, 7). U4 will be (16, 49).
    # V2 is (4, 7). V4 will be (16, 49).
    # W2 is (4, 7). W4 will be (16, 49).
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Verify dimensions for 4x4x4
    # U4: (4*4, 7*7) = (16, 49)
    # V4: (4*4, 7*7) = (16, 49)
    # W4: (4*4, 7*7) = (16, 49)
    
    # 3. Construct 4x4x8 factors
    # We multiply a 4x4 matrix A by a 4x8 matrix B.
    # B = [B_left, B_right] where B_left and B_right are 4x4.
    # C = [C_left, C_right]
    # C_left = A * B_left
    # C_right = A * B_right
    # Since the multiplication of A is independent for the two blocks of B,
    # we can reuse the 4x4x4 decomposition for both blocks.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (columns 0-3 of output C, corresponding to rows 0-15 of V and 0-15 of W)
    # U is the same for both blocks in terms of A's coefficients
    U[:, 0:49] = U4
    
    # V for left block: rows 0-15 (corresponding to B_left)
    V[0:16, 0:49] = V4
    
    # W for left block: rows 0-15 (corresponding to C_left)
    W[0:16, 0:49] = W4
    
    # Right block (columns 4-7 of output C, corresponding to rows 16-31 of V and 16-31 of W)
    U[:, 49:98] = U4
    
    # V for right block: rows 16-31 (corresponding to B_right)
    V[16:32, 49:98] = V4
    
    # W for right block: rows 16-31 (corresponding to C_right)
    W[16:32, 49:98] = W4
    
    # Check time limit
    if time.time() - start_time > 140:
        # This is extremely fast, so this shouldn't trigger, but good for safety
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
