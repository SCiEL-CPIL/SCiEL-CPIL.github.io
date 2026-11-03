import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: i, j, k in {0, 1, 2, 3} for 2x2
    # U[i, r], V[j, r], W[k, r]
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # Map 2x2 indices to 1D: 00->0, 01->1, 10->2, 11->3
    # A: [[0, 1], [2, 3]]
    # B: [[0, 1], [2, 3]]
    # C: [[0, 1], [2, 3]]
    
    # M1
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[1, 0] = 1; W_S[2, 0] = 1; W_S[3, 0] = 1
    
    # M2
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1; W_S[3, 1] = 1
    
    # M3
    U_S[0, 2] = 1
    V_S[2, 2] = 1; V_S[3, 2] = -1
    W_S[0, 2] = 1; W_S[2, 2] = 1
    
    # M4
    U_S[3, 3] = 1
    V_S[1, 3] = 1; V_S[0, 3] = -1
    W_S[1, 3] = 1; W_S[3, 3] = 1
    
    # M5
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[1, 4] = 1; W_S[3, 4] = 1
    
    # M6
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[2, 5] = 1; W_S[0, 5] = 1
    
    # M7
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[3, 6] = 1; W_S[1, 6] = 1
    
    # Part 1: 4x4x4 multiplication (first 4 cols of B)
    # Kronecker product S7 x S7
    U_1 = np.kron(U_S, U_S)
    V_1 = np.kron(V_S, V_S)
    W_1 = np.kron(W_S, W_S)
    
    # Part 2: 4x4x2 multiplication (last 2 cols of B)
    # We need 4x4x2.
    # Use S7 (for 4x4) x S2 (for 2x1? No, 2x2x1 is rank 4)
    # Let's use S7 x S_4 where S_4 is 2x2x1 decomposition.
    
    # S_4 for 2x2x1:
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Terms: A00B0, A01B1, A10B0, A11B1
    U_4 = np.zeros((4, 4))
    V_4 = np.zeros((4, 4))
    W_4 = np.zeros((4, 4)) # W is for C (2x1), so size 2? No, W is p*n?
    # For 2x2x1, n=2, m=2, p=1.
    # U: n*m = 4. V: m*p = 2. W: p*n = 2.
    # Wait, the interface is U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # For 2x2x1: n=2, m=2, p=1.
    # U: 4xR. V: 2xR. W: 2xR.
    
    U_4 = np.zeros((4, 4))
    V_4 = np.zeros((2, 4))
    W_4 = np.zeros((2, 4))
    
    # Term 1: A00 B0 -> C0
    U_4[0, 0] = 1
    V_4[0, 0] = 1
    W_4[0, 0] = 1
    
    # Term 2: A01 B1 -> C0
    U_4[1, 1] = 1
    V_4[1, 1] = 1
    W_4[0, 1] = 1
    
    # Term 3: A10 B0 -> C1
    U_4[2, 2] = 1
    V_4[0, 2] = 1
    W_4[1, 2] = 1
    
    # Term 4: A11 B1 -> C1
    U_4[3, 3] = 1
    V_4[1, 3] = 1
    W_4[1, 3] = 1
    
    # Kronecker S7 x S4
    # S7 is 2x2x2. S4 is 2x2x1.
    # Product is 4x4x2.
    U_2 = np.kron(U_S, U_4)
    V_2 = np.kron(V_S, V_4)
    W_2 = np.kron(W_S, W_4)
    
    # Combine
    U_total = np.hstack([U_1, U_2])
    V_total = np.hstack([V_1, V_2])
    W_total = np.hstack([W_1, W_2])
    
    # Convert to lists of integers/half-integers
    # The entries are integers.
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
