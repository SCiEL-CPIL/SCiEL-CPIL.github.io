import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 125

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Rank 26 decomposition
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1.0):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx, col] += val
        for idx in v_indices:
            V[idx, col] += val
        for idx in w_indices:
            W[idx, col] += val
        col += 1

    # 1. j=0 block (2x1x6) Rank 1
    # U[i, 0, 0] = A[i,0] -> U[i*5+0, 0] = 1
    # V[0, k, 0] = B[0,k] -> V[0*6+k, 0] = 1
    # W[k, i, 0] = 1 -> W[k*2+i, 0] = 1
    u_idx = [0*5+0, 1*5+0]
    v_idx = [0*6+k for k in range(6)]
    w_idx = [k*2+i for k in range(6) for i in range(2)]
    add_term(u_idx, v_idx, w_idx)
    
    # 2. j=1 block (2x1x6) Rank 1
    u_idx = [0*5+1, 1*5+1]
    v_idx = [1*6+k for k in range(6)]
    w_idx = [k*2+i for k in range(6) for i in range(2)]
    add_term(u_idx, v_idx, w_idx)
    
    # 3. j in {2,3,4}, k in {0,1,2} (2x3x3) Rank 12
    # Local j: 0,1,2 correspond to global 2,3,4
    # Local k: 0,1,2 correspond to global 0,1,2
    
    # 3a. A[2x2] B[2x2] (Rank 7, Strassen)
    # Local j 0,1 -> Global 2,3
    # Local k 0,1 -> Global 0,1
    
    # P1 = (A00+A11)(B00+B11)
    u_idx = [0*5+2, 1*5+3]
    v_idx = [2*6+0, 3*6+1]
    w_idx = [0*2+0, 1*2+1]
    add_term(u_idx, v_idx, w_idx)
    
    # P2 = (A10+A11)B00
    u_idx = [1*5+2, 1*5+3]
    v_idx = [2*6+0]
    w_idx = [0*2+1, 1*2+1] # W[k,i] -> W[0,1], W[1,1]
    # Wait, Strassen P2: (A10+A11)B00 contributes to C10 and C11?
    # C10 = A10B00 + A11B10.
    # C11 = A10B01 + A11B11.
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
    # P3 = A00(B01-B11) -> C00 -= P3, C01 += P3
    # P4 = A11(B10-B00) -> C10 += P4, C11 -= P4
    # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
    # P6 = (A10-A00)(B00+B01) -> C10 += P6
    # P7 = (A01-A11)(B10+B11) -> C01 += P7
    
    # Let's re-map indices carefully.
    # Global i=0,1. Global j=2,3. Global k=0,1.
    # A[i,j] -> U[i*5+j, col]
    # B[j,k] -> V[j*6+k, col]
    # C[i,k] -> W[k*2+i, col]
    
    # Reset col to 2 (after j=0, j=1)
    # We are at col=2.
    
    # P1
    u_idx = [0*5+2, 1*5+3] # A02, A13
    v_idx = [2*6+0, 3*6+1] # B20, B31
    w_idx = [0*2+0, 1*2+1] # W00, W11
    add_term(u_idx, v_idx, w_idx)
    
    # P2
    u_idx = [1*5+2, 1*5+3] # A12, A13
    v_idx = [2*6+0] # B20
    w_idx = [0*2+1, 1*2+1] # W01, W11 (Note: W index is k*n+i)
    # C10 is i=1, k=0 -> W[0*2+1]. C11 is i=1, k=1 -> W[1*2+1].
    # P2 adds to C10 and subtracts from C11?
    # In standard Strassen:
    # C10 = P2 + P4 + P6
    # C11 = P1 - P2 - P4 + P5
    # So for P2: W[0,1] += 1, W[1,1] -= 1.
    add_term(u_idx, v_idx, [0*2+1], 1.0)
    add_term([], [], [1*2+1], -1.0)
    # Wait, add_term increments col. I need to handle signs within one term or use multiple calls?
    # The interface allows any integer/half-integer entries.
    # I can just set the values directly.
    
    # Let's rewrite the addition logic to be more flexible or just build the arrays directly.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def set_term(col_idx, u_entries, v_entries, w_entries):
        for idx, val in u_entries:
            U[idx, col_idx] += val
        for idx, val in v_entries:
            V[idx, col_idx] += val
        for idx, val in w_entries:
            W[idx, col_idx] += val

    col = 0
    # j=0
    set_term(col, [(0*5+0, 1), (1*5+0, 1)], 
             [(0*6+k, 1) for k in range(6)], 
             [(k*2+i, 1) for k in range(6) for i in range(2)])
    col += 1
    
    # j=1
    set_term(col, [(0*5+1, 1), (1*5+1, 1)], 
             [(1*6+k, 1) for k in range(6)], 
             [(k*2+i, 1) for k in range(6) for i in range(2)])
    col += 1
    
    # Block j in {2,3}, k in {0,1} (2x2x2 Strassen)
    # P1
    set_term(col, [(0*5+2, 1), (1*5+3, 1)], [(2*6+0, 1), (3*6+1, 1)], [(0*2+0, 1), (1*2+1, 1)])
    col += 1
    # P2
    set_term(col, [(1*5+2, 1), (1*5+3, 1)], [(2*6+0, 1)], [(0*2+1, 1), (1*2+1, -1)])
    col += 1
    # P3
    set_term(col, [(0*5+2, 1)], [(2*6+1, 1), (3*6+1, -1)], [(0*2+0, -1), (0*2+1, 1)])
    col += 1
    # P4
    set_term(col, [(1*5+3, 1)], [(3*6+0, 1), (2*6+0, -1)], [(0*2+0, 1), (1*2+0, -1)])
    col += 1
    # P5
    set_term(col, [(0*5+2, 1), (0*5+3, 1)], [(3*6+1, 1)], [(0*2+0, -1), (0*2+1, 1)])
    col += 1
    # P6
    set_term(col, [(1*5+2, 1), (0*5+2, -1)], [(2*6+0, 1), (2*6+1, 1)], [(0*2+1, 1)])
    col += 1
    # P7
    set_term(col, [(0*5+3, 1), (1*5+3, -1)], [(3*6+0, 1), (3*6+1, 1)], [(0*2+0, 1)])
    col += 1
    
    # Block j in {2,3}, k=2 (2x2x1) Rank 2
    # C02 = A02 B22 + A03 B32
    # C12 = A12 B22 + A13 B32
    # Term 1: A02 B22 -> C02
    set_term(col, [(0*5+2, 1)], [(2*6+2, 1)], [(2*2+0, 1)])
    col += 1
    # Term 2: A12 B22 -> C12
    set_term(col, [(1*5+2, 1)], [(2*6+2, 1)], [(2*2+1, 1)])
    col += 1
    # Wait, I need to cover A03 B32 and A13 B32 too.
    # My previous Rank 2 logic for 2x2x1 was:
    # r=0: U00=1, U10=1, V00=1, V10=0, W00=1, W01=1. (Covers A00B0, A10B0)
    # r=1: U01=1, U11=1, V00=0, V10=1, W00=1, W01=1. (Covers A01B1, A11B1)
    
    # So for j=2,3 and k=2:
    # Term 1 (for j=2 part):
    set_term(col, [(0*5+2, 1), (1*5+2, 1)], [(2*6+2, 1)], [(2*2+0, 1), (2*2+1, 1)])
    col += 1
    # Term 2 (for j=3 part):
    set_term(col, [(0*5+3, 1), (1*5+3, 1)], [(3*6+2, 1)], [(2*2+0, 1), (2*2+1, 1)])
    col += 1
    
    # Block j=4, k in {0,1} (2x1x2) Rank 2
    # C00 = A04 B40, C01 = A04 B41
    # C10 = A14 B40, C11 = A14 B41
    # Term 1:
    set_term(col, [(0*5+4, 1), (1*5+4, 1)], [(4*6+0, 1)], [(0*2+0, 1), (0*2+1, 1)])
    col += 1
    # Term 2:
    set_term(col, [(0*5+4, 1), (1*5+4, 1)], [(4*6+1, 1)], [(0*2+0, 1), (0*2+1, 1)])
    col += 1
    
    # Block j=4, k=2 (2x1x1) Rank 1
    # C02 = A04 B42, C12 = A14 B42
    set_term(col, [(0*5+4, 1), (1*5+4, 1)], [(4*6+2, 1)], [(2*2+0, 1), (2*2+1, 1)])
    col += 1
    
    # Total terms so far: 1 + 1 + 7 + 2 + 2 + 1 = 14.
    # This covers j in {2,3,4} and k in {0,1,2}.
    # Now repeat for k in {3,4,5} with offset 3.
    
    # Block j in {2,3}, k in {3,4} (2x2x2 Strassen)
    # P1
    set_term(col, [(0*5+2, 1), (1*5+3, 1)], [(2*6+3, 1), (3*6+4, 1)], [(3*2+0, 1), (4*2+1, 1)])
    col += 1
    # P2
    set_term(col, [(1*5+2, 1), (1*5+3, 1)], [(2*6+3, 1)], [(3*2+1, 1), (4*2+1, -1)])
    col += 1
    # P3
    set_term(col, [(0*5+2, 1)], [(2*6+4, 1), (3*6+4, -1)], [(3*2+0, -1), (3*2+1, 1)])
    col += 1
    # P4
    set_term(col, [(1*5+3, 1)], [(3*6+3, 1), (2*6+3, -1)], [(3*2+0, 1), (4*2+0, -1)])
    col += 1
    # P5
    set_term(col, [(0*5+2, 1), (0*5+3, 1)], [(3*6+4, 1)], [(3*2+0, -1), (3*2+1, 1)])
    col += 1
    # P6
    set_term(col, [(1*5+2, 1), (0*5+2, -1)], [(2*6+3, 1), (2*6+4, 1)], [(3*2+1, 1)])
    col += 1
    # P7
    set_term(col, [(0*5+3, 1), (1*5+3, -1)], [(3*6+3, 1), (3*6+4, 1)], [(3*2+0, 1)])
    col += 1
    
    # Block j in {2,3}, k=5 (2x2x1) Rank 2
    set_term(col, [(0*5+2, 1), (1*5+2, 1)], [(2*6+5, 1)], [(5*2+0, 1), (5*2+1, 1)])
    col += 1
    set_term(col, [(0*5+3, 1), (1*5+3, 1)], [(3*6+5, 1)], [(5*2+0, 1), (5*2+1, 1)])
    col += 1
    
    # Block j=4, k in {3,4} (2x1x2) Rank 2
    set_term(col, [(0*5+4, 1), (1*5+4, 1)], [(4*6+3, 1)], [(3*2+0, 1), (3*2+1, 1)])
    col += 1
    set_term(col, [(0*5+4, 1), (1*5+4, 1)], [(4*6+4, 1)], [(3*2+0, 1), (3*2+1, 1)])
    col += 1
    
    # Block j=4, k=5 (2x1x1) Rank 1
    set_term(col, [(0*5+4, 1), (1*5+4, 1)], [(4*6+5, 1)], [(5*2+0, 1), (5*2+1, 1)])
    col += 1
    
    # Total rank 26.
    
    if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
        # Fallback to schoolbook
        rank = 60
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
                    rank = c
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

