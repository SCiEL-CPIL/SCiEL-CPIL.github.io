import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Base case: p=1, Rank 1
    def get_base_p1():
        # n=2, m=2, p=1
        # U: 4x1, V: 2x1, W: 2x1
        U = np.ones((n*m, 1))
        V = np.ones((m*p, 1))
        W = np.ones((p*n, 1))
        return U, V, W

    # Base case: p=2, Rank 7 (Strassen)
    def get_base_p2():
        # n=2, m=2, p=2
        rank = 7
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11 parts?
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # W indices: k*n + i
        # C00 (k=0, i=0) -> idx 0
        # C11 (k=1, i=1) -> idx 3
        # C01 (k=1, i=0) -> idx 2
        # C10 (k=0, i=1) -> idx 1
        
        # M1 contribution to W
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 = M2 + M4
        W[1, 1] = 1
        # C11 = M1 - M2 + ...
        W[3, 1] = -1
        
        # M3 = A00(B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 = M3 + M5
        W[2, 2] = 1
        # C11 = ... + M3
        W[3, 2] = 1
        
        # M4 = A11(B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 = M1 + M4 ...
        W[0, 3] = 1
        # C10 = M2 + M4
        W[1, 3] = 1
        
        # M5 = (A00 + A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00 = ... - M5
        W[0, 4] = -1
        # C01 = M3 + M5
        W[2, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 = ... + M6
        W[3, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 = ... + M7
        W[0, 6] = 1
        
        return U, V, W

    def combine(U1, V1, W1, U2, V2, W2, p1, p2):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        # U: hstack
        U_full = np.hstack([U1, U2])
        
        # V: vstack with padding
        V_full = np.zeros((m * (p1 + p2), rank))
        V_full[:m * p1, :r1] = V1
        V_full[m * p1:, r1:] = V2
        
        # W: vstack with padding
        W_full = np.zeros((p * n, rank))
        W_full[:p1 * n, :r1] = W1
        W_full[p1 * n:, r1:] = W2
        
        return U_full, V_full, W_full

    def build_decomp(p_val):
        if p_val == 1:
            return get_base_p1()
        elif p_val == 2:
            return get_base_p2()
        else:
            # Split p_val into two parts, preferably powers of 2
            # Greedy split: largest power of 2 <= p_val
            p1 = 1
            while p1 * 2 <= p_val:
                p1 *= 2
            p2 = p_val - p1
            
            U1, V1, W1 = build_decomp(p1)
            U2, V2, W2 = build_decomp(p2)
            
            return combine(U1, V1, W1, U2, V2, W2, p1, p2)

    U, V, W = build_decomp(p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
