import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115

    def verify_decomp(U, V, W, trials=20):
        """Fast verification using numpy vectorized ops."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        rank = Ua.shape[1]
        for _ in range(trials):
            A = np.random.randint(-10, 11, size=(n, m))
            B = np.random.randint(-10, 11, size=(m, p))
            C_true = A @ B
            u_vals = (Ua * A.flatten()[:, None]).sum(axis=0)
            v_vals = (Va * B.flatten()[:, None]).sum(axis=0)
            W_r = Wa.reshape(p, n, rank)
            C_calc = (W_r * (u_vals * v_vals)[None, None, :]).sum(axis=2).T
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """Build the known rank-25 Strassen-based decomposition for 2x2 * 2x7."""
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # M1 = (A00+A11)*(B0,ks + B1,ks+1)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1

            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1

            # M3 = A00*(B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1

            # M4 = A11*(B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1

            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1+k_start+1, col_start+4] += 1
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1

            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
            W[(k_start+1)*n+1, col_start+5] += 1

            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
            W[k_start*n+0, col_start+6] += 1

        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)

        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
        U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
        U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
        U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_rank24_attempt():
        """
        Try to achieve rank 24 by merging the last schoolbook column with Strassen blocks.
        The idea: instead of 3 Strassen blocks (7 each) + 4 schoolbook = 25,
        try to use a different decomposition that shares some terms.
        
        For 2x2 * 2x7, we can try:
        - Use Strassen for columns 0-1 (rank 7)
        - Use Strassen for columns 2-3 (rank 7)  
        - Use Strassen for columns 4-5 (rank 7)
        - For column 6, instead of 4 schoolbook terms, see if we can express
          the column 6 product using some of the Strassen intermediates.
        
        Actually, let's try a different approach: use a 2x2*2x3 block (rank 10?) 
        + 2x2*2x4 block (rank 14?) = 24? No, that's worse.
        
        Let me try: 2x2*2x7 = 2x2*2x2 + 2x2*2x5
        Rank = 7 + ? 
        For 2x2*2x5: schoolbook = 10, Strassen-like = ?
        2x2*2x5 = 2x2*2x2 + 2x2*2x3 = 7 + 6 = 13? Total = 20? No, that's not right.
        
        Actually, the minimal rank for 2x2*2xk is not simply additive.
        
        Let me try a direct rank-24 construction by modifying the Strassen patterns.
        """
        rank = 24
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Use Strassen for columns 0-1
        add_strassen = lambda cs, ks: None  # Placeholder
        
        # For now, return None to indicate this approach didn't work out
        return None

    # First, try the known rank-25 solution
    result = build_rank25()
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
