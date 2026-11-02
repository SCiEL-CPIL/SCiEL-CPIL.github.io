"""
Combined program: Robust Rank-25 Strassen-based decomposition for 2x2 * 2x7 matrix multiplication.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # Safety margin before 150s kill

    def verify_decomp(U, V, W, trials=50):
        """
        Verify that the decomposition exactly computes matrix multiplication 
        using the trace convention W[k*n+i, r].
        """
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        rank = Ua.shape[1]
        
        for _ in range(trials):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
            C_true = A @ B

            C_calc = np.zeros((n, p))
            for r in range(rank):
                # u_val = sum(U[i*m+j, r] * A[i, j])
                u_val = 0.0
                for i in range(n):
                    for j in range(m):
                        u_val += Ua[i*m+j, r] * A[i, j]
                
                # v_val = sum(V[j*p+k, r] * B[j, k])
                v_val = 0.0
                for j in range(m):
                    for k in range(p):
                        v_val += Va[j*p+k, r] * B[j, k]
                
                prod = u_val * v_val
                
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += Wa[k*n+i, r] * prod

            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """
        Build the known rank-25 Strassen-based decomposition for 2x2 * 2x7.
        Uses Strassen (rank 7) for three 2x2 blocks and schoolbook (rank 4) for the last 2x1 block.
        """
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # M1 = (A00+A11)*(B0,ks + B1,ks+1)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
            # Output: C[0,ks] += M1, C[1,ks+1] += M1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
            
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            # Output: C[1,ks] += M2, C[1,ks+1] -= M2
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
            
            # M3 = A00*(B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
            # Output: C[0,ks+1] += M3, C[1,ks+1] += M3
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
            
            # M4 = A11*(B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
            # Output: C[0,ks] += M4, C[1,ks] += M4
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
            
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1+k_start+1, col_start+4] += 1
            # Output: C[0,ks] -= M5, C[0,ks+1] += M5
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
            
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
            # Output: C[1,ks+1] += M6
            W[(k_start+1)*n+1, col_start+5] += 1
            
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
            # Output: C[0,ks] += M7
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

    # Build and verify the rank-25 decomposition
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
# EVOLVE-BLOCK-END
