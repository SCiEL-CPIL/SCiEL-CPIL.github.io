"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Partition into 8 sub-problems of type (2,4,2)
    # A (4x4) partitioned into 2x2 blocks: A[i_b, j_b] for i_b, j_b in {0,1}
    # B (4x8) partitioned into 2x2 blocks: B[j_b, k_b] for j_b in {0,1}, k_b in {0,1,2,3}
    # C (4x8) partitioned into 2x2 blocks: C[i_b, k_b] for i_b in {0,1}, k_b in {0,1,2,3}
    #
    # C[i_b, k_b] = A[i_b, 0] * B[0, k_b] + A[i_b, 1] * B[1, k_b]
    #
    # Each term is a (2,2,2) matrix product. Using Strassen for each:
    # M*N where M, N are 2x2:
    # P1 = (M[0,0]+M[1,1]) * (N[0,0]+N[1,1])
    # P2 = (M[1,0]+M[1,1]) * N[0,0]
    # P3 = M[0,0] * (N[0,1]-N[1,1])
    # P4 = M[1,1] * (N[1,0]+N[0,0])
    # P5 = (M[0,0]+M[0,1]) * N[1,1]
    # P6 = (M[1,0]-M[0,0]) * (N[0,0]+N[0,1])
    # P7 = (M[0,1]-M[1,1]) * (N[1,0]+N[1,1])
    #
    # Result:
    # R[0,0] = P1 + P4 - P5 + P7
    # R[0,1] = P3 + P5
    # R[1,0] = P2 + P6
    # R[1,1] = P1 - P2 + P4 + P6
    #
    # For C[i_b, k_b] = A[i_b, 0] * B[0, k_b] + A[i_b, 1] * B[1, k_b]:
    # We use Strassen on M = A[i_b, 0], N = B[0, k_b] to get R1
    # We use Strassen on M = A[i_b, 1], N = B[1, k_b] to get R2
    # C[i_b, k_b] = R1 + R2
    #
    # Total multiplications: 8 sub-problems * 2 Strassen * 7 = 112
    
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    col = 0
    
    # Strassen coefficients for P1..P7
    # For M (2x2) and N (2x2):
    # P1 = (M[0,0]+M[1,1]) * (N[0,0]+N[1,1])
    # P2 = (M[1,0]+M[1,1]) * N[0,0]
    # P3 = M[0,0] * (N[0,1]-N[1,1])
    # P4 = M[1,1] * (N[1,0]+N[0,0])
    # P5 = (M[0,0]+M[0,1]) * N[1,1]
    # P6 = (M[1,0]-M[0,0]) * (N[0,0]+N[0,1])
    # P7 = (M[0,1]-M[1,1]) * (N[1,0]+N[1,1])
    
    # For each sub-problem (i_b, k_b) where i_b in {0,1}, k_b in {0,1,2,3}:
    for i_b in range(2):
        for k_b in range(4):
            # Strassen for A[i_b, 0] * B[0, k_b]
            for s in range(7):
                # M = A[i_b, 0] (2x2 block), N = B[0, k_b] (2x2 block)
                # A[i_b, 0] has indices: (i_b*2 + i_l, 0*2 + j_l) for i_l, j_l in {0,1}
                # B[0, k_b] has indices: (0*2 + j_l, k_b*2 + k_l) for j_l, k_l in {0,1}
                
                # Determine which elements of M and N contribute to P_s
                # M[i_l, j_l] is at U index: (i_b*2 + i_l) * m + (0*2 + j_l) = (i_b*2 + i_l)*4 + j_l
                # N[j_l, k_l] is at V index: (0*2 + j_l) * p + (k_b*2 + k_l) = j_l*8 + k_b*2 + k_l
                
                if s == 0:  # P1 = (M[0,0]+M[1,1]) * (N[0,0]+N[1,1])
                    # M coefficients: M[0,0]=1, M[1,1]=1
                    # N coefficients: N[0,0]=1, N[1,1]=1
                    m_coeffs = [(0, 0, 1), (1, 1, 1)]  # (i_l, j_l, coeff)
                    n_coeffs = [(0, 0, 1), (1, 1, 1)]  # (j_l, k_l, coeff)
                elif s == 1:  # P2 = (M[1,0]+M[1,1]) * N[0,0]
                    m_coeffs = [(1, 0, 1), (1, 1, 1)]
                    n_coeffs = [(0, 0, 1)]
                elif s == 2:  # P3 = M[0,0] * (N[0,1]-N[1,1])
                    m_coeffs = [(0, 0, 1)]
                    n_coeffs = [(0, 1, 1), (1, 1, -1)]
                elif s == 3:  # P4 = M[1,1] * (N[1,0]+N[0,0])
                    m_coeffs = [(1, 1, 1)]
                    n_coeffs = [(1, 0, 1), (0, 0, 1)]
                elif s == 4:  # P5 = (M[0,0]+M[0,1]) * N[1,1]
                    m_coeffs = [(0, 0, 1), (0, 1, 1)]
                    n_coeffs = [(1, 1, 1)]
                elif s == 5:  # P6 = (M[1,0]-M[0,0]) * (N[0,0]+N[0,1])
                    m_coeffs = [(1, 0, 1), (0, 0, -1)]
                    n_coeffs = [(0, 0, 1), (0, 1, 1)]
                else:  # s == 6, P7 = (M[0,1]-M[1,1]) * (N[1,0]+N[1,1])
                    m_coeffs = [(0, 1, 1), (1, 1, -1)]
                    n_coeffs = [(1, 0, 1), (1, 1, 1)]
                
                # Set U values
                for (i_l, j_l, coeff) in m_coeffs:
                    u_idx = (i_b * 2 + i_l) * m + (0 * 2 + j_l)
                    U[u_idx][col] += coeff
                
                # Set V values
                for (j_l, k_l, coeff) in n_coeffs:
                    v_idx = (0 * 2 + j_l) * p + (k_b * 2 + k_l)
                    V[v_idx][col] += coeff
                
                # Set W values: W[k*n+i] where k = k_b*2 + k_l, i = i_b*2 + i_l
                # The result R[i_l, k_l] contributes to C[i_b*2+i_l, k_b*2+k_l]
                # For Strassen, the result coefficients are:
                # R[0,0] = P1 + P4 - P5 + P7
                # R[0,1] = P3 + P5
                # R[1,0] = P2 + P6
                # R[1,1] = P1 - P2 + P4 + P6
                
                if s == 0:  # P1
                    w_coeffs = [(0, 0, 1), (1, 1, 1)]  # (i_l, k_l, coeff)
                elif s == 1:  # P2
                    w_coeffs = [(1, 0, 1), (1, 1, -1)]
                elif s == 2:  # P3
                    w_coeffs = [(0, 1, 1)]
                elif s == 3:  # P4
                    w_coeffs = [(0, 0, 1), (1, 1, 1)]
                elif s == 4:  # P5
                    w_coeffs = [(0, 0, -1), (0, 1, 1)]
                elif s == 5:  # P6
                    w_coeffs = [(1, 0, 1), (1, 1, 1)]
                else:  # s == 6, P7
                    w_coeffs = [(0, 0, 1)]
                
                for (i_l, k_l, coeff) in w_coeffs:
                    w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                    W[w_idx][col] += coeff
                
                col += 1
            
            # Strassen for A[i_b, 1] * B[1, k_b]
            for s in range(7):
                # M = A[i_b, 1] (2x2 block), N = B[1, k_b] (2x2 block)
                # A[i_b, 1] has indices: (i_b*2 + i_l, 1*2 + j_l) for i_l, j_l in {0,1}
                # B[1, k_b] has indices: (1*2 + j_l, k_b*2 + k_l) for j_l, k_l in {0,1}
                
                if s == 0:  # P1 = (M[0,0]+M[1,1]) * (N[0,0]+N[1,1])
                    m_coeffs = [(0, 0, 1), (1, 1, 1)]
                    n_coeffs = [(0, 0, 1), (1, 1, 1)]
                elif s == 1:  # P2 = (M[1,0]+M[1,1]) * N[0,0]
                    m_coeffs = [(1, 0, 1), (1, 1, 1)]
                    n_coeffs = [(0, 0, 1)]
                elif s == 2:  # P3 = M[0,0] * (N[0,1]-N[1,1])
                    m_coeffs = [(0, 0, 1)]
                    n_coeffs = [(0, 1, 1), (1, 1, -1)]
                elif s == 3:  # P4 = M[1,1] * (N[1,0]+N[0,0])
                    m_coeffs = [(1, 1, 1)]
                    n_coeffs = [(1, 0, 1), (0, 0, 1)]
                elif s == 4:  # P5 = (M[0,0]+M[0,1]) * N[1,1]
                    m_coeffs = [(0, 0, 1), (0, 1, 1)]
                    n_coeffs = [(1, 1, 1)]
                elif s == 5:  # P6 = (M[1,0]-M[0,0]) * (N[0,0]+N[0,1])
                    m_coeffs = [(1, 0, 1), (0, 0, -1)]
                    n_coeffs = [(0, 0, 1), (0, 1, 1)]
                else:  # s == 6, P7 = (M[0,1]-M[1,1]) * (N[1,0]+N[1,1])
                    m_coeffs = [(0, 1, 1), (1, 1, -1)]
                    n_coeffs = [(1, 0, 1), (1, 1, 1)]
                
                # Set U values
                for (i_l, j_l, coeff) in m_coeffs:
                    u_idx = (i_b * 2 + i_l) * m + (1 * 2 + j_l)
                    U[u_idx][col] += coeff
                
                # Set V values
                for (j_l, k_l, coeff) in n_coeffs:
                    v_idx = (1 * 2 + j_l) * p + (k_b * 2 + k_l)
                    V[v_idx][col] += coeff
                
                # Set W values
                if s == 0:  # P1
                    w_coeffs = [(0, 0, 1), (1, 1, 1)]
                elif s == 1:  # P2
                    w_coeffs = [(1, 0, 1), (1, 1, -1)]
                elif s == 2:  # P3
                    w_coeffs = [(0, 1, 1)]
                elif s == 3:  # P4
                    w_coeffs = [(0, 0, 1), (1, 1, 1)]
                elif s == 4:  # P5
                    w_coeffs = [(0, 0, -1), (0, 1, 1)]
                elif s == 5:  # P6
                    w_coeffs = [(1, 0, 1), (1, 1, 1)]
                else:  # s == 6, P7
                    w_coeffs = [(0, 0, 1)]
                
                for (i_l, k_l, coeff) in w_coeffs:
                    w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                    W[w_idx][col] += coeff
                
                col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
