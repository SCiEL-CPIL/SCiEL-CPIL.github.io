"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen-based decomposition: rank 28
    # 4 subproblems of 2x2 * 2x2, each with Strassen rank 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen coefficients for 2x2 * 2x2
    # u[f][pi][pj], v[f][pj][qk], w[f][qk][pi]
    u_coeffs = [
        [[1,0],[0,1]],  # M1: P[0,0]+P[1,1]
        [[0,0],[1,1]],  # M2: P[1,0]+P[1,1]
        [[1,0],[0,0]],  # M3: P[0,0]
        [[0,0],[0,1]],  # M4: P[1,1]
        [[1,1],[0,0]],  # M5: P[0,0]+P[0,1]
        [[-1,0],[1,0]], # M6: P[1,0]-P[0,0]
        [[0,1],[0,-1]], # M7: P[0,1]-P[1,1]
    ]
    v_coeffs = [
        [[1,0],[0,1]],  # M1: Q[0,0]+Q[1,1]
        [[1,0],[0,0]],  # M2: Q[0,0]
        [[0,1],[0,-1]], # M3: Q[0,1]-Q[1,1]
        [[-1,0],[1,0]], # M4: Q[1,0]-Q[0,0]
        [[0,0],[0,1]],  # M5: Q[1,1]
        [[1,1],[0,0]],  # M6: Q[0,0]+Q[0,1]
        [[0,0],[1,1]],  # M7: Q[1,0]+Q[1,1]
    ]
    w_coeffs = [
        [[1,0],[0,1]],  # M1: C[0,0]+C[1,1]
        [[0,1],[0,-1]], # M2: C[1,0]-C[1,1]
        [[0,0],[1,1]],  # M3: C[0,1]+C[1,1]
        [[1,1],[0,0]],  # M4: C[0,0]+C[1,0]
        [[-1,1],[0,0]], # M5: -C[0,0]+C[0,1]
        [[0,0],[0,1]],  # M6: C[1,1]
        [[1,0],[0,0]],  # M7: C[0,0]
    ]
    
    # 4 subproblems: (a, b, c) where a=A col block, b=B row block, c=B col block
    # s=0: a=0,b=0,c=0; s=1: a=0,b=0,c=1; s=2: a=1,b=1,c=0; s=3: a=1,b=1,c=1
    for s in range(4):
        a = s // 2
        b = s // 2
        c = s % 2
        for f in range(7):
            col = s * 7 + f
            for pi in range(2):
                for pj in range(2):
                    val = u_coeffs[f][pi][pj]
                    if val != 0:
                        U[pi * m + (2*a + pj)][col] += val
            for pj in range(2):
                for qk in range(2):
                    val = v_coeffs[f][pj][qk]
                    if val != 0:
                        V[(2*b + pj) * p + (2*c + qk)][col] += val
            for qk in range(2):
                for pi in range(2):
                    val = w_coeffs[f][qk][pi]
                    if val != 0:
                        W[(2*c + qk) * n + pi][col] += val
    
    # Verify the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
