import random
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 90 for 3x5x6
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # (15, 90)
    V = [[0] * rank for _ in range(m * p)]  # (30, 90)
    W = [[0] * rank for _ in range(p * n)]  # (18, 90)
    
    # Schoolbook decomposition:
    # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    # Should never happen for this small problem
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Verify shapes
    assert len(U) == n * m, f"U rows: {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows: {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows: {len(W)} != {p*n}"
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    # Verify correctness with random test
    import random
    random.seed(12345)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # True product
    C_true = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C_true[i][k] = s
    
    # Reconstructed product via tensor decomposition
    # C[i,k] = sum_c (sum_j U[i*m+j, c]*A[i,j]) * (sum_j V[j*p+k, c]*B[j,k]) * W[k*n+i, c]
    # Wait, that's not right. Let me re-derive.
    # The tensor decomposition gives: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # That doesn't work dimensionally. Let me check the seed code's verification.
    
    # From seed code:
    # sum_u = sum(U[i*m+j][c] * A[i][j] for i in range(n) for j in range(m))
    # sum_v = sum(V[j*p+k][c] * B[j][k] for j in range(m) for k in range(p))
    # C_prime[i][k] += sum_u * sum_v * W[k*n+i][c]
    
    # So: C[i,k] = sum_c W[k*n+i,c] * (sum_{i',j'} U[i'*m+j',c]*A[i',j']) * (sum_{j',k'} V[j'*p+k',c]*B[j',k'])
    # This is: C[i,k] = sum_c W[k*n+i,c] * u_c * v_c
    # where u_c = sum_{i',j'} U[i'*m+j',c]*A[i',j'] and v_c = sum_{j',k'} V[j'*p+k',c]*B[j',k']
    
    C_prime = [[0] * p for _ in range(n)]
    for c in range(rank):
        u_c = 0
        for i in range(n):
            for j in range(m):
                u_c += U[i * m + j][c] * A[i][j]
        v_c = 0
        for j in range(m):
            for k in range(p):
                v_c += V[j * p + k][c] * B[j][k]
        for i in range(n):
            for k in range(p):
                C_prime[i][k] += u_c * v_c * W[k * n + i][c]
    
    for i in range(n):
        for k in range(p):
            assert C_true[i][k] == C_prime[i][k], f"Mismatch at ({i},{k}): {C_true[i][k]} vs {C_prime[i][k]}"
    
    print(f"Decomposition verified. Rank = {rank}")
