import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    n, m, p = 2, 4, 8

    # Strassen coefficients for 2x2 matrix multiplication (rank 7)
    # X forms: how A sub-block (2x2) is split
    X_forms = [
        [1, 0, 0, 1],   # m0: A11 + A22
        [0, 0, 1, 1],   # m1: A21 + A22
        [1, 0, 0, 0],   # m2: A11
        [0, 0, 0, 1],   # m3: A22
        [1, 1, 0, 0],   # m4: A11 + A12
        [-1, 0, 1, 0],  # m5: A21 - A11
        [0, 1, 0, -1],  # m6: A22 - A12
    ]
    
    # Y forms: how B sub-block (2x2) is split
    Y_forms = [
        [1, 0, 0, 1],   # m0: B11 + B22
        [1, 0, 0, 0],   # m1: B11
        [0, 1, 0, -1],  # m2: B21 - B11
        [-1, 0, 1, 0],  # m3: B21 - B11 (different indexing)
        [0, 0, 0, 1],   # m4: B22
        [1, 1, 0, 0],   # m5: B11 + B12
        [0, 0, 1, 1],   # m6: B21 + B22
    ]
    
    # Z combos: how to combine 7 products to get C sub-block (2x2)
    # C11 = m0 - m2 + m3 + m5
    # C12 = m4 + m6
    # C21 = m1 + m4
    # C22 = m0 - m4 + m1 + m6  (wait, let me re-derive)
    # Actually standard Strassen:
    # C11 = m0 - m2 + m3 + m5
    # C12 = m4 + m6
    # C21 = m1 + m4
    # C22 = m0 - m4 + m1 + m6
    Z_combos = [
        [1, 0, -1, 1, 0, 1, 0],   # C11: m0 - m2 + m3 + m5
        [0, 0, 0, 0, 1, 0, 1],    # C12: m4 + m6
        [0, 1, 0, 0, 1, 0, 0],    # C21: m1 + m4
        [1, 1, 0, 0, -1, 0, 1],   # C22: m0 + m1 - m4 + m6
    ]
    
    num_groups = 4      # groups of 2 columns in C (8/2)
    num_ablocks = 2     # blocks of 2 columns in A (4/2)
    num_strassen = 7
    rank = num_groups * num_ablocks * num_strassen  # 56
    
    U = [[0] * rank for _ in range(n * m)]   # (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (16, 56)
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # Fill U: A[i, j] where i in [0,2), j = b*2 + jp
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # Fill V: B[j, k] where j = b*2 + jp, k = 2*g + kp
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # Fill W: C[k, i] where k = 2*g + kp, i in [0,2)
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
    
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    
    n, m, p = 2, 4, 8
    assert len(U) == n*m, "U shape error"
    assert len(U[0]) == len(V[0]), "Rank mismatch U V"
    assert len(V) == m*p, "V shape error"
    assert len(W) == p*n, "W shape error"
    assert len(W[0]) == len(U[0]), "Rank mismatch W U"
    
    # Verify correctness with random test
    import random
    random.seed(42)
    A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Compute C' using the decomposition
    C_prime = [[0] * p for _ in range(n)]
    for r in range(len(U[0])):
        u_val = sum(U[i*m + j][r] * A[i][j] for i in range(n) for j in range(m))
        v_val = sum(V[j*p + k][r] * B[j][k] for j in range(m) for k in range(p))
        prod = u_val * v_val
        for k in range(p):
            for i in range(n):
                C_prime[i][k] += W[k*n + i][r] * prod
    
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_prime[i][k]:
                print(f"Mismatch at C[{i}][{k}]: {C[i][k]} vs {C_prime[i][k]}")
                sys.exit(1)
    
    print(f"Verified correct. Rank = {len(U[0])}")
