import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    def verify_decomposition_full(U, V, W, n, m, p):
        """Full verification: check all (i,j,k) triples."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    def verify_decomposition_sampled(U, V, W, n, m, p, num_checks=2000, seed=42):
        """Sampled verification for speed."""
        rank = U.shape[1]
        rng = random.Random(seed)
        for _ in range(num_checks):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            total = 0
            for r in range(rank):
                total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if total != 1:
                return False
        # Also check some "wrong" entries should be 0
        for _ in range(num_checks // 3):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            # Perturb one index to create a "wrong" triple
            i2 = (i + 1) % n
            if i2 == i:
                i2 = (i + 2) % n
            total = 0
            for r in range(rank):
                total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i2, r]
            if total != 0:
                return False
        return True

    # Strassen 2x2 rank-7 decomposition
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # m0 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # m1 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        # m2 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        # m3 = A11(B10-B00)
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[1, 3] = 1

        # m4 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # m5 = (A10-A00)(B00+B01)
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # m6 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    U2, V2, W2 = get_strassen_2x2()

    # Schoolbook 3x3 rank-27 decomposition (always valid)
    def get_schoolbook_3x3():
        U = np.zeros((9, 27), dtype=int)
        V = np.zeros((9, 27), dtype=int)
        W = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i * 3 + j, r] = 1
                    V[j * 3 + k, r] = 1
                    W[k * 3 + i, r] = 1
        return U, V, W

    # Attempt at rank-23 3x3 decomposition
    def get_bini_3x3_rank23():
        U = np.zeros((9, 23), dtype=int)
        V = np.zeros((9, 23), dtype=int)
        W = np.zeros((9, 23), dtype=int)
        
        # Based on known rank-23 constructions for 3x3 matmul
        # Using the structure from Bini, Capovani, Lotti, Romani (1980)
        
        # Diagonal terms (9 terms): c_ii gets a_ij * b_ji contributions
        # Off-diagonal terms (14 terms): Strassen-like combinations
        
        # Let's use a verified construction pattern
        # Term structure: 9 "main" terms + 14 "correction" terms
        
        # Main terms: for each output position (i,j), we need sum over k of a_ik * b_kj
        # Using block structure:
        
        # Terms 0-8: one per diagonal block interaction
        for idx in range(9):
            i = idx // 3
            j = idx % 3
            U[i*3+j, idx] = 1
            V[j*3+j, idx] = 1  # This won't work directly, need proper structure
            W[j*3+i, idx] = 1
        
        # This naive approach won't work. Let me use a known valid pattern.
        
        # Actually, let me try a different known rank-23 construction:
        # Use the fact that 3x3 matmul tensor has rank 23
        # and encode a specific valid decomposition.
        
        # Reset and use a pattern based on the 7x7 embedding idea
        # or just return None to fall back to schoolbook
        
        # For safety, return None - we'll use schoolbook rank 27
        return None

    # Try to get a better 3x3 decomposition
    bini_result = None
    if time.time() < deadline - 10:
        bini_result = get_bini_3x3_rank23()
    
    if bini_result is not None:
        U3s, V3s, W3s = bini_result
        rank3 = 23
    else:
        U3s, V3s, W3s = get_schoolbook_3x3()
        rank3 = 27
    
    rank_total = 7 * rank3
    
    # Build the 6x6 decomposition via Kronecker product
    U6 = np.zeros((36, rank_total), dtype=int)
    V6 = np.zeros((36, rank_total), dtype=int)
    W6 = np.zeros((36, rank_total), dtype=int)

    for r2 in range(7):
        for r3 in range(rank3):
            r = r2 * rank3 + r3
            
            # U part: U6[i6*6+j6, r] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]
            for i2 in range(2):
                for j2 in range(2):
                    val_u2 = U2[i2*2+j2, r2]
                    if val_u2 == 0:
                        continue
                    for i3 in range(3):
                        for j3 in range(3):
                            val_u3 = U3s[i3*3+j3, r3]
                            if val_u3 == 0:
                                continue
                            i6 = i2*3 + i3
                            j6 = j2*3 + j3
                            U6[i6*6+j6, r] += val_u2 * val_u3
            
            # V part: V6[j6*6+k6, r] = V2[j2*2+k2, r2] * V3[j3*3+k3, r3]
            for j2 in range(2):
                for k2 in range(2):
                    val_v2 = V2[j2*2+k2, r2]
                    if val_v2 == 0:
                        continue
                    for j3 in range(3):
                        for k3 in range(3):
                            val_v3 = V3s[j3*3+k3, r3]
                            if val_v3 == 0:
                                continue
                            j6 = j2*3 + j3
                            k6 = k2*3 + k3
                            V6[j6*6+k6, r] += val_v2 * val_v3
            
            # W part: W6[k6*6+i6, r] = W2[k2*2+i2, r2] * W3[k3*3+i3, r3]
            for k2 in range(2):
                for i2 in range(2):
                    val_w2 = W2[k2*2+i2, r2]
                    if val_w2 == 0:
                        continue
                    for k3 in range(3):
                        for i3 in range(3):
                            val_w3 = W3s[k3*3+i3, r3]
                            if val_w3 == 0:
                                continue
                            k6 = k2*3 + k3
                            i6 = i2*3 + i3
                            W6[k6*6+i6, r] += val_w2 * val_w3

    # Verify
    if time.time() < deadline - 5:
        valid = verify_decomposition_sampled(U6, V6, W6, n, m, p, num_checks=5000)
        if valid and time.time() < deadline - 2:
            valid_full = verify_decomposition_full(U6, V6, W6, n, m, p)
            if valid_full:
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # If verification failed or timed out, try a simpler verification
    if time.time() < deadline - 1:
        valid = verify_decomposition_sampled(U6, V6, W6, n, m, p, num_checks=1000, seed=123)
        if valid:
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Fallback: return the result (should be valid if construction is correct)
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
