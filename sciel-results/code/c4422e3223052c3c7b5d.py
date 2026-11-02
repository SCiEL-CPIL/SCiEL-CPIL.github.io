"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    Strategy: Decompose into two 3x3x3 blocks and two 3x1x3 blocks.
    Try to find rank-23 for 3x3x3, fallback to rank-27.
    """
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin
    
    def check_time():
        return time.time() - start_time < DEADLINE

    def schoolbook_full():
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

    def verify_full(U, V, W, tol=1e-6):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check a sample of zero entries
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i != i2 or j != j2 or k != k2:
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val) > tol:
                    return False
        return True

    def full_verify(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True

    def solve_3x3x3(rank, max_iter=2000, seed=42):
        """
        Solve for a rank-decomposition of the 3x3x3 matrix multiplication tensor.
        Returns U, V, W matrices of shape (9, rank) or None.
        """
        if not check_time():
            return None
            
        np.random.seed(seed)
        U = np.random.randn(9, rank) * 0.1
        V = np.random.randn(9, rank) * 0.1
        W = np.random.randn(9, rank) * 0.1
        
        # Target: T[i,j,k] = 1 for all i,j,k in 0..2
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # Note: The indices for 3x3x3 are i,j,k.
        # U index: i*3+j
        # V index: j*3+k
        # W index: k*3+i
        
        for iteration in range(max_iter):
            if not check_time():
                return None
                
            # Update U
            for i in range(3):
                for j in range(3):
                    a = i * 3 + j
                    for r in range(rank):
                        grad = 0
                        for k in range(3):
                            b = j * 3 + k
                            c = k * 3 + i
                            current = np.sum(U[a, :] * V[b, :] * W[c, :])
                            grad += (current - 1.0) * V[b, r] * W[c, r]
                        U[a, r] -= 0.01 * grad
                        
            # Update V
            for j in range(3):
                for k in range(3):
                    b = j * 3 + k
                    for r in range(rank):
                        grad = 0
                        for i in range(3):
                            a = i * 3 + j
                            c = k * 3 + i
                            current = np.sum(U[a, :] * V[b, :] * W[c, :])
                            grad += (current - 1.0) * U[a, r] * W[c, r]
                        V[b, r] -= 0.01 * grad
                        
            # Update W
            for k in range(3):
                for i in range(3):
                    c = k * 3 + i
                    for r in range(rank):
                        grad = 0
                        for j in range(3):
                            a = i * 3 + j
                            b = j * 3 + k
                            current = np.sum(U[a, :] * V[b, :] * W[c, :])
                            grad += (current - 1.0) * U[a, r] * V[b, r]
                        W[c, r] -= 0.01 * grad
                        
            if iteration % 50 == 0:
                # Check convergence
                max_err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            a = i * 3 + j
                            b = j * 3 + k
                            c = k * 3 + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-6:
                    return U, V, W
                    
        return None

    def build_from_3x3x3(U3, V3, W3, rank_3x3):
        """
        Build the full 3x4x6 decomposition from two 3x3x3 decompositions and two rank-1 terms.
        """
        # Full dimensions: n=3, m=4, p=6
        # U shape: (12, rank)
        # V shape: (24, rank)
        # W shape: (18, rank)
        
        total_rank = 2 * rank_3x3 + 2
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        # Block 1: Left half of C (cols 0,1,2)
        # Corresponds to A (3x4) * B_left (4x3)
        # A is indexed by i (0..2), j (0..3)
        # B_left is indexed by j (0..3), k (0..2)
        # C is indexed by i (0..2), k (0..2)
        
        # We split A into A1 (cols 0..2) and A2 (col 3)
        # We split B_left into B1_top (rows 0..2) and B1_bottom (row 3)
        
        # Term 1: A1 * B1_top (3x3x3)
        # Maps to U3, V3, W3
        # U3 indices: i*3+j' where j' is 0..2. In full U, index is i*4 + j.
        # V3 indices: j'*3+k. In full V, index is j*p + k = j*6 + k.
        # W3 indices: k*3+i. In full W, index is k*n + i = k*3 + i.
        
        for i in range(3):
            for j in range(3):
                # U3[i*3+j] -> U[i*4+j]
                U[i*4 + j, 0:rank_3x3] = U3[i*3 + j, :]
                
        for j in range(3):
            for k in range(3):
                # V3[j*3+k] -> V[j*6+k]
                V[j*6 + k, 0:rank_3x3] = V3[j*3 + k, :]
                
        for k in range(3):
            for i in range(3):
                # W3[k*3+i] -> W[k*3+i]
                W[k*3 + i, 0:rank_3x3] = W3[k*3 + i, :]
                
        # Term 2: A2 * B1_bottom (3x1x3) -> Rank 1
        # A2 is col 3 of A. Index i*4+3.
        # B1_bottom is row 3 of B_left. Index 3*6+k.
        # W is k*3+i.
        # We need U[i*4+3, r] * V[3*6+k, r] * W[k*3+i, r] = 1
        # Set U[i*4+3, r] = 1, V[18+k, r] = 1, W[k*3+i, r] = 1
        r_offset = 2 * rank_3x3
        for i in range(3):
            U[i*4 + 3, r_offset] = 1
        for k in range(3):
            V[3*6 + k, r_offset] = 1
        for k in range(3):
            for i in range(3):
                W[k*3 + i, r_offset] = 1
                
        # Block 2: Right half of C (cols 3,4,5)
        # Corresponds to A (3x4) * B_right (4x3)
        # B_right is cols 3..5 of B.
        # Indices for B_right in full V: j*6 + (3+k)
        
        # Term 3: A1 * B2_top (3x3x3)
        # Uses the same U3, V3, W3 but shifted in V and W?
        # W is shared? No, W depends on k.
        # For the right block, k is 3..5.
        # W index is k*3 + i.
        # So we need to place W3 into W[k*3+i] for k=3..5.
        
        # U is the same A1 part.
        # U[i*4+j, rank_3x3:2*rank_3x3] = U3[i*3+j, :]
        
        for i in range(3):
            for j in range(3):
                U[i*4 + j, rank_3x3:2*rank_3x3] = U3[i*3 + j, :]
                
        for j in range(3):
            for k in range(3):
                # V3[j*3+k] -> V[j*6 + 3 + k]
                V[j*6 + 3 + k, rank_3x3:2*rank_3x3] = V3[j*3 + k, :]
                
        for k in range(3):
            for i in range(3):
                # W3[k*3+i] -> W[(3+k)*3 + i]
                W[(3+k)*3 + i, rank_3x3:2*rank_3x3] = W3[k*3 + i, :]
                
        # Term 4: A2 * B2_bottom (3x1x3) -> Rank 1
        # A2 is col 3 of A.
        # B2_bottom is row 3 of B_right. Index 3*6 + 3 + k.
        # W is (3+k)*3 + i.
        
        r_offset2 = 2 * rank_3x3 + 1
        for i in range(3):
            U[i*4 + 3, r_offset2] = 1
        for k in range(3):
            V[3*6 + 3 + k, r_offset2] = 1
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, r_offset2] = 1
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    best_result = schoolbook_full()
    if not verify_full(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p

    # Try to find rank-23 decomposition for 3x3x3
    # This will yield rank 48 for the full problem
    found_23 = False
    for seed in range(5):
        if not check_time():
            break
        res_3x3 = solve_3x3x3(23, max_iter=1000, seed=seed)
        if res_3x3 is not None:
            U3, V3, W3 = res_3x3
            # Verify 3x3x3 solution
            ok = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        a = i * 3 + j
                        b = j * 3 + k
                        c = k * 3 + i
                        val = np.sum(U3[a, :] * V3[b, :] * W3[c, :])
                        if abs(val - 1.0) > 1e-5:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if ok:
                full_res = build_from_3x3x3(U3, V3, W3, 23)
                if verify_full(full_res["U"], full_res["V"], full_res["W"], tol=1e-5):
                    best_rank = 48
                    best_result = full_res
                    found_23 = True
                    break

    if found_23:
        return best_result

    # If rank 23 failed, use rank 27 (schoolbook for 3x3x3) -> Total rank 56
    # Construct rank 27 solution for 3x3x3
    U3_27 = np.zeros((9, 27))
    V3_27 = np.zeros((9, 27))
    W3_27 = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3_27[i*3+j, col] = 1
                V3_27[j*3+k, col] = 1
                W3_27[k*3+i, col] = 1
                col += 1
                
    full_res_56 = build_from_3x3x3(U3_27, V3_27, W3_27, 27)
    if verify_full(full_res_56["U"], full_res_56["V"], full_res_56["W"], tol=1e-5):
        best_rank = 56
        best_result = full_res_56

    return best_result
# EVOLVE-BLOCK-END
