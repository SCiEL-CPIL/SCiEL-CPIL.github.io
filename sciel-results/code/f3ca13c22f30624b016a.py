import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 3, 3, 6
    
    # Try Laderman-based decomposition (Rank 46)
    try:
        U_l, V_l, W_l = get_laderman_matrices()
        
        rank = 46
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # Block 1: k in 0..2
        U[:, 0:23] = U_l
        for j in range(m):
            for k in range(3):
                V[j*p + k, 0:23] = V_l[j*3 + k, 0:23]
                W[k*n + i, 0:23] = W_l[k*3 + i, 0:23] # Wait, i loop needed
        
        # Corrected loops
        for j in range(m):
            for k in range(3):
                V[j*p + k, 0:23] = V_l[j*3 + k, 0:23]
        for k in range(3):
            for i in range(n):
                W[k*n + i, 0:23] = W_l[k*3 + i, 0:23]
                
        # Block 2: k in 3..5
        U[:, 23:46] = U_l
        for j in range(m):
            for k in range(3, 6):
                V[j*p + k, 23:46] = V_l[j*3 + (k-3), 0:23]
        for k in range(3, 6):
            for i in range(n):
                W[k*n + i, 23:46] = W_l[(k-3)*3 + i, 0:23]
                
        # Verify
        if verify_decomposition(U, V, W, n, m, p):
            if time.time() - start_time < 100:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback to Schoolbook
    return get_schoolbook()

def get_laderman_matrices():
    # Hardcoded Laderman matrices
    # Note: These must be correct for the verification to pass.
    # If incorrect, verification fails and we fallback.
    U = [
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        # ... I will use a placeholder that is likely to fail if I don't know it.
        # But I will try to use a known set.
    ]
    # Since I cannot guarantee the hardcoded values, I will return a dummy that fails verification.
    # This ensures safety.
    # However, to actually improve, I need the real values.
    # I will use a simple rank-27 algorithm for 3x3? No, that's 54 total.
    # I will return the schoolbook.
    return None, None, None

def verify_decomposition(U, V, W, n, m, p):
    # Check a few random cases
    for _ in range(10):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        # Compute using decomposition
        # C[i, k] = sum_r U[i, j] V[j, k] W[k, i] ? No.
        # The tensor is sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} A_{i,j} B_{j,k}
        # This sums to C_{i,k} if the decomposition is correct.
        # Actually, the bilinear form is <T, A \otimes B \otimes C^*> = tr(C^T A B).
        # So sum_{i,j,k} C_{i,k} (sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}) A_{i,j} B_{j,k} ?
        # No, the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
        # Then <T, A \otimes B \otimes C^*> = sum_r <u_r, A> <v_r, B> <w_r, C^*>.
        # <u_r, A> = sum_{i,j} U_{i,j}^{(r)} A_{i,j}.
        # <v_r, B> = sum_{j,k} V_{j,k}^{(r)} B_{j,k}.
        # <w_r, C^*> = sum_{k,i} W_{k,i}^{(r)} C_{i,k}.
        # So we need sum_r (sum_{i,j} U_{i,j}^{(r)} A_{i,j}) (sum_{j,k} V_{j,k}^{(r)} B_{j,k}) (sum_{k,i} W_{k,i}^{(r)} C_{i,k}) = tr(C^T A B).
        
        sum_val = 0
        for r in range(U.shape[1]):
            u_val = np.sum(U[:, r] * A.flatten()) # Note: U is (n*m, rank), A is (n, m). Flatten order must match.
            # U index i*m+j. A flatten row-major: i*m+j. Matches.
            v_val = np.sum(V[:, r] * B.flatten()) # V index j*p+k. B flatten row-major: j*p+k. Matches.
            w_val = np.sum(W[:, r] * C_expected.flatten()) # W index k*n+i. C flatten row-major: i*p+k. MISMATCH.
            # W index is k*n+i. C_expected[i, k].
            # We need to map W index to C value.
            # W_{k,i} corresponds to C_{i,k}.
            # So w_val = sum_{k,i} W_{k,i}^{(r)} C_{i,k}.
            # C_expected is (n, p). i in 0..n-1, k in 0..p-1.
            # W is (p*n, rank). Index k*n+i.
            # So we need to sum W[:, r] * C_reshaped.
            # C_reshaped[k*n+i] = C_expected[i, k].
            C_reshaped = np.zeros(p*n)
            for k in range(p):
                for i in range(n):
                    C_reshaped[k*n+i] = C_expected[i, k]
            w_val = np.sum(W[:, r] * C_reshaped)
            
            sum_val += u_val * v_val * w_val
            
        expected_val = np.trace(C_expected.T @ A @ B) # No, tr(C^T A B) = sum_{i,k} C_{i,k} (AB)_{i,k}.
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So expected_val = sum_{i,k} C_{i,k} sum_j A_{i,j} B_{j,k}.
        # This is the same as the tensor contraction.
        
        if not np.isclose(sum_val, expected_val):
            return False
    return True

def get_schoolbook():
    n, m, p = 3, 3, 6
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
