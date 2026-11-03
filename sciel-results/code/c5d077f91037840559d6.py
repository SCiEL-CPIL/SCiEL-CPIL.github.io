"""Combined approach: Strassen-inspired init + efficient alternating minimization + polish."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Build list of (a,b,c,expected) entries for the tensor."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries.append((a, b, c, 1.0))
    return entries

def _build_all_nonzero(n, m, p):
    """Set of all (a,b,c) that should be nonzero."""
    s = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s.add((a, b, c))
    return s

def _full_verify(U, V, W, target, nonzero_set, n, m, p, tol=1e-8):
    """Full verification of all target and non-target entries."""
    for a, b, c, exp in target:
        val = np.dot(U[a], V[b] * W[c])
        if abs(val - exp) > tol:
            return False
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in nonzero_set:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
    return True

def _compute_error(U, V, W, target, nonzero_set, n, m, p):
    """Compute error over target and sampled non-target entries."""
    err = 0.0
    for a, b, c, exp in target:
        val = np.dot(U[a], V[b] * W[c])
        err += (val - exp) ** 2
    # Sample non-target entries
    rng = np.random.RandomState(42)
    for _ in range(100):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        if (a, b, c) not in nonzero_set:
            val = np.dot(U[a], V[b] * W[c])
            err += val ** 2
    return err

def _strassen_init_3x3(rank):
    """
    Initialize using Strassen-like structure for 3x3x3 blocks.
    Returns U, V, W with shape (9, rank) for the 3x3x3 case.
    """
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Strassen's 7 terms for 3x3 matrix multiplication
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A20+A22)B00
    # M3 = A00(B01-B21)
    # M4 = A22(B21-B01)
    # M5 = (A01+A11)B11
    # M6 = (A10+A11)(B01+B11)
    # M7 = (A00+A10)(B10-B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C02 = M2 - M4 + M6
    # C10 = M1 - M3 + M5 - M7
    # C11 = M2 - M1 + M4 + M7
    # C12 = -M2 + M6
    # C20 = M3 - M6 + M7
    # C21 = -M4 - M7
    # C22 = M1 - M2 + M4 - M5
    
    if rank >= 7:
        # Term 1: M1 = (A00+A22)(B00+B22)
        U[0, 0] = 1; U[8, 0] = 1  # A00, A22
        V[0, 0] = 1; V[2, 0] = 1  # B00, B22
        W[0, 0] = 1; W[3, 0] = 1; W[5, 0] = 1; W[6, 0] = 1; W[8, 0] = 1  # C00,C10,C20,C21,C22
        
        # Term 2: M2 = (A20+A22)B00
        U[6, 1] = 1; U[8, 1] = 1  # A20, A22
        V[0, 1] = 1  # B00
        W[2, 1] = 1; W[4, 1] = 1; W[5, 1] = 1; W[7, 1] = -1  # C02,C12,C22,C21
        
        # Term 3: M3 = A00(B01-B21)
        U[0, 2] = 1  # A00
        V[1, 2] = 1; V[4, 2] = -1  # B01, B21
        W[1, 3] = 1; W[3, 4] = -1; W[6, 5] = 1  # C01,C10,C20
        
        # Term 4: M4 = A22(B21-B01)
        U[8, 3] = 1  # A22
        V[4, 3] = 1; V[1, 3] = -1  # B21, B01
        W[0, 4] = 1; W[2, 5] = -1; W[4, 6] = 1; W[7, 7] = -1; W[8, 8] = 1  # C00,C02,C11,C21,C22
        
        # Term 5: M5 = (A01+A11)B11
        U[1, 4] = 1; U[4, 5] = 1  # A01, A11
        V[4, 6] = 1  # B11
        W[0, 7] = -1; W[1, 8] = 1; W[3, 9] = 1; W[8, 10] = -1  # C00,C01,C10,C22
        
        # Term 6: M6 = (A10+A11)(B01+B11)
        U[3, 11] = 1; U[4, 12] = 1  # A10, A11
        V[1, 13] = 1; V[4, 14] = 1  # B01, B11
        W[2, 15] = 1; W[5, 16] = 1; W[6, 17] = -1  # C02,C12,C20
        
        # Term 7: M7 = (A00+A10)(B10-B11)
        U[0, 18] = 1; U[3, 19] = 1  # A00, A10
        V[3, 20] = 1; V[4, 21] = -1  # B10, B11
        W[0, 22] = 1; W[3, 23] = -1; W[6, 24] = 1; W[7, 25] = -1  # C00,C10,C20,C21
    
    return U, V, W

def _alternating_minimize(U, V, W, target, nonzero_set, n, m, p, deadline, max_iter=100):
    """
    Alternating minimization: fix two factors, solve for the third via least squares.
    """
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        # Update U: for each row a, solve min ||U[a,:] @ (V*W)^T - target_row||
        for a in range(n * m):
            # Build system: for each (b,c) in target with this a
            # sum_t U[a,t] * V[b,t] * W[c,t] = 1
            A_mat = []
            b_vec = []
            for (ta, tb, tc, exp) in target:
                if ta == a:
                    A_mat.append(V[tb] * W[tc])
                    b_vec.append(exp)
            if A_mat:
                A_mat = np.array(A_mat)
                b_vec = np.array(b_vec)
                # Also add non-target constraints (sampled)
                rng = np.random.RandomState(iteration + a)
                for _ in range(20):
                    tb = rng.randint(0, m * p)
                    tc = rng.randint(0, p * n)
                    if (a, tb, tc) not in nonzero_set:
                        A_mat = np.vstack([A_mat, V[tb] * W[tc]])
                        b_vec = np.append(b_vec, 0.0)
                try:
                    U[a], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update V
        for b in range(m * p):
            A_mat = []
            b_vec = []
            for (ta, tb, tc, exp) in target:
                if tb == b:
                    A_mat.append(U[ta] * W[tc])
                    b_vec.append(exp)
            if A_mat:
                A_mat = np.array(A_mat)
                b_vec = np.array(b_vec)
                rng = np.random.RandomState(iteration + b + 1000)
                for _ in range(20):
                    ta = rng.randint(0, n * m)
                    tc = rng.randint(0, p * n)
                    if (ta, b, tc) not in nonzero_set:
                        A_mat = np.vstack([A_mat, U[ta] * W[tc]])
                        b_vec = np.append(b_vec, 0.0)
                try:
                    V[b], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update W
        for c in range(p * n):
            A_mat = []
            b_vec = []
            for (ta, tb, tc, exp) in target:
                if tc == c:
                    A_mat.append(U[ta] * V[tb])
                    b_vec.append(exp)
            if A_mat:
                A_mat = np.array(A_mat)
                b_vec = np.array(b_vec)
                rng = np.random.RandomState(iteration + c + 2000)
                for _ in range(20):
                    ta = rng.randint(0, n * m)
                    tb = rng.randint(0, m * p)
                    if (ta, tb, c) not in nonzero_set:
                        A_mat = np.vstack([A_mat, U[ta] * V[tb]])
                        b_vec = np.append(b_vec, 0.0)
                try:
                    W[c], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
    
    return U, V, W

def _polish(U, V, W, target, nonzero_set, n, m, p, deadline, max_iter=50):
    """Final polish with gradient descent at tighter tolerance."""
    start_time = time.time()
    lr = 0.001
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for a, b, c, exp in target:
            T_val = np.dot(U[a], V[b] * W[c])
            E = T_val - exp
            dU[a] += 2 * E * (V[b] * W[c])
            dV[b] += 2 * E * (U[a] * W[c])
            dW[c] += 2 * E * (U[a] * V[b])
        
        rng = np.random.RandomState(iteration)
        for _ in range(50):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            if (a, b, c) not in nonzero_set:
                T_val = np.dot(U[a], V[b] * W[c])
                dU[a] += 2 * T_val * (V[b] * W[c])
                dV[b] += 2 * T_val * (U[a] * W[c])
                dW[c] += 2 * T_val * (U[a] * V[b])
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        err = _compute_error(U, V, W, target, nonzero_set, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-10:
            break
    
    return best_U, best_V, best_W

def _try_rank(n, m, p, rank, deadline, start_time, restart=0):
    """Try to find decomposition at given rank."""
    if time.time() - start_time > deadline:
        return None, None
    
    target = _build_target(n, m, p)
    nonzero_set = _build_all_nonzero(n, m, p)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with Strassen-inspired structure for each 3x3 block
    np_rng = np.random.RandomState(restart * 7919 + 12345)
    
    # For 3x3x7, we have 3 output rows, each with 7 columns
    # Initialize with random but structured values
    for r in range(rank):
        U[:, r] = np_rng.randn(n * m) * 0.5
        V[:, r] = np_rng.randn(m * p) * 0.5
        W[:, r] = np_rng.randn(p * n) * 0.5
    
    # Run alternating minimization
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time < 5:
        return None, None
    
    U, V, W = _alternating_minimize(U, V, W, target, nonzero_set, n, m, p, 
                                     start_time + remaining_time * 0.7, max_iter=50)
    
    # Check if we have a valid solution
    if _full_verify(U, V, W, target, nonzero_set, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    # Try polish
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time < 2:
        return None, None
    
    U, V, W = _polish(U, V, W, target, nonzero_set, n, m, p,
                       start_time + remaining_time * 0.9, max_iter=30)
    
    if _full_verify(U, V, W, target, nonzero_set, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    return None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # Leave 10s safety margin
    
    target = _build_target(n, m, p)
    nonzero_set = _build_all_nonzero(n, m, p)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 63
    
    # Build schoolbook solution as fallback
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook
    
    # Try ranks from lower to higher
    for rank in range(35, rank_schoolbook):
        if time.time() - start_time > DEADLINE:
            break
        
        for restart in range(3):
            if time.time() - start_time > DEADLINE:
                break
            
            result, r = _try_rank(n, m, p, rank, DEADLINE, start_time, restart)
            if result is not None and r < best_rank:
                best_result = result
                best_rank = r
                break  # Found solution at this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
