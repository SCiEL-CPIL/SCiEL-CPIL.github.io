"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, atol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    def compute_tensor(U_arr, V_arr, W_arr):
        r = U_arr.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U_arr[:, c]
            v = V_arr[:, c]
            w = W_arr[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        return T
    
    U_school = np.array(_schoolbook()["U"], dtype=float)
    V_school = np.array(_schoolbook()["V"], dtype=float)
    W_school = np.array(_schoolbook()["W"], dtype=float)
    
    T_target = compute_tensor(U_school, V_school, W_school)
    T_candidate = compute_tensor(U, V, W)
    return np.allclose(T_target, T_candidate, atol=atol)

def _is_halfint(x):
    """Check if x is an integer or half-integer."""
    x = float(x)
    return abs(x * 2 - round(x * 2)) < 1e-9

def _greedy_rank_reduction():
    """
    Start from schoolbook and greedily try to remove rank-1 terms.
    A term can be removed if it's linearly dependent on the remaining terms
    in the tensor decomposition sense.
    """
    n, m, p = DIMENSIONS
    U = np.array(_schoolbook()["U"], dtype=float)
    V = np.array(_schoolbook()["V"], dtype=float)
    W = np.array(_schoolbook()["W"], dtype=float)
    
    rank = U.shape[1]
    active = list(range(rank))
    
    # Precompute target tensor
    T_target = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    def current_tensor(active_set):
        T = np.zeros((n, m, p))
        for c in active_set:
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        return T
    
    # Try to remove each term
    removed = []
    for trial in range(50):  # Multiple passes
        if time.time() - _start_time > _deadline - 5:
            break
        found_removal = False
        for idx in range(len(active)):
            if time.time() - _start_time > _deadline - 5:
                break
            candidate_active = active[:idx] + active[idx+1:]
            T_test = current_tensor(candidate_active)
            if np.allclose(T_test, T_target, atol=1e-8):
                # Can remove this term
                active = candidate_active
                removed.append(active[idx] if idx < len(active) else active[-1])
                found_removal = True
                break
        if not found_removal:
            break
    
    # Build final matrices
    final_rank = len(active)
    U_final = [[0] * final_rank for _ in range(n * m)]
    V_final = [[0] * final_rank for _ in range(m * p)]
    W_final = [[0] * final_rank for _ in range(p * n)]
    
    for new_c, old_c in enumerate(active):
        for i in range(n * m):
            U_final[i][new_c] = U[i, old_c]
        for i in range(m * p):
            V_final[i][new_c] = V[i, old_c]
        for i in range(p * n):
            W_final[i][new_c] = W[i, old_c]
    
    return {"U": U_final, "V": V_final, "W": W_final}

def _random_perturbation_search():
    """
    Try random perturbations of the schoolbook decomposition to find lower rank.
    Uses the idea that some rank-1 terms might be combined.
    """
    n, m, p = DIMENSIONS
    U0 = np.array(_schoolbook()["U"], dtype=float)
    V0 = np.array(_schoolbook()["V"], dtype=float)
    W0 = np.array(_schoolbook()["W"], dtype=float)
    
    rank = U0.shape[1]
    T_target = np.zeros((n, m, p))
    for c in range(rank):
        u = U0[:, c]
        v = V0[:, c]
        w = W0[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    best_rank = rank
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    np.random.seed(42)
    
    # Try merging pairs of terms
    for attempt in range(100):
        if time.time() - _start_time > _deadline - 10:
            break
        
        # Pick two random terms to try to merge
        c1, c2 = np.random.choice(rank, 2, replace=False)
        
        # Try different merge strategies
        for alpha in [0.5, 0.25, 0.75, 1.0]:
            if time.time() - _start_time > _deadline - 10:
                break
            
            # Create a merged term
            U_new = U0.copy()
            V_new = V0.copy()
            W_new = W0.copy()
            
            # Merge c1 and c2 into c1
            U_new[:, c1] = alpha * U0[:, c1] + (1-alpha) * U0[:, c2]
            V_new[:, c1] = alpha * V0[:, c1] + (1-alpha) * V0[:, c2]
            W_new[:, c1] = alpha * W0[:, c1] + (1-alpha) * W0[:, c2]
            U_new[:, c2] = 0
            V_new[:, c2] = 0
            W_new[:, c2] = 0
            
            # Check if still valid
            T_test = np.zeros((n, m, p))
            for c in range(rank):
                if np.all(U_new[:, c] == 0) and np.all(V_new[:, c] == 0) and np.all(W_new[:, c] == 0):
                    continue
                u = U_new[:, c]
                v = V_new[:, c]
                w = W_new[:, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_test[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            if np.allclose(T_test, T_target, atol=1e-8):
                # Count non-zero terms
                active = [c for c in range(rank) if not (np.all(U_new[:, c] == 0) and 
                                                        np.all(V_new[:, c] == 0) and 
                                                        np.all(W_new[:, c] == 0))]
                if len(active) < best_rank:
                    best_rank = len(active)
                    best_U = U_new[:, active].copy()
                    best_V = V_new[:, active].copy()
                    best_W = W_new[:, active].copy()
    
    # Convert to list format
    final_rank = best_U.shape[1]
    U_final = [[0] * final_rank for _ in range(n * m)]
    V_final = [[0] * final_rank for _ in range(m * p)]
    W_final = [[0] * final_rank for _ in range(p * n)]
    
    for c in range(final_rank):
        for i in range(n * m):
            U_final[i][c] = best_U[i, c]
        for i in range(m * p):
            V_final[i][c] = best_V[i, c]
        for i in range(p * n):
            W_final[i][c] = best_W[i, c]
    
    return {"U": U_final, "V": V_final, "W": W_final}

def construct():
    global _start_time, _deadline
    _start_time = time.time()
    _deadline = _start_time + 140  # 140s deadline, 10s margin
    
    # First, get the schoolbook decomposition
    schoolbook = _schoolbook()
    schoolbook_rank = len(schoolbook["U"][0]) if schoolbook["U"] else 0
    
    best_result = schoolbook
    best_rank = schoolbook_rank
    
    # Try greedy rank reduction
    if time.time() - _start_time < _deadline - 20:
        greedy = _greedy_rank_reduction()
        greedy_rank = len(greedy["U"][0]) if greedy["U"] else 0
        if _verify(greedy["U"], greedy["V"], greedy["W"]):
            if greedy_rank < best_rank:
                best_result = greedy
                best_rank = greedy_rank
    
    # Try random perturbation search
    if time.time() - _start_time < _deadline - 10:
        random_search = _random_perturbation_search()
        random_rank = len(random_search["U"][0]) if random_search["U"] else 0
        if _verify(random_search["U"], random_search["V"], random_search["W"]):
            if random_rank < best_rank:
                best_result = random_search
                best_rank = random_rank
    
    return best_result
# EVOLVE-BLOCK-END
