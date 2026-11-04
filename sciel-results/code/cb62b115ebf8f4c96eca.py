import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    """Build the schoolbook (naive) matrix multiplication decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    # Check all "one" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    rng = random.Random(42)
    for _ in range(2000):
        a = rng.randint(0, n * m - 1)
        b = rng.randint(0, m * p - 1)
        g = rng.randint(0, p * n - 1)
        i = a // m
        j = a % m
        jp = b // p
        k = b % p
        kn = g // n
        ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.sum(U[a, :] * V[b, :] * W[g, :])
            if abs(val) > tol:
                return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _tensor_unfolds(U, V, W, n, m, p):
    """Compute the three mode-unfoldings of the tensor T = sum_r U[:,r] V[:,r] W[:,r]."""
    r = U.shape[1]
    # Mode-1 unfolding: (n*m) x (m*p * p*n)
    # T[a, b, g] = sum_r U[a,r] V[b,r] W[g,r]
    # U1[a, (b,g)] = sum_r U[a,r] V[b,r] W[g,r]
    # This is: U1 = U * (V ⊙ W) where ⊙ is Khatri-Rao product
    # V_kr[b,g,r] = V[b,r]*W[g,r], shape (m*p*p*n, r)
    # U1[a, b*g_index] = sum_r U[a,r] * V_kr[b*g_index, r]
    
    # More efficiently: for each a, U1[a, :] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
    # But that's expensive. Let's use the structure.
    
    # Actually, for ALS we need:
    # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r] flattened) = T[a, :, :] flattened
    # For fixed b: sum_r V[b,r] * (U[:,r] * W[:,r] flattened) = T[:, b, :] flattened
    # For fixed g: sum_r W[g,r] * (U[:,r] * V[:,r] flattened) = T[:, :, g] flattened
    
    return None  # Placeholder

def _als_smooth(n, m, p, target_rank, max_time=20, seed=42):
    """
    ALS with smooth surrogate for the tensor decomposition.
    
    Uses log-sum-exp as a smooth max for the constraint violations.
    Adds epigraph variables for the zero constraints.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    rng = np.random.RandomState(seed)
    
    # Pre-compute index mappings
    one_entries = []  # (a, b, g) where T[a,b,g] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                one_entries.append((a, b, g))
    
    # Zero entries: all (a,b,g) not in one_entries
    # We'll sample these
    
    # Initialization: use truncated SVD of the tensor
    # Build the full tensor T (sparse)
    # T has shape (n*m, m*p, p*n) with 1 at one_entries, 0 elsewhere
    
    # Mode-1 unfolding: (n*m) x (m*p * p*n)
    # U1[a, (b,g)] = T[a,b,g]
    # This is a (n*m) x (m*p*p*n) matrix, mostly zeros
    # We can compute its SVD via the factor structure
    
    # Better initialization: random but structured
    best_result = None
    best_score = float('inf')
    
    for init in range(5):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed + init)
        
        # Initialize with random values, scaled
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Precompute for fast ALS
        # For updating U[a]: need V_kr = V ⊙ W (Khatri-Rao), shape (m*p*p*n, target_rank)
        # U1_target[a, :] = T[a, :, :] flattened
        # U[a, :] = (V_kr^T V_kr)^{-1} V_kr^T U1_target[a, :]
        
        # But V_kr is huge. Instead, use the structure:
        # V_kr[b, g, r] = V[b, r] * W[g, r]
        # So (V_kr^T V_kr)[r1, r2] = sum_{b,g} V[b,r1]*W[g,r1]*V[b,r2]*W[g,r2]
        # = (V^T V)[r1,r2] * (W^T W)[r1,r2]  (element-wise product)
        
        # And (V_kr^T U1_target[a, :])[r] = sum_{b,g} V[b,r]*W[g,r]*T[a,b,g]
        # = sum_{b,g where T[a,b,g]=1} V[b,r]*W[g,r]
        # For fixed a, T[a,b,g]=1 only for specific (b,g) pairs
        
        # Precompute for each a: the list of (b,g) where T[a,b,g]=1
        a_to_bg = {}
        for a in range(n * m):
            i = a // m
            j = a % m
            bg_list = []
            for k in range(p):
                b = j * p + k
                g = k * n + i
                bg_list.append((b, g))
            a_to_bg[a] = bg_list
        
        b_to_ag = {}
        for b in range(m * p):
            j = b // p
            k = b % p
            ag_list = []
            for i in range(n):
                a = i * m + j
                g = k * n + i
                ag_list.append((a, g))
            b_to_ag[b] = ag_list
        
        g_to_ab = {}
        for g in range(p * n):
            k = g // n
            i = g % n
            ab_list = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                ab_list.append((a, b))
            g_to_ab[g] = ab_list
        
        for iteration in range(50):
            if time.time() > deadline - 2:
                break
            
            # Update U
            # G_V = V^T V, G_W = W^T W
            G_V = V.T @ V
            G_W = W.T @ W
            A_mat = G_V * G_W  # element-wise
            A_mat += 1e-10 * np.eye(target_rank)
            
            for a in range(n * m):
                b_vec = np.zeros(target_rank)
                for b, g in a_to_bg[a]:
                    b_vec += V[b, :] * W[g, :]
                try:
                    U[a, :] = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update V
            G_U = U.T @ U
            G_W = W.T @ W
            A_mat = G_U * G_W
            A_mat += 1e-10 * np.eye(target_rank)
            
            for b in range(m * p):
                b_vec = np.zeros(target_rank)
                for a, g in b_to_ag[b]:
                    b_vec += U[a, :] * W[g, :]
                try:
                    V[b, :] = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update W
            G_U = U.T @ U
            G_V = V.T @ V
            A_mat = G_U * G_V
            A_mat += 1e-10 * np.eye(target_rank)
            
            for g in range(p * n):
                b_vec = np.zeros(target_rank)
                for a, b in g_to_ab[g]:
                    b_vec += U[a, :] * V[b, :]
                try:
                    W[g, :] = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    W[g, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Compute smooth error: log-sum-exp of |violations|
            # One-entry violations: |T[a,b,g] - sum_r U[a,r]V[b,r]W[g,r]|
            # Zero-entry violations: |sum_r U[a,r]V[b,r]W[g,r]|
            
            # Sample for efficiency
            one_violations = []
            for a, b, g in one_entries:
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                one_violations.append(abs(val - 1.0))
            
            # Sample zero entries
            zero_violations = []
            sample_size = min(500, (n*m)*(m*p)*(p*n) - len(one_entries))
            for _ in range(sample_size):
                a = rng.randint(0, n * m - 1)
                b = rng.randint(0, m * p - 1)
                g = rng.randint(0, p * n - 1)
                i = a // m
                j = a % m
                jp = b // p
                k = b % p
                kn = g // n
                ip = g % n
                if i != ip or j != jp or k != kn:
                    val = np.sum(U[a, :] * V[b, :] * W[g, :])
                    zero_violations.append(abs(val))
            
            all_violations = np.array(one_violations + zero_violations)
            if len(all_violations) == 0:
                continue
            
            # Log-sum-exp for smooth max
            max_v = np.max(all_violations)
            if max_v < 1e-12:
                score = 0.0
            else:
                score = max_v + np.log(np.sum(np.exp(all_violations - max_v)))
            
            # Also track raw max
            raw_max = np.max(all_violations)
            
            if raw_max < best_score:
                best_score = raw_max
                best_result = (U.copy(), V.copy(), W.copy())
            
            if raw_max < 1e-10:
                break
        
        if best_result is not None:
            U_r, V_r, W_r = best_result
            U_rounded = _round_to_half_integer(U_r)
            V_rounded = _round_to_half_integer(V_r)
            W_rounded = _round_to_half_integer(W_r)
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
    
    return None

def _polish_pass(U, V, W, n, m, p, max_time=5):
    """
    Final polish: given a near-exact decomposition, try to make it exact
    by adjusting entries to half-integers.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Try rounding to nearest half-integer and verify
    U_rounded = _round_to_half_integer(U)
    V_rounded = _round_to_half_integer(V)
    W_rounded = _round_to_half_integer(W)
    
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # Try rounding to nearest integer
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    
    if _verify_decomposition(U_int, V_int, W_int, n, m, p):
        return U_int, V_int, W_int
    
    # Try small perturbations around rounded values
    for scale in [0.25, 0.1, 0.05]:
        if time.time() > deadline:
            break
        
        U_pert = U_rounded + scale * np.random.randn(*U_rounded.shape) * 0.1
        V_pert = V_rounded + scale * np.random.randn(*V_rounded.shape) * 0.1
        W_pert = W_rounded + scale * np.random.randn(*W_rounded.shape) * 0.1
        
        U_pert = _round_to_half_integer(U_pert)
        V_pert = _round_to_half_integer(V_pert)
        W_pert = _round_to_half_integer(W_pert)
        
        if _verify_decomposition(U_pert, V_pert, W_pert, n, m, p):
            return U_pert, V_pert, W_pert
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try ALS for various ranks, starting from lower (harder) to higher (easier)
    for target_rank in [80, 90, 100, 110]:
        if time.time() > deadline - 20:
            break
        
        remaining = int(deadline - time.time() - 10)
        if remaining < 10:
            break
        
        result = _als_smooth(n, m, p, target_rank, max_time=remaining, seed=target_rank)
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
                break  # Found a valid decomposition, stop
    
    # If we found something, try to polish
    if best_rank < n * m * p and time.time() < deadline - 5:
        polished = _polish_pass(best_U, best_V, best_W, n, m, p, max_time=5)
        if polished is not None:
            best_U, best_V, best_W = polished
            best_rank = best_U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
