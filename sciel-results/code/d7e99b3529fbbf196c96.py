"""Matrix multiplication 4,4,8 rank-minimal search with greedy constructive decomposition."""
import random
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _get_tensor_indices(n, m, p):
    """Get all (a, b, c) index triples for the multiplication tensor."""
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _validate_fast(U, V, W, n, m, p):
    """Fast validation: check all tensor entries equal 1."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    indices = _get_tensor_indices(n, m, p)
    for a, b, c in indices:
        s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
        if abs(s - 1.0) > 1e-9:
            return False
    return True

def _compute_residuals(U, V, W, n, m, p):
    """Compute the current residual for each tensor entry (should be 1 - actual)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    indices = _get_tensor_indices(n, m, p)
    residuals = {}
    for a, b, c in indices:
        s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
        residuals[(a, b, c)] = 1.0 - s
    return residuals

def _greedy_construct(n, m, p, deadline, max_rank=40):
    """
    Greedy constructive algorithm: build rank-1 terms one at a time.
    Each new term is chosen to maximize the number of unsatisfied equations it helps.
    """
    nm, mp, pn = n * m, m * p, p * n
    indices = _get_tensor_indices(n, m, p)
    
    # Start with empty matrices
    U = np.zeros((nm, 0))
    V = np.zeros((mp, 0))
    W = np.zeros((pn, 0))
    
    # Track which equations are satisfied
    # residual[a,b,c] = 1 - sum_r U[a,r]*V[b,r]*W[c,r]
    residual = {(a, b, c): 1.0 for a, b, c in indices}
    
    unsatisfied = set(indices)
    
    for rank_step in range(max_rank):
        if time.time() > deadline - 20:
            break
        
        if not unsatisfied:
            break
        
        # Try to find a good rank-1 term (u, v, w)
        # Strategy: pick a random unsatisfied equation, then try to satisfy it
        # while maximizing additional equations satisfied
        
        best_gain = 0
        best_term = None
        
        # Sample some unsatisfied equations to anchor on
        sample_size = min(50, len(unsatisfied))
        sample = random.sample(list(unsatisfied), sample_size)
        
        for (a0, b0, c0) in sample:
            if time.time() > deadline - 20:
                break
            
            # For this anchor equation, we need U[a0]*V[b0]*W[c0] = 1 (approximately)
            # Try small integer/half-integer values
            
            # Strategy: set U[a0]=1, V[b0]=1, W[c0]=1 as base
            # Then extend to satisfy more equations
            
            # Randomized approach: generate random sparse vectors
            for attempt in range(20):
                if time.time() > deadline - 20:
                    break
                
                # Generate random sparse vectors with entries in {-1, 0, 1}
                density_u = random.choice([0.2, 0.3, 0.4, 0.5])
                density_v = random.choice([0.2, 0.3, 0.4, 0.5])
                density_w = random.choice([0.2, 0.3, 0.4, 0.5])
                
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                
                # Ensure the anchor equation is satisfied
                u[a0] = 1.0
                v[b0] = 1.0
                w[c0] = 1.0
                
                # Add random non-zero entries
                for idx in range(nm):
                    if idx != a0 and random.random() < density_u:
                        u[idx] = random.choice([-1.0, 1.0])
                for idx in range(mp):
                    if idx != b0 and random.random() < density_v:
                        v[idx] = random.choice([-1.0, 1.0])
                for idx in range(pn):
                    if idx != c0 and random.random() < density_w:
                        w[idx] = random.choice([-1.0, 1.0])
                
                # Compute how many unsatisfied equations this term helps
                gain = 0
                for (a, b, c) in unsatisfied:
                    contrib = u[a] * v[b] * w[c]
                    if abs(contrib) > 1e-9:
                        # This term contributes to this equation
                        # Check if it moves toward satisfaction
                        if contrib > 0:
                            gain += 1
                        else:
                            # Negative contribution - might still help if residual is negative
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_term = (u, v, w)
        
        if best_term is None:
            # Fallback: just use the first unsatisfied equation
            a0, b0, c0 = next(iter(unsatisfied))
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            u[a0] = 1.0
            v[b0] = 1.0
            w[c0] = 1.0
            best_term = (u, v, w)
        
        u, v, w = best_term
        
        # Add this term to our matrices
        U = np.column_stack([U, u]) if U.shape[1] > 0 else u.reshape(-1, 1)
        V = np.column_stack([V, v]) if V.shape[1] > 0 else v.reshape(-1, 1)
        W = np.column_stack([W, w]) if W.shape[1] > 0 else w.reshape(-1, 1)
        
        # Update residuals
        for (a, b, c) in unsatisfied:
            contrib = u[a] * v[b] * w[c]
            residual[(a, b, c)] -= contrib
            if abs(residual[(a, b, c)]) < 1e-9:
                unsatisfied.discard((a, b, c))
        
        # Check if all satisfied
        if not unsatisfied:
            break
    
    if not unsatisfied:
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _greedy_with_backtracking(n, m, p, deadline, max_rank=35):
    """
    Greedy construction with limited backtracking and local repair.
    """
    nm, mp, pn = n * m, m * p, p * n
    indices = _get_tensor_indices(n, m, p)
    
    U_list = []
    V_list = []
    W_list = []
    
    residual = {(a, b, c): 1.0 for a, b, c in indices}
    unsatisfied = set(indices)
    
    for rank_step in range(max_rank):
        if time.time() > deadline - 20:
            break
        
        if not unsatisfied:
            break
        
        # Find the equation with the largest residual magnitude
        # to prioritize
        target_eq = max(unsatisfied, key=lambda eq: abs(residual[eq]))
        a0, b0, c0 = target_eq
        target_sign = 1.0 if residual[(a0, b0, c0)] > 0 else -1.0
        
        best_term = None
        best_score = -1
        
        # Try multiple random candidates
        for attempt in range(30):
            if time.time() > deadline - 20:
                break
            
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            
            # Base: satisfy the target equation
            u[a0] = target_sign
            v[b0] = 1.0
            w[c0] = 1.0
            
            # Random extensions
            for idx in range(nm):
                if idx != a0 and random.random() < 0.3:
                    u[idx] = random.choice([-1.0, 1.0])
            for idx in range(mp):
                if idx != b0 and random.random() < 0.3:
                    v[idx] = random.choice([-1.0, 1.0])
            for idx in range(pn):
                if idx != c0 and random.random() < 0.3:
                    w[idx] = random.choice([-1.0, 1.0])
            
            # Score: how many unsatisfied equations does this help?
            score = 0
            for (a, b, c) in unsatisfied:
                contrib = u[a] * v[b] * w[c]
                if contrib * residual[(a, b, c)] > 0:
                    score += 1
                elif contrib * residual[(a, b, c)] < 0:
                    score -= 1
            
            if score > best_score:
                best_score = score
                best_term = (u, v, w)
        
        if best_term is None:
            break
        
        u, v, w = best_term
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Update residuals
        for (a, b, c) in list(unsatisfied):
            contrib = u[a] * v[b] * w[c]
            residual[(a, b, c)] -= contrib
            if abs(residual[(a, b, c)]) < 1e-9:
                unsatisfied.discard((a, b, c))
    
    if not unsatisfied:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition (rank = n*m*p)."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin (evaluator kills at 150s)

    best_result = None
    best_rank = n * m * p  # schoolbook rank as initial best

    # Strategy 1: Greedy constructive with backtracking
    try:
        result = _greedy_with_backtracking(n, m, p, deadline, max_rank=35)
        if result:
            U, V, W = result
            if _validate_fast(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Strategy 2: Greedy constructive (simpler version)
    try:
        result = _greedy_construct(n, m, p, deadline, max_rank=40)
        if result:
            U, V, W = result
            if _validate_fast(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Strategy 3: Try to find even lower rank with more aggressive search
    if best_result is not None and time.time() < deadline - 30:
        try:
            # Try to reduce the found rank by removing redundant columns
            U = best_result['U']
            V = best_result['V']
            W = best_result['W']
            current_rank = len(U[0])
            
            # Try removing each column
            keep = list(range(current_rank))
            for col in list(keep):
                if time.time() > deadline - 25:
                    break
                trial_keep = [c for c in keep if c != col]
                U_trial = [[row[c] for c in trial_keep] for row in U]
                V_trial = [[row[c] for c in trial_keep] for row in V]
                W_trial = [[row[c] for c in trial_keep] for row in W]
                
                if _validate_fast(U_trial, V_trial, W_trial, n, m, p):
                    keep = trial_keep
                    best_rank -= 1
                    best_result = {"U": U_trial, "V": V_trial, "W": W_trial}
        except Exception:
            pass

    if best_result is not None:
        return best_result

    # Fallback to schoolbook
    return _schoolbook(n, m, p)
