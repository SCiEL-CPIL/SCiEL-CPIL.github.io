import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def _validate(U, V, W, n, m, p):
    """Validate exact decomposition with half-integer check."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    if r == 0:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    # Check half-integer entries
    for arr in (U, V, W):
        for row in arr:
            for x in row:
                if abs(2*x - round(2*x)) > 1e-9:
                    return False
    # Check tensor identity
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    U3 = Ua.reshape(n, m, r)
    V3 = Va.reshape(m, p, r)
    W3 = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _snap(arr):
    """Snap to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2, rank 7.
    Returns U (4,7), V (4,7), W (4,7) such that
    sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}."""
    # Standard Strassen for 2x2x2
    # U[i,j] corresponds to input A[i,j], V[j,k] to B[j,k], W[k,i] to C[k,i]
    # Indices: i,j,k in {0,1}
    # U rows: (0,0),(0,1),(1,0),(1,1)
    # V rows: (0,0),(0,1),(1,0),(1,1)
    # W rows: (0,0),(0,1),(1,0),(1,1)
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    # C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    # M1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    # M2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    # M3: A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1
    # M4: A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    # M5: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    # M6: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    # M7: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    # W: C00=M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3-M6
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=-1  # C00
    W[1,2]=1; W[1,4]=1  # C01
    W[2,1]=1; W[2,3]=1  # C10
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=-1  # C11
    return U, V, W

def _laderman_3x3x3():
    """Laderman decomposition for 3x3x3, rank 23.
    Returns U (9,23), V (9,23), W (9,23)."""
    # Laderman's rank-23 decomposition for 3x3x3
    # This is a known result. Let me construct it.
    # The 23 terms are defined by specific patterns.
    # I'll use a known construction.
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Laderman's decomposition uses 23 terms.
    # The construction involves specific index patterns.
    # Let me use a verified construction.
    # 
    # Actually, let me use a simpler approach: build from Strassen blocks + corrections
    # Or use the known explicit form.
    
    # For now, let me use a construction based on the fact that
    # 3x3x3 can be decomposed using 23 terms.
    # I'll use a known explicit set of 23 tensors.
    
    # The Laderman decomposition (1976) for 3x3x3:
    # Terms 1-8: Strassen-like on the top-left 2x2x2
    # Terms 9-16: Strassen-like on other 2x2x2 blocks
    # Terms 17-23: correction terms
    
    # Actually, let me just use a known valid set.
    # The 23 rank-1 tensors (u_r, v_r, w_r) for 3x3x3:
    
    # I'll use a construction that I can verify.
    # For the purpose of this program, let me construct it carefully.
    
    # A known Laderman decomposition:
    # Let me define the 23 terms explicitly.
    
    # Term definitions (u, v, w) where each is a 3x3 matrix (flattened to 9)
    # This is complex. Let me use an alternative: 
    # Start from schoolbook rank 27 and try to reduce.
    
    # Actually, for the 3x5x6 problem, let me use a different strategy:
    # Decompose 3x5x6 as a combination of smaller blocks.
    
    # Let me return a placeholder that will be filled by the search.
    # For now, return None to indicate we need to search.
    return None

def _gradient_optimize(U, V, W, n, m, p, max_iter=500, lr=0.01):
    """Gradient-based optimization to minimize error."""
    Ua = np.array(U, dtype=np.float64).copy()
    Va = np.array(V, dtype=np.float64).copy()
    Wa = np.array(W, dtype=np.float64).copy()
    r = Ua.shape[1]
    
    U3 = Ua.reshape(n, m, r)
    V3 = Va.reshape(m, p, r)
    W3 = Wa.reshape(p, n, r)
    
    best_err = np.inf
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        loss = np.sum(err**2)
        
        if loss < best_err:
            best_err = loss
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if best_err < 1e-10:
            break
        
        # Gradients
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        
        # Adaptive learning rate
        lr_eff = lr / (1.0 + it * 0.01)
        
        U3 -= lr_eff * grad_U
        V3 -= lr_eff * grad_V
        W3 -= lr_eff * grad_W
    
    Ua = best_U.reshape(n*m, r)
    Va = best_V.reshape(m*p, r)
    Wa = best_W.reshape(p*n, r)
    return Ua, Va, Wa, best_err

def _alternating_minimize(U, V, W, n, m, p, max_iter=300):
    """Alternating minimization: fix two, solve for the third."""
    Ua = np.array(U, dtype=np.float64).copy()
    Va = np.array(V, dtype=np.float64).copy()
    Wa = np.array(W, dtype=np.float64).copy()
    r = Ua.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = Va[j*p+k] * Wa[k*n+i]
                try:
                    Ua[i*m+j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = Ua[i*m+j] * Wa[k*n+i]
                try:
                    Va[j*p+k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = Ua[i*m+j] * Va[j*p+k]
                try:
                    Wa[k*n+i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = Ua.reshape(n, m, r)
        V3 = Va.reshape(m, p, r)
        W3 = Wa.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.sum((T - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        
        if best_err < 1e-12:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, rank, seed=0, max_iter=300):
    """Try to find a valid decomposition of given rank."""
    rng = random.Random(seed)
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Random initialization
    U = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n*m)])
    V = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m*p)])
    W = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p*n)])
    initializations.append((U, V, W))
    
    # 2. Sparse random initialization (many zeros)
    for trial in range(3):
        U = np.array([[rng.gauss(0, 1) if rng.random() < 0.5 else 0 for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[rng.gauss(0, 1) if rng.random() < 0.5 else 0 for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[rng.gauss(0, 1) if rng.random() < 0.5 else 0 for _ in range(rank)] for _ in range(p*n)])
        initializations.append((U, V, W))
    
    # 3. Try to use Strassen blocks where applicable
    # For 3x5x6, we can try to build from 2x2x2 Strassen blocks
    # This is complex, so let's try a structured initialization
    U_struct = np.zeros((n*m, rank))
    V_struct = np.zeros((m*p, rank))
    W_struct = np.zeros((p*n, rank))
    
    # Fill with Strassen-like patterns for 2x2x2 sub-blocks
    # Use the first 7 columns for a 2x2x2 Strassen block on indices (0,1) x (0,1) x (0,1)
    if rank >= 7:
        S_U, S_V, S_W = _strassen_2x2x2()
        # Map: i in {0,1}, j in {0,1}, k in {0,1}
        for i in range(2):
            for j in range(2):
                U_struct[i*m+j, 0:7] = S_U[i*2+j, 0:7]
        for j in range(2):
            for k in range(2):
                V_struct[j*p+k, 0:7] = S_V[j*2+k, 0:7]
        for k in range(2):
            for i in range(2):
                W_struct[k*n+i, 0:7] = S_W[k*2+i, 0:7]
        
        # Fill remaining columns with random
        for col in range(7, rank):
            for row in range(n*m):
                U_struct[row, col] = rng.gauss(0, 0.5)
            for row in range(m*p):
                V_struct[row, col] = rng.gauss(0, 0.5)
            for row in range(p*n):
                W_struct[row, col] = rng.gauss(0, 0.5)
        initializations.append((U_struct, V_struct, W_struct))
    
    # Try each initialization
    for U_init, V_init, W_init in initializations:
        if _remaining() < 5.0:
            break
        
        # Alternate between gradient and alternating minimization
        Ua, Va, Wa, err = _alternating_minimize(U_init, V_init, W_init, n, m, p, max_iter=max_iter)
        
        if err < 1e-6:
            # Refine with gradient
            Ua, Va, Wa, err2 = _gradient_optimize(Ua, Va, Wa, n, m, p, max_iter=100, lr=0.001)
            if err2 < err:
                Ua, Va, Wa, err = Ua, Va, Wa, err2
        
        if err < 1e-6:
            # Snap to half-integers
            U_snap = _snap(Ua)
            V_snap = _snap(Va)
            W_snap = _snap(Wa)
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from just below schoolbook down to lower bound
    # Limit total time
    max_time = 140.0
    
    ranks_to_try = list(range(best_rank - 1, lower_bound - 1, -1))
    
    # Prioritize: try a few ranks near the expected optimum
    # Known: 3x3x3 has rank 23, so 3x5x6 might be around 40-60
    # Let's try a strategic subset
    strategic_ranks = []
    for r in range(max(lower_bound, 30), min(best_rank, 60)):
        strategic_ranks.append(r)
    
    # Combine: try strategic ranks first, then others
    all_ranks = strategic_ranks + [r for r in ranks_to_try if r not in strategic_ranks]
    
    attempts_per_rank = 3
    
    for rank in all_ranks:
        if time.time() - start_time > max_time - 10:
            break
        
        for attempt in range(attempts_per_rank):
            if time.time() - start_time > max_time - 5:
                break
            
            seed = rank * 100 + attempt
            result = _try_rank(n, m, p, rank, seed=seed, max_iter=200)
            
            if result is not None:
                U_list, V_list, W_list = result
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = rank
                break
        
        if best_rank < n * m * p:
            # Found improvement, try to go lower
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
