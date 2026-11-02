"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

def _build_schoolbook(n, m, p):
    """Return (U, V, W) lists-of-lists for schoolbook decomposition."""
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
    return U, V, W

def _verify_fast(Ua, Va, Wa, n, m, p, tol=1e-9):
    """Fast vectorized verification using numpy."""
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(Ua[a], Va[b] * Wa[c])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_all(Ua, Va, Wa, n, m, p, tol=1e-9):
    """Verify all tensor entries equal 1 (the only non-zero entries)."""
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the matrix multiplication
    # Actually, for matrix multiplication, the tensor is defined as:
    # T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B
    # And the output C[i, k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid triples.
    return _verify_fast(Ua, Va, Wa, n, m, p, tol)

def _try_remove_column(Ua, Va, Wa, n, m, p, col_idx, tol=1e-9):
    """Try removing column col_idx and see if the remaining columns still satisfy the tensor."""
    rank = Ua.shape[1]
    if rank <= 1:
        return None, None, None
    
    Ua_new = np.delete(Ua, col_idx, axis=1)
    Va_new = np.delete(Va, col_idx, axis=1)
    Wa_new = np.delete(Wa, col_idx, axis=1)
    
    if _verify_fast(Ua_new, Va_new, Wa_new, n, m, p, tol):
        return Ua_new, Va_new, Wa_new
    return None, None, None

def _greedy_column_removal(Ua, Va, Wa, n, m, p, deadline, tol=1e-9):
    """Greedily try to remove columns while maintaining validity."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = Ua.shape[1]
        for col_idx in range(rank):
            if (time.time() - start_time) > deadline:
                break
            Ua_new, Va_new, Wa_new = _try_remove_column(Ua, Va, Wa, n, m, p, col_idx, tol)
            if Ua_new is not None:
                Ua, Va, Wa = Ua_new, Va_new, Wa_new
                improved = True
                break
    return Ua, Va, Wa

def _structured_kronecker_decomposition(n, m, p):
    """
    Try to build a structured decomposition using Kronecker product ideas.
    For n x m times m x p, we can think of this as:
    - A is n x m, B is m x p, C = A @ B is n x p
    - The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    A structured approach: if we can factor the index space, we might get lower rank.
    For example, if m is composite, we can split j into j1, j2.
    
    For 3x5x5: n=3, m=5, p=5. Since 5 is prime, this doesn't help directly.
    But we can try other structured decompositions.
    
    One idea: use the fact that the tensor is "separable" in certain ways.
    T[i,j,k] = 1 for all valid (i,j,k). This is actually the all-ones tensor on the valid indices.
    
    Wait, no. The matrix multiplication tensor is NOT the all-ones tensor.
    Let me reconsider.
    
    The tensor for matrix multiplication is defined by:
    T[a, b, c] = 1 if a, b, c are consistent indices, 0 otherwise.
    
    In the flattened form:
    - a = i*m + j where i in [0,n), j in [0,m)
    - b = j*p + k where j in [0,m), k in [0,p)
    - c = k*n + i where k in [0,p), i in [0,n)
    
    The consistency condition is that the j in a equals the j in b, and the k in b equals the k in c, and the i in c equals the i in a.
    
    So T[i*m+j, j*p+k, k*n+i] = 1, and T = 0 for all other index combinations.
    
    This is a sparse tensor with exactly n*m*p = 75 non-zero entries.
    
    The rank of this tensor is what we're trying to minimize.
    
    For a structured decomposition, we can try:
    1. Random linear combinations of the standard basis
    2. Exploit symmetries if any
    
    Since 5 is prime and 3 is prime, there's not much structure to exploit.
    
    Let's try a different approach: random search for lower rank decompositions.
    """
    # For now, return None to indicate no structured decomposition found
    return None

def _random_search_for_lower_rank(n, m, p, current_rank, deadline, max_attempts=1000):
    """
    Try to find a lower rank decomposition via random search.
    This is unlikely to succeed for large rank reductions, but might find small improvements.
    """
    start_time = time.time()
    np.random.seed(42)
    random.seed(42)
    
    best_rank = current_rank
    best_decomp = None
    
    # Try ranks from current_rank-1 down to some minimum
    for target_rank in range(current_rank - 1, max(current_rank - 20, 1), -1):
        if (time.time() - start_time) > deadline:
            break
        
        attempts = 0
        while attempts < max_attempts and (time.time() - start_time) < deadline:
            attempts += 1
            
            # Generate random matrices U, V, W with the target rank
            # Use sparse random matrices to increase chance of exact match
            density = 0.3
            U = np.random.choice([0, 1, -1], size=(n*m, target_rank), p=[0.7, 0.15, 0.15])
            V = np.random.choice([0, 1, -1], size=(m*p, target_rank), p=[0.7, 0.15, 0.15])
            W = np.random.choice([0, 1, -1], size=(p*n, target_rank), p=[0.7, 0.15, 0.15])
            
            # Verify
            if _verify_fast(U.astype(np.float64), V.astype(np.float64), W.astype(np.float64), n, m, p):
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
        
        if best_decomp is not None:
            break
    
    return best_rank, best_decomp

def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    best_rank = n * m * p
    best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Verify schoolbook
    if not _verify_fast(Ua, Va, Wa, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Phase 1: Greedy column removal
    phase1_deadline = start_time + 20.0
    Ua, Va, Wa = _greedy_column_removal(Ua, Va, Wa, n, m, p, phase1_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 2: Try structured Kronecker decomposition
    phase2_deadline = start_time + 40.0
    structured = _structured_kronecker_decomposition(n, m, p)
    if structured is not None:
        Ua_s, Va_s, Wa_s = structured
        if Ua_s.shape[1] < best_rank and _verify_fast(Ua_s, Va_s, Wa_s, n, m, p):
            best_rank = Ua_s.shape[1]
            best_Ua, best_Va, best_Wa = Ua_s.copy(), Va_s.copy(), Wa_s.copy()
    
    # Phase 3: Random search for lower rank
    phase3_deadline = start_time + 120.0
    new_rank, new_decomp = _random_search_for_lower_rank(n, m, p, best_rank, phase3_deadline, max_attempts=500)
    if new_decomp is not None and new_rank < best_rank:
        best_rank = new_rank
        best_Ua, best_Va, best_Wa = new_decomp[0].astype(np.float64), new_decomp[1].astype(np.float64), new_decomp[2].astype(np.float64)
    
    # Phase 4: Final greedy removal on the best found
    phase4_deadline = deadline
    Ua, Va, Wa = _greedy_column_removal(best_Ua, best_Va, best_Wa, n, m, p, phase4_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Final verification
    if not _verify_fast(best_Ua, best_Va, best_Wa, n, m, p):
        # Fallback to schoolbook
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
        best_rank = n * m * p
    
    # Convert back to lists
    best_U = best_Ua.tolist()
    best_V = best_Va.tolist()
    best_W = best_Wa.tolist()
    
    # Ensure entries are exact (0, 1, or simple fractions)
    # For safety, round small values to 0
    for mat in [best_U, best_V, best_W]:
        for row in mat:
            for i in range(len(row)):
                if abs(row[i]) < 1e-10:
                    row[i] = 0.0
                elif abs(row[i] - round(row[i])) < 1e-10:
                    row[i] = float(round(row[i]))
    
    return {"U": best_U, "V": best_V, "W": best_W}
