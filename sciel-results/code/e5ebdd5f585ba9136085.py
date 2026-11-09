"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, trials=100, seed=42):
    """Verify that the decomposition exactly reconstructs matrix multiplication."""
    rng = np.random.RandomState(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        
        C_recon = np.zeros((n, p))
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    C_recon[i,k] += s
        
        if not np.allclose(C, C_recon, atol=1e-10):
            return False
    return True

def _construct_rank27(n, m, p):
    """
    Construct a rank-27 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy: Start from schoolbook rank 30 and merge 3 pairs of terms.
    For each merge, we exploit the bilinear structure.
    
    The MM tensor: T[i,j,k] = delta such that sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
    Actually T[i,j,k] = 1 for all i,j,k in the sense that the tensor is the 
    identity mapping from A⊗B to C via contraction over j.
    
    More precisely: The bilinear map (A,B) -> A@B can be written as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Schoolbook: 30 terms, one for each (i,j,k).
    
    For rank 27, we need to eliminate 3 degrees of freedom.
    
    Key insight: For fixed j, the terms across i and k form a rank-1 structure.
    We can use the fact that A[i,j] and B[j,k] are shared across i and k respectively.
    
    Let's use a structured approach:
    - Terms 0-9: handle (i=0, j=0, all k) and (i=1, j=0, all k) with shared B[0,k]
    - Actually, let me use a different partitioning.
    
    Better approach: Use the decomposition where we group by j:
    For each j (0,1,2), we have a subproblem of multiplying a 2-vector by a 5-vector.
    That's a 2x1 * 1x5 multiplication, which has rank 2*5 = 10 in schoolbook.
    But we can do better: the rank of 2x1 * 1x5 is actually 2+5-1 = 6 (Strassen-like).
    
    Wait, that's not quite right either. Let me think differently.
    
    For the full problem, let me try to directly construct rank 27 by solving a linear system.
    
    Fix a sparse pattern for U, V, W with 27 columns. Most entries are 0 or 1.
    Leave a small number of entries as unknowns. Then the tensor equations become linear.
    """
    # For 2x3 * 3x5, n=2, m=3, p=5
    # Schoolbook uses 30 terms. We want 27.
    
    # Let me try a specific construction:
    # Use 27 terms where 24 are "standard" and 3 are "correction" terms.
    
    # Alternative: Use the known result that rank(2x3 * 3x5) <= 27
    # by explicit construction.
    
    # Let me try: partition the 30 (i,j,k) triples into 27 groups,
    # where 3 groups each contain 2 triples that can be combined.
    
    # The tensor entry T[i,j,k] must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is 30 equations. With 27 terms, we have 27*6 = 162 unknowns (before sparsity).
    # If we fix most to 0 or 1, we get a small linear system.
    
    # Let me use a specific pattern:
    # Terms 0-26. For each term c, assign it to a specific (i,j,k) or pair.
    
    # Construction: 
    # - 27 terms
    # - For j=0: 10 triples (i in {0,1}, k in {0..4})
    # - For j=1: 10 triples
    # - For j=2: 10 triples
    # Total: 30. We need to merge 3 pairs.
    
    # For each j, we have a 2x5 grid. We can merge pairs that share structure.
    # E.g., for j=0: merge (i=0,k=0) with (i=1,k=0)? No, they have different i.
    
    # Actually, the constraint is that for a single term c, the product
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must be nonzero only for specific (i,j,k).
    
    # Let me try a different approach: use random sparse initialization and solve.
    
    r = 27
    # Initialize with schoolbook-like structure but with 27 columns
    # Use 27 of the 30 standard terms, then add correction.
    
    # Actually, let me try to directly solve using numpy.
    # Set up: we want U, V, W with r=27 columns.
    
    # Use a structured initialization:
    # Columns 0-9: for j=0, all (i,k) pairs
    # Columns 10-19: for j=1, all (i,k) pairs  
    # Columns 20-26: for j=2, only 7 of the 10 (i,k) pairs, plus 3 "mixed" terms
    
    # This is getting complicated. Let me try a simpler approach:
    # Use the fact that we can express 3 of the 30 terms as linear combinations
    # of others, if we allow rational coefficients.
    
    # For now, return a known valid rank-30 solution and let the caller verify.
    # But we want rank 27...
    
    # Let me try a direct construction with specific values.
    
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # Standard terms for 27 of the 30 (i,j,k) triples
    # Skip 3 specific triples and handle them with mixed terms
    
    col = 0
    skipped = set()
    
    # For j=0, j=1: use all 10 terms each (20 terms)
    for j in range(2):
        for i in range(n):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # For j=2: we have 10 triples but only 7 columns left (col=20..26)
    # Use 7 standard terms and 3 "correction" terms
    
    # Standard terms for j=2, skip (i=0,k=0), (i=0,k=1), (i=1,k=0)
    skip_j2 = {(0,0), (0,1), (1,0)}
    for i in range(n):
        for k in range(p):
            if (i,k) not in skip_j2:
                U[i*m+2, col] = 1.0
                V[2*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Now col should be 27. The 3 skipped triples need correction.
    # For (i=0, j=2, k=0): need U[0*3+2, c]*V[2*5+0, c]*W[0*2+0, c] to contribute 1
    # But U[2, c], V[10, c], W[0, c] are already set for other terms.
    
    # This approach doesn't work cleanly because the entries are shared.
    
    # Let me abandon this and use a numerical approach.
    return None

def _try_rank_numeric(r, n, m, p, deadline):
    """
    Try to find a rank-r decomposition by solving a linear system
    with a structured sparse template.
    """
    # For rank 27, use a specific structured template
    if r != 27:
        return None
    
    # Use a random sparse initialization and iteratively refine
    rng = np.random.RandomState(42)
    
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # Initialize with a perturbed schoolbook-like structure
    for c in range(r):
        # Assign each column to a random (i,j,k) triple
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    
    # Now try to adjust to satisfy all 30 tensor equations
    # The equations are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of 30 nonlinear equations in 162 unknowns (27*6).
    # With the initialization, most are already satisfied.
    
    # Check which equations are violated
    def check_residuals(U, V, W):
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res[i,j,k] = s - 1.0
        return res
    
    res = check_residuals(U, V, W)
    if np.max(np.abs(res)) < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try to fix by adjusting specific entries
    # For each violated (i,j,k), find a column c where we can adjust
    # This is a heuristic approach
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank 27 first
    result = _construct_rank27(n, m, p)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p, trials=100):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
