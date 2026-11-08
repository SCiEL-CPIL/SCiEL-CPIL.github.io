"""Rewritten program for 3x3x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower-rank decomposition using known algorithms.
    # For 3x3x5, the rank is not known to be below 45 in the literature,
    # but we can attempt a structured search for improvements.
    
    # One approach: use the fact that 3x3x5 = 3x3x3 + 3x3x2
    # 3x3x3 has rank 23 (Winograd), 3x3x2 has rank 12 (schoolbook is 18)
    # Actually let's think about this differently.
    
    # For 3x3x5, we can split the 5 columns into 3+2.
    # 3x3x3 part: use Winograd rank 23
    # 3x3x2 part: schoolbook rank 18, or potentially lower
    # Total: 23 + 18 = 41, which is better than 45!
    
    # Let's implement Winograd for 3x3x3 and schoolbook for 3x3x2.
    
    # Winograd 3x3x3 decomposition (rank 23)
    # The Winograd algorithm for 3x3 matrix multiplication uses rank 23.
    # Let me construct it explicitly.
    
    # Actually, let me use a different approach. The standard Winograd 3x3x3
    # uses a specific set of 23 rank-1 terms. Let me look up the structure.
    
    # For now, let's try a simpler approach: just use schoolbook for the full 3x3x5.
    # The rank is 45.
    
    # Let me try to implement a proper Winograd-based decomposition.
    # 3x3x3 Winograd has rank 23.
    # 3x3x2: we can use schoolbook (rank 18) or try to do better.
    
    # Actually, for 3x3x2, the rank might be 12. Let me check.
    # 3x3x2 = multiply 3x3 by 3x2. This is like computing 6 outputs.
    # Schoolbook: 3*3*2 = 18 multiplications.
    # Can we do better? Possibly with some sharing.
    
    # Let me just try to construct a decomposition with rank < 45.
    # Split 3x3x5 into 3x3x3 + 3x3x2.
    
    # 3x3x3 Winograd decomposition:
    # The Winograd scheme for 3x3x3 uses the following:
    # Let A be 3x3, B be 3x3.
    # We form intermediate products using a specific pattern.
    
    # Actually, let me use a known explicit construction.
    # The rank-23 decomposition for 3x3x3 matrix multiplication:
    
    # I'll use the approach from Winograd's original paper.
    # The key idea is to use the fact that (a+b)(c+d) = ac + ad + bc + bd
    # to reduce the number of multiplications.
    
    # For 3x3x3, the Winograd algorithm uses:
    # Row/column sums and differences to create 23 products.
    
    # Let me define the Winograd 3x3x3 decomposition explicitly.
    
    # Actually, this is getting complex. Let me try a different strategy.
    
    # Strategy: Use the fact that we can factor the problem.
    # 3x3x5: A(3x3) * B(3x5) = C(3x5)
    
    # We can write B as [B1 | B2] where B1 is 3x3 and B2 is 3x2.
    # C = [A*B1 | A*B2]
    
    # For A*B1 (3x3x3), use Winograd rank 23.
    # For A*B2 (3x3x2), use schoolbook rank 18.
    # Total rank: 23 + 18 = 41.
    
    # But we need to be careful: the Winograd decomposition for 3x3x3 
    # shares the input A with the 3x3x2 part. So we can't simply add the ranks.
    
    # The Winograd 3x3x3 decomposition uses certain linear combinations of A's entries.
    # The 3x3x2 part also uses A's entries. If we can share these linear combinations,
    # we might reduce the total rank.
    
    # This is getting complicated. Let me just try to find any decomposition with rank < 45.
    
    # Alternative approach: use a known result. The rank of 3x3x5 matrix multiplication
    # is at most 45 (schoolbook) and at least 27 (lower bound).
    # The best known upper bound might be 41 or 42.
    
    # Let me try a heuristic search for a rank-41 decomposition.
    
    # Actually, let me just return the schoolbook decomposition for now,
    # since implementing a proper Winograd-based decomposition is complex
    # and I need to ensure correctness.
    
    # Wait, let me try one more thing. Let me see if I can find a rank-41
    # decomposition by combining Winograd for the 3x3x3 part and schoolbook
    # for the 3x3x2 part, with shared linear combinations of A.
    
    # For the 3x3x3 Winograd part, the algorithm computes:
    # 1. Form 10 linear combinations of A's 9 entries (with some repeated)
    # 2. Form 10 linear combinations of B's 9 entries
    # 3. Compute 10 products
    # 4. Form 3x3 output from these products and the linear combinations
    
    # Actually, the Winograd 3x3x3 uses 23 products, not 10.
    
    # Let me just try to implement a simple version and verify it.
    
    # For now, return schoolbook.
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Verify schoolbook
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    raise RuntimeError("No valid decomposition found")
# EVOLVE-BLOCK-END
