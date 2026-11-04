"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
  1. Start with a known valid decomposition (Strassen-based, rank ~27 or schoolbook 42).
  2. Use a local search / coordinate descent approach to try to reduce rank by:
     - Merging terms: if two rank-1 terms can be combined into one (i.e., their
       outer products are linearly dependent in a way that allows a single term),
       reduce rank by 1.
     - Removing redundant terms: if a term is in the span of others, remove it.
  3. Verify the decomposition at each step.

The key insight for reducing rank below schoolbook is to use Strassen-like
combinations. The current program already does this but may not be optimal.

For 2x3 x 3x7, the theoretical lower bound is max(2*3, 3*7, 2*7) = 21.
The best known upper bound for this specific size is not trivially known,
but we can aim for something below 27.

Approach:
  - Use a block decomposition: split the 3x7 into 3x3 + 3x4, or other partitions.
  - For each block, use the best known decomposition.
  - Then try to optimize by removing redundant terms.

Known results:
  - 2x2 x 2x2: rank 7 (Strassen)
  - 2x2 x 2x3: rank ? (can be done with 7+4=11 or better)
  - 2x3 x 3x3: rank ?
  - 3x3 x 3x3: rank 23 (Bini et al.) or 21 (lower bound)

Let me try a different partition:
  A (2x3), B (3x7)
  Split B into B1 (3x3) and B2 (3x4)
  C = A*B1 + A*B2

  For A*B1 (2x3 x 3x3): 
    We can split A into A1 (2x2) and A2 (2x1)
    A1*B1: 2x2 x 2x3 (B1 first 2 rows) + A2*B1_last_row (2x1 x 1x3)
    
  This is getting complex. Let me just use the Strassen-based approach and verify.
"""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    """
    n, m, p = 2, 3, 7
    
    # Deadline guard
    deadline = time.time() + 140  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Helper to build a decomposition from a list of (u, v, w) dicts
    def build_matrices(terms, n, m, p):
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (u, v, w) in enumerate(terms):
            for (i, j), val in u.items():
                U[i * m + j, c] = val
            for (j, k), val in v.items():
                V[j * p + k, c] = val
            for (k, i), val in w.items():
                W[k * n + i, c] = val
        return U, V, W
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(U.shape[1]):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Strassen 2x2 x 2x2 terms
    def get_strassen_2x2():
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (0,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1, (1,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1, (1,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1, (1,1):1}),
        ]
    
    def offset_terms(terms, i_off, j_off, k_off):
        result = []
        for u, v, w in terms:
            u_new = {(i + i_off, j + j_off): val for (i, j), val in u.items()}
            v_new = {(j + j_off, k + k_off): val for (j, k), val in v.items()}
            w_new = {(k + k_off, i + i_off): val for (k, i), val in w.items()}
            result.append((u_new, v_new, w_new))
        return result
    
    # Build initial decomposition
    all_terms = []
    strassen = get_strassen_2x2()
    
    # Partition: A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x7) = [B1 (2x7); B2 (1x7)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 x 2x7
    # Split B1 into three 2x? blocks: 2x3 + 2x2 + 2x2
    # Or 2x2 + 2x2 + 2x2 + 2x1
    
    # Let's do: B1 cols 0-1 (2x2), cols 2-3 (2x2), cols 4-5 (2x2), col 6 (2x1)
    all_terms.extend(offset_strassen(0, 0, 0))
    all_terms.extend(offset_strassen(0, 0, 2))
    all_terms.extend(offset_strassen(0, 0, 4))
    
    # 2x2 x 2x1 schoolbook (4 terms)
    for i in range(2):
        for j in range(2):
            k = 6
            all_terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    
    # A2*B2: 2x1 x 1x7
    # This is rank 2: for each i, A[i,2] * B[2,k] for all k
    # But in tensor form, we need separate terms for each (i,k) pair? No.
    # Actually, for 2x1 x 1x7, the tensor has entries T[i,2,k] = 1 for i in {0,1}, k in {0,...,6}
    # This is a sum of 14 rank-1 tensors (schoolbook), but we can do better.
    # Actually, it's rank 2: 
    #   Term 1: u={(0,2):1}, v={(2,k):1 for k in range(7)}, w={(k,0):1 for k in range(7)}
    #   Term 2: u={(1,2):1}, v={(2,k):1 for k in range(7)}, w={(k,1):1 for k in range(7)}
    for i in range(2):
        u = {(i, 2): 1}
        v = {(2, k): 1 for k in range(7)}
        w = {(k, i): 1 for k in range(7)}
        all_terms.append((u, v, w))
    
    rank = len(all_terms)
    
    # Verify
    U, V, W = build_matrices(all_terms, n, m, p)
    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": [[int(val) for val in row] for row in U.tolist()],
            "V": [[int(val) for val in row] for row in V.tolist()],
            "W": [[int(val) for val in row] for row in W.tolist()]
        }
    
    # Try to reduce rank by removing redundant terms
    # A term c is redundant if for all (i,j,k), 
    #   U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] can be expressed as a linear combination
    #   of other terms' contributions.
    
    # Simple approach: try removing each term and check if the decomposition still holds.
    # This is O(rank * n*m*p * rank) which is fine for rank ~27.
    
    removed = True
    while removed and time_left() > 5:
        removed = False
        for c in range(len(all_terms)):
            if time_left() < 5:
                break
            # Try removing term c
            trial_terms = all_terms[:c] + all_terms[c+1:]
            U_t, V_t, W_t = build_matrices(trial_terms, n, m, p)
            if verify(U_t, V_t, W_t, n, m, p):
                all_terms = trial_terms
                removed = True
                break
    
    rank = len(all_terms)
    U, V, W = build_matrices(all_terms, n, m, p)
    
    # Final verification
    if not verify(U, V, W, n, m, p):
        # Should not happen, but fall back
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": [[int(val) for val in row] for row in U.tolist()],
            "V": [[int(val) for val in row] for row in V.tolist()],
            "W": [[int(val) for val in row] for row in W.tolist()]
        }
    
    U_final = [[int(val) for val in row] for row in U.tolist()]
    V_final = [[int(val) for val in row] for row in V.tolist()]
    W_final = [[int(val) for val in row] for row in W.tolist()]
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res['U'][0])
    print("Rank:", rank)
