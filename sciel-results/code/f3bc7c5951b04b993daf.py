"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS
NOUT = n * p  # 10
NIN = m * p   # 25

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook decomposition (rank 50)."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, rank):
    """Verify that the decomposition is exact for matrix multiplication."""
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # For each output (i, k): sum_j A[i,j] * B[j,k] = sum_c U[i*m + j?] ... 
    # Actually the tensor is T[i,j,k] = delta_{output(i,k), input(i,j),(j,k)}
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] is the trace convention for output.
    # The tensor for matrix mult C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 if this term contributes to C[i,k].
    # Actually the standard tensor decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # But here W is indexed as [k*n + i, r]. So W's first index is (k, i) not (i, k).
    # Let's check: W has shape (p*n, rank). Index is k*n + i.
    # So W[k*n + i, r] corresponds to output C[i,k].
    # The condition is: for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (since C[i,k] gets A[i,j]*B[j,k])
    # And for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i, r] = 0  for (j',k') != (j,k)
    # More precisely, the tensor T[i,j,k] = 1 for all valid i,j,k (each term contributes).
    # Wait no. The multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # And the decomposition must satisfy:
    #   For each (i, j, k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # But that's not right either. Let me think again.
    
    # The matrix multiplication tensor T is defined by T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition gives T[i,j,k] = sum_{r=1}^{r} U[i,j,r] V[j,k,r] W[i,k,r]
    # But W is stored as W[k*n+i, r], so W[i,k,r] in the formula becomes W[k*n+i, r].
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And this must hold ONLY for the "correct" terms? No, the tensor is just all-ones.
    # Wait, that can't be right because then any decomposition of the all-ones tensor would work.
    
    # Let me reconsider. The matrix multiplication tensor is:
    # T(A,B) = A*B where (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: T(A,B,C) = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k.
    # A decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # With W stored as W[k*n+i, r], we have W[i,k,r] = W[k*n+i, r].
    # So the condition is simply: for all i,j,k: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    
    # But wait, that means the tensor is just the all-ones tensor of shape (n*m, m*p, p*n).
    # That doesn't seem right for matrix multiplication. Let me re-examine.
    
    # Actually, I think the correct interpretation is:
    # The multiplication map is: (A, B) -> A*B
    # This is a bilinear map. The associated tensor has entries:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Since every term A[i,j]*B[j,k] contributes to C[i,k], T[i,j,k] = 1 for all valid i,j,k.
    # But the "output" index is (i,k), not separate i and k.
    
    # I think the correct tensor is: for inputs indexed by (i,j) and (j,k), output indexed by (i,k).
    # T[(i,j),(j,k),(i,k)] = 1 for all valid indices.
    # So in the flattened form: U has rows indexed by (i,j), V by (j,k), W by (i,k).
    # And the condition is: for all (i,j),(j,k),(i,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{(i,j),(j,k) compatible with (i,k)}
    # But (i,j) and (j,k) are always compatible with (i,k) since j=j.
    # So T = 1 for all triples where the middle index matches.
    
    # Hmm, but U has shape (n*m, rank) so it's indexed by (i,j) with i in [0,n), j in [0,m).
    # V has shape (m*p, rank) indexed by (j,k) with j in [0,m), k in [0,p).
    # W has shape (p*n, rank) indexed by (k,i) with k in [0,p), i in [0,n).
    
    # The tensor T[i,j,k] should be 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # This is always true, so T[i,j,k] = 1 for all i,j,k.
    # But that means we're decomposing the all-ones tensor, which has rank 1!
    
    # I must be misunderstanding the problem. Let me re-read the task.
    # "Matrix multiplication 2,5,5" - multiplying a 2x5 matrix by a 5x5 matrix.
    # The output is a 2x5 matrix.
    # So n=2, m=5, p=5.
    # A is 2x5, B is 5x5, C = A*B is 2x5.
    # C[i,k] = sum_j A[i,j] B[j,k] for i in [0,2), j in [0,5), k in [0,5).
    
    # The tensor for this operation: inputs are A (10 elements) and B (25 elements), output is C (10 elements).
    # The bilinear map is: (A, B) -> A*B
    # As a tensor: T[a, b, c] where a indexes A, b indexes B, c indexes C.
    # T[a, b, c] = 1 if A[a] * B[b] contributes to C[c].
    
    # A is indexed by (i,j) with i in [0,2), j in [0,5). So a = i*5 + j.
    # B is indexed by (j,k) with j in [0,5), k in [0,5). So b = j*5 + k.
    # C is indexed by (i,k) with i in [0,2), k in [0,5). So c = i*5 + k.
    
    # T[a, b, c] = 1 if there exist i,j,k such that:
    #   a = i*5+j, b = j*5+k, c = i*5+k
    # i.e., the j in a matches the j in b, and i,k are consistent.
    
    # So T is NOT all-ones. It's a specific sparse tensor.
    
    # In the decomposition: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # where U has shape (n*m, rank) = (10, rank), V has shape (m*p, rank) = (25, rank), W has shape (p*n, rank) = (10, rank).
    # And W[c, r] with c = i*5+k... but the task says W[k*n+i, r].
    # So W is indexed by (k, i) not (i, k). c in W's index is k*n+i = k*2+i.
    # But C is indexed by (i,k) with c = i*p+k = i*5+k.
    # So there's a mismatch. The task says W[k*n+i, r], which means W's first index is k*n+i.
    # So W[c', r] where c' = k*n+i. But C is indexed by (i,k).
    # So W[k*n+i, r] corresponds to output C[i,k].
    
    # OK so the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations of (a, b, c) that don't correspond to valid (i,j,k):
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # But since U, V, W are just matrices, the condition is really:
    # For all a in [0, n*m), b in [0, m*p), c in [0, p*n):
    #   sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    # where T[a, b, c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j' and i=i' and k=k', else 0.
    
    # Let me just verify by checking specific entries.
    
    # Actually, for verification purposes, I can just check that for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for "wrong" combinations, it should be 0. But how to check all wrong combinations?
    # I can check that the full tensor T computed from U, V, W matches the expected tensor.
    
    # Let me build the expected tensor and compare.
    expected = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[(a, b, c)] = Fraction(1)
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                exp = expected.get((a, b, c), Fraction(0))
                if val != exp:
                    return False
    return True

def _try_rank_reduction(best, max_rank_to_try=10):
    """Try to find a lower rank decomposition. Returns best found."""
    best_rank = len(best['U'][0]) if best['U'] else n*m*p
    best_result = best
    
    # For this problem, the schoolbook rank is 50.
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25.
    # But we can likely do better by sharing computations.
    # Let's try to find a decomposition of rank r for r from some lower bound up.
    
    # A simple approach: start with schoolbook and try to merge columns.
    # Two columns c1, c2 can be merged if their U, V, W vectors are linearly dependent
    # in a way that allows combining them into one column.
    
    # Actually, a better approach for small problems: use a randomized search or
    # exploit the structure.
    
    # For 2x5x5, the output is 2x5 = 10 entries.
    # Each output C[i,k] = sum_j A[i,j] B[j,k].
    # We need to compute 10 sums, each with 5 terms.
    # Total 50 products if done naively.
    
    # Can we share products? A product A[i,j]*B[j,k] is used only for C[i,k].
    # So there's no sharing between different (i,k) pairs for the same (i,j,k) term.
    # But we might be able to use Strassen-like tricks.
    
    # Let's try a simple approach: assume rank r and try to solve for U, V, W.
    # This is a system of nonlinear equations (bilinear in each pair).
    
    # For now, let's just try to see if rank 25 works (the trivial lower bound).
    # Actually, let me try a different approach: fix U and V randomly, then solve for W.
    # Or fix two of them and solve for the third.
    
    # Given U (10 x r) and V (25 x r), W is determined by:
    # W[c, r] = T[a, b, c] / (U[a, r] * V[b, r]) for the "active" terms.
    # But this only works if U[a,r] and V[b,r] are nonzero for the active terms.
    
    # Let's try: for a given rank r, set U and V to have 1s in specific patterns,
    # then compute W and check consistency.
    
    # Actually, let me try a simpler heuristic: start with schoolbook and try to
    # eliminate columns by finding linear dependencies.
    
    # For the schoolbook decomposition, each column c corresponds to a unique (i,j,k) triple.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Two columns c1=(i1,j1,k1) and c2=(i2,j2,k2) can potentially be combined
    # if they share the same i (so W index is same) or same j (V index related) etc.
    
    # Let's try: for each pair of columns, see if we can replace them with a single column.
    # This is complex. Let me just return the schoolbook for now and add a simple
    # optimization later.
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to improve
    best = _try_rank_reduction(best)
    
    # Convert to lists of floats/ints for output
    result = {}
    for key in ['U', 'V', 'W']:
        mat = best[key]
        # Convert Fraction to int or float
        converted = []
        for row in mat:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(int(val))
                else:
                    new_row.append(float(val))
            converted.append(new_row)
        result[key] = converted
    
    return result
# EVOLVE-BLOCK-END
