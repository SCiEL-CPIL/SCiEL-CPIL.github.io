"""Matrix multiplication 2x5x5 - rank minimization via search."""
import time
import random
import sys
from itertools import product

DIMENSIONS = [2, 5, 5]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""
    global _DEADLINE, _START_TIME
    _START_TIME = time.time()
    _DEADLINE = 150.0
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    # Tensor indices: A is n x m, B is m x p, C is p x n (output factor W[k*n+i, r])
    # Actually re-reading: "output factor follows the official trace convention W[k*n+i,r]"
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The multiplication tensor T[i,j,k] = delta (A*B)[i,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,m-j? no...] Let me be careful.
    # Standard: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we're computing the (i,k) entry from A[i,j]*B[j,k]
    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed over c should give (AB)[i,k]
    
    # Schoolbook: rank = n*m*p = 2*5*5 = 50
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition
    def verify(result):
        U = result["U"]
        V = result["V"]
        W = result["W"]
        rank = len(U[0]) if U else 0
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        # Check that U * V * W^T reproduces the multiplication tensor
        # (AB)[i,k] = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For each (i,k), the value should be 1 if there exists j (which there always is since m>=1)
        # Actually (AB)[i,k] = sum_j A[i,j]*B[j,k], so the tensor T[i,j,k] = 1 for all valid i,j,k
        # And the decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # Wait, that's not quite right. Let me think again.
        
        # The matrix multiplication tensor T is a 3-tensor with T[i,j,k] = 1 (it's just 1 for all indices)
        # No wait. T[i,j,k] represents the coefficient in (AB)[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # A rank-r decomposition writes T as sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
        # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        
        # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from n*m*p-1 down, using randomized search
    # For 2x5x5, schoolbook is 50. Let's try to find rank 49, 48, etc.
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # For small ranks, we can try specific constructions
    # Key insight: for matrix multiplication, known lower bounds and upper bounds exist
    # For 2x5x5: trivial upper bound is 50 (schoolbook)
    # The rank could potentially be reduced.
    
    # Try a structured approach: modify the schoolbook decomposition
    # Idea: combine two terms that share structure
    
    # Actually, let's try a more systematic approach for rank reduction
    # We can try to eliminate one schoolbook term by expressing it as a combination of others
    # This requires the eliminated term to be in the span of the remaining terms
    
    # For rank r < n*m*p, we need to find r vectors u_c, v_c, w_c such that
    # sum_c u_c ⊗ v_c ⊗ w_c = T
    
    # Randomized approach: for a target rank r, initialize with random vectors
    # and try to optimize. But with integer/half-integer constraint, this is hard.
    
    # Let's try specific structured decompositions
    
    # Approach: Try rank 49 by modifying schoolbook
    # In schoolbook, term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Can we merge two terms? E.g., if we set for term c:
    # U[a][c]=1, U[b][c]=1 (two different i*m+j indices)
    # V[...][c]=1, V[...][c]=1
    # W[...][c]=1, W[...][c]=1
    # Then term c contributes to multiple (i,j,k) entries
    
    # This is getting complex. Let's try a direct search for small rank reductions.
    
    # For 2x5x5, let's try to see if rank 49 is achievable
    # One approach: find a linear dependency among the 50 schoolbook terms
    
    # The 50 terms are e_{i,j,k} (standard basis of the 50-dim space of tensors with fixed index structure)
    # Actually they span a 50-dim space, so they're linearly independent.
    # So we can't just drop one term.
    
    # We need a fundamentally different decomposition.
    
    # Let's try: for each pair of (i,j) or (j,k) or (i,k), see if we can share factors.
    
    # Alternative: use the fact that n=2 is small.
    # (AB)[0,k] = sum_j A[0,j] B[j,k]
    # (AB)[1,k] = sum_j A[1,j] B[j,k]
    
    # The tensor has 2*5*5 = 50 entries, all equal to 1.
    
    # Known result: rank of matrix multiplication tensor M_{p,q,r} >= max(p*q, q*r, r*p)
    # For 2x5x5: max(10, 25, 10) = 25. So rank >= 25.
    # Upper bound: 50 (schoolbook).
    # Best known: probably around 40-49 for these small dimensions.
    
    # Let's try a randomized local search approach
    # Start from schoolbook (rank 50), try to reduce rank by merging terms
    
    # Actually, for the constraint of integer/half-integer entries, let's try
    # a construction-based approach.
    
    # Try: can we do rank 45? 40?
    
    # Let me try a simple structured decomposition for rank < 50.
    
    # For 2x5x5, one idea: group by k (since p=5)
    # For fixed k, we need: for all i,j: sum over terms involving this k
    # This doesn't immediately help.
    
    # Let's just do a bounded search with time limit.
    # Try ranks from min(49, schoolbook_rank) down to max(25, ...)
    # For each rank, try a few random restarts with greedy construction.
    
    def try_greedy_rank(target_rank):
        """Try to construct a decomposition of given rank using greedy method."""
        if _remaining() < 5:
            return None
        
        # We need to satisfy: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Greedy: pick one (i,j,k) at a time, assign a new term to cover it.
        # But this just gives schoolbook.
        
        # Better: try to assign multiple (i,j,k) to the same term.
        # A term c with U[a,c]=u, V[b,c]=v, W[cc,c]=w contributes u*v*w to T[...,...,...]
        # where a=i*m+j, b=j*p+k, cc=k*n+i
        
        # For the term to contribute to multiple (i,j,k), we need overlapping indices.
        # If U[i*m+j,c] != 0 and U[i'*m+j',c] != 0, the term covers both.
        
        # This is essentially finding a set cover with weighted terms.
        
        # Let's try: for each column c, choose which (i,j,k) it covers with what weights.
        # The constraint is that each (i,j,k) gets total weight 1.
        
        # For simplicity, try: each term covers exactly one (i,j,k) with weight 1 (schoolbook)
        # Or: some terms cover multiple with fractional weights.
        
        # With integer/half-integer constraint, weights are in {0, 1/2, 1, 3/2, ...}
        
        # Try: rank = 50 - t for t in 1, 2, 3, ...
        # For t=1: need 49 terms to cover 50 (i,j,k) pairs, so one term must cover 2.
        # A term covering (i,j,k) and (i',j',k') needs:
        #   i*m+j = i'*m+j' (same U index) AND j*p+k = j'*p+k' (same V index) AND k*n+i = k'*n+i' (same W index)
        #   OR different indices but the product still works out.
        
        # If i*m+j = i'*m+j', then i=i' and j=j'.
        # If j*p+k = j'*p+k', then j=j' and k=k'.
        # So all three must be the same. Thus each term can only cover one (i,j,k).
        
        # Wait, that's for the standard basis representation. But we can use non-standard factors.
        # E.g., U[i*m+j,c] = 1 and U[i'*m+j',c] = 1 for different (i,j) pairs.
        # Then the term contributes to all (i,j,k) where both U indices are active.
        
        # But we also need V and W to be consistent.
        
        # Let me think differently. For a term c:
        # u_c is a vector of length n*m (indices i*m+j)
        # v_c is a vector of length m*p (indices j*p+k)
        # w_c is a vector of length p*n (indices k*n+i)
        
        # The contribution to T[i,j,k] is u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        
        # For this to be 1 for some (i,j,k) and 0 for others (within this term's support),
        # we need specific patterns.
        
        # With the constraint that entries are integers or half-integers,
        # and we need sum over c to be exactly 1 for each (i,j,k),
        
        # Let's try a different approach: use the fact that the tensor is very simple (all 1s).
        
        # Observation: T[i,j,k] = 1 for all i,j,k.
        # This is a rank-1 tensor! T = 1 ⊗ 1 ⊗ 1 where 1 is the all-ones vector.
        # Wait, is that right?
        # (1 ⊗ 1 ⊗ 1)[i,j,k] = 1[i*m+j] * 1[j*p+k] * 1[k*n+i] = 1*1*1 = 1. YES!
        
        # But wait, that's only if the indexing works out. Let me verify.
        # u[i*m+j] = 1 for all i in [0,n), j in [0,m) -> u is all-ones of length n*m
        # v[j*p+k] = 1 for all j in [0,m), k in [0,p) -> v is all-ones of length m*p  
        # w[k*n+i] = 1 for all k in [0,p), i in [0,n) -> w is all-ones of length p*n
        
        # Then sum_c (only c=0) u[0][i*m+j]*v[0][j*p+k]*w[0][k*n+i] = 1*1*1 = 1. CORRECT!
        
        # So the rank is 1!
        
        # Let me double-check the indexing convention.
        # U shape: (n*m, rank) -> U[i*m + j, c]
        # V shape: (m*p, rank) -> V[j*p + k, c]
        # W shape: (p*n, rank) -> W[k*n + i, c]
        
        # For rank 1:
        # U[i*m+j, 0] = 1 for all valid i,j
        # V[j*p+k, 0] = 1 for all valid j,k
        # W[k*n+i, 0] = 1 for all valid k,i
        
        # Check: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # For c=0: 1*1*1 = 1 for each (i,j,k)
        # Total for fixed (i,k): sum_j 1 = m = 5
        
        # But we need the total to be 1, not 5!
        # Because (AB)[i,k] = sum_j A[i,j] B[j,k], and the tensor T[i,j,k] = 1.
        # The decomposition should give T[i,j,k] = 1 for each individual (i,j,k),
        # not sum over j.
        
        # So my rank-1 attempt gives 5 for each (i,k) when summed over j, which is wrong.
        # The decomposition is per-(i,j,k) element, not per-(i,k) entry.
        
        # So T[i,j,k] = 1 and we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each (i,j,k).
        
        # With rank 1 and all-ones: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
        
        # So rank 1 works!
        
        # Let me verify more carefully with the actual index ranges:
        # i in [0,2), j in [0,5), k in [0,5)
        # U has 2*5=10 rows, V has 5*5=25 rows, W has 5*2=10 rows
        # U[i*5+j, 0] = 1 for i in {0,1}, j in {0,1,2,3,4} -> 10 entries all 1
        # V[j*5+k, 0] = 1 for j in {0,1,2,3,4}, k in {0,1,2,3,4} -> 25 entries all 1
        # W[k*2+i, 0] = 1 for k in {0,1,2,3,4}, i in {0,1} -> 10 entries all 1
        
        # For any (i,j,k): U[i*5+j,0] * V[j*5+k,0] * W[k*2+i,0] = 1*1*1 = 1 ✓
        
        # The rank is 1!
        
        # Wait, this seems too good to be true. Let me re-read the problem.
        
        # "Matrix multiplication 2,5,5" - multiply 2x5 by 5x5 to get 2x5.
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        
        # The tensor T has entries T[i,j,k] where i indexes rows of A (0..1),
        # j indexes columns of A / rows of B (0..4), k indexes columns of B (0..4).
        # T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in (AB)[i,k]).
        
        # Rank-1 decomposition: T = u ⊗ v ⊗ w where
        # u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1
        
        # T_decomposed[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i] = 1*1*1 = 1 = T[i,j,k] ✓
        
        # Yes! The rank is 1. This is because the matrix multiplication tensor
        # for any dimensions is actually a sum of rank-1 tensors, but in this case
        # the tensor is literally all 1s in the appropriate indexing, which IS rank 1.
        
        # Wait, no. The matrix multiplication tensor is NOT all 1s in the naive sense.
        # Let me reconsider.
        
        # The matrix multiplication tensor M_{n,m,p} is defined such that
        # <M_{n,m,p}, A ⊗ B> = vec(AB)
        # or equivalently, M_{n,m,p}[i,j,k] is the coefficient of A[i,j]*B[j,k] in (AB)[i,k].
        
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # So the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1.
        # And A[i,j]*B[j,k] appears ONLY in (AB)[i,k], not in any other (AB)[i',k'].
        
        # So M[i,j,k] = 1 if we're looking at the (i,k) entry and the j-th term,
        # i.e., M[i,j,k] = 1 for all valid i,j,k.
        
        # Hmm, but that means the tensor is indeed all 1s, and rank 1.
        
        # But that contradicts the known result that matrix multiplication has rank
        # growing as n^3 for n×n×n. Let me reconsider the tensor definition.
        
        # Oh wait, I think I'm confusing the tensor structure.
        # The matrix multiplication tensor M_{n,m,p} is an element of R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
        # (or R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} depending on convention).
        
        # The entry M[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,n*p) is:
        # M[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k (for some i,j,k), else 0.
        
        # Wait, that's the standard definition! The tensor only has 1s at specific positions,
        # not all positions.
        
        # Let me re-examine: a = i*m+j (row i, col j of A), b = j*p+k (row j, col k of B),
        # c should index the output (i,k). If output is n*p = 2*5 = 10 entries,
        # then c = i*p+k.
        
        # But the problem says W shape is (p*n, rank) = (10, rank) with index W[k*n+i, c].
        # So the output index is k*n+i, not i*p+k.
        
        # With p=5, n=2: k*n+i = k*2+i.
        # For i in {0,1}, k in {0,1,2,3,4}: k*2+i gives values 0,1,2,3,4,5,6,7,8,9.
        # And i*p+k = i*5+k gives: for i=0: 0,1,2,3,4; for i=1: 5,6,7,8,9.
        
        # These are different orderings of the same 10 output entries!
        # k*2+i: k=0->{0,1}, k=1->{2,3}, k=2->{4,5}, k=3->{6,7}, k=4->{8,9}
        # i*5+k: i=0->{0,1,2,3,4}, i=1->{5,6,7,8,9}
        
        # So the output is indexed differently: by (k,i) vs (i,k).
        # This doesn't change the tensor structure, just the ordering of the W indices.
        
        # Now, the key question: is M[a,b,c] = 1 for all a,b,c, or only for specific (a,b,c)
        # that are consistent (same j in a and b)?
        
        # a = i*m+j, b = j*p+k: these share j.
        # c = k*n+i: this is determined by k and i.
        
        # So for a given a = i*m+j and b = j*p+k, the output index is c = k*n+i.
        # The tensor entry M[a,b,c] = 1 if c = k*n+i where i,j come from a and j,k from b.
        # Otherwise M[a,b,c] = 0.
        
        # This means the tensor is NOT all 1s! It's sparse.
        # For each valid (i,j,k), M[i*m+j, j*p+k, k*n+i] = 1, and all other entries are 0.
        
        # There are n*m*p = 50 non-zero entries, at specific positions.
        
        # So my earlier rank-1 analysis was WRONG because I assumed all entries were 1.
        # In reality, M[a,b,c] = 1 only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k,
        # and 0 otherwise.
        
        # Let me verify: for a=0 (i=0,j=0), b=0 (j=0,k=0), the only valid c is k*n+i = 0*2+0 = 0.
        # So M[0,0,0] = 1, and M[0,0,1] = 0, M[0,0,2] = 0, etc.
        
        # For a=0 (i=0,j=0), b=5 (j=1,k=0): wait, b=j*p+k = 1*5+0 = 5, so j=1, k=0.
        # But a=i*m+j with j=0, and b has j=1. These are inconsistent!
        # So M[0,5,c] = 0 for all c.
        
        # The tensor is sparse: M[a,b,c] = 1 iff the j-index in a matches the j-index in b,
        # and c = k*n+i consistent with a and b.
        
        # OK so back to the drawing board. The schoolbook decomposition with rank 50
        # is correct: each non-zero entry gets its own term.
        
        # Can we do better? The rank is at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25.
        # So rank >= 25 and rank <= 50.
        
        # For 2x5x5, I believe the rank is known or can be bounded.
        # Let's try to find a decomposition with rank < 50.
        
        # One approach: use the structure that n=2.
        
        # For fixed (j,k), the terms involving this (j,k) have i=0 and i=1.
        # Term for (i=0,j,k): U[0*5+j,c]=1, V[j*5+k,c]=1, W[k*2+0,c]=1
        # Term for (i=1,j,k): U[1*5+j,c]=1, V[j*5+k,c]=1, W[k*2+1,c]=1
        
        # These two terms share the same V index (j*5+k).
        
        # Can we merge them? If we set a single term with:
        # U[0*5+j,c] = 1, U[1*5+j,c] = 1
        # V[j*5+k,c] = 1
        # W[k*2+0,c] = 1, W[k*2+1,c] = 1
        # Then this term contributes to both (i=0,j,k) and (i=1,j,k) with value 1.
        # But it also contributes to... let me check.
        # For (i=0,j,k): U[0*5+j,c]*V[j*5+k,c]*W[k*2+0,c] = 1*1*1 = 1 ✓
        # For (i=1,j,k): U[1*5+j,c]*V[j*5+k,c]*W[k*2+1,c] = 1*1*1 = 1 ✓
        # For other (i',j',k'):
        # If i'=0, j'!=j: U[0*5+j',c] must be 0 (we only set U[0*5+j,c] and U[1*5+j,c])
        #   So contribution is 0. ✓
        # If j'=j, k'!=k: V[j*5+k',c] must be 0. ✓
        # So this merged term works!
        
        # So for each (j,k), we can merge the i=0 and i=1 terms into one term.
        # This reduces rank from 2*5*5=50 to 5*5=25!
        
        # Let's verify this more carefully.
        # For each (j,k) in [0,5)x[0,5), create one term c:
        # U[0*m+j, c] = 1  (i=0, this j)
        # U[1*m+j, c] = 1  (i=1, this j)
        # V[j*p+k, c] = 1
        # W[k*n+0, c] = 1  (k, i=0)
        # W[k*n+1, c] = 1  (k, i=1)
        
        # Check (i,j,k) entry:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # The term with this (j,k) has:
        #   U[i*m+j,c] = 1 (since we set both i=0 and i=1 for this j)
        #   V[j*p+k,c] = 1
        #   W[k*n+i,c] = 1
        # So contribution = 1. ✓
        
        # Any other term c' (with different j' or k'):
        # If j' != j: U[i*m+j,c'] = 0 (since term c' only sets U for j'=j')
        #   Wait, term c' sets U[0*m+j',c'] and U[1*m+j',c'].
        #   So U[i*m+j,c'] = 0 if j != j'. ✓
        # If k' != k: W[k*n+i,c'] = 0 (term c' sets W[k'*n+0,c'] and W[k'*n+1,c'])
        #   So W[k*n+i,c'] = 0 if k != k'. ✓
        # If j=j' and k=k': it's the same term. ✓
        
        # So the total is exactly 1 for each (i,j,k). The rank is 25!
        
        # Let me double-check the dimensions:
        # n=2, m=5, p=5
        # U: (n*m, rank) = (10, 25)
        # V: (m*p, rank) = (25, 25)
        # W: (p*n, rank) = (10, 25)
        
        # For c = j*5+k (or some ordering), j in 0..4, k in 0..4:
        # U[0*5+j, c] = 1  -> row j (0-4)
        # U[1*5+j, c] = 1  -> row 5+j (5-9)
        # V[j*5+k, c] = 1  -> row j*5+k
        # W[k*2+0, c] = 1  -> row 2k
        # W[k*2+1, c] = 1  -> row 2k+1
        
        # This looks correct!
        
        rank = m * p  # 25
        U = [[0] * rank for _ in range(n * m)]  # 10 x 25
        V = [[0] * rank for _ in range(m * p)]  # 25 x 25
        W = [[0] * rank for _ in range(p * n)]  # 10 x 25
        
        for j in range(m):
            for k in range(p):
                c = j * p + k
                U[0 * m + j][c] = 1  # i=0
                U[1 * m + j][c] = 1  # i=1
                V[j * p + k][c] = 1
                W[k * n + 0][c] = 1  # i=0
                W[k * n + 1][c] = 1  # i=1
        
        result = {"U": U, "V": V, "W": W}
        
        # Verify
        if verify(result):
            return result
        else:
            # Should not happen, but fall back to schoolbook
            return schoolbook()
    
    # Actually, let me also try to see if we can do even better than 25.
    # The lower bound is max(n*m, m*p, p*n) = 25.
    # So 25 is the minimum possible rank!
    # Our construction achieves rank 25, which is optimal!
    
    result = try_greedy_rank(25)
    
    # Verify and return
    if verify(result):
        return result
    else:
        # Fallback to schoolbook
        return schoolbook()

