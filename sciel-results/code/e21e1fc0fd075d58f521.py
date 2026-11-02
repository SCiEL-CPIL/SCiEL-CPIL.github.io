"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    The constraint is:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1  for all i,j,k
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _try_landsberg_like(n, m, p, max_rank, timeout=30):
    """
    Try to find a decomposition with rank < n*m*p using structured patterns.
    For 3x3 x 3x8, the schoolbook rank is 72.
    
    One approach: exploit the structure. Since n=3, m=3, p=8,
    we can try to factor the m dimension.
    
    Actually, a key observation: if we can write the m=3 dimension as a
    combination, we might reduce rank. But for exact integer/half-integer
    decompositions, the structure is quite rigid.
    
    Let's try a direct approach: parameterize U, V, W with rank r and solve.
    This is a system of nonlinear equations, so we use a structured ansatz.
    
    Ansatz: U, V, W have entries in {0, 1/2, 1}.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Also, rank >= n*m*p / max possible overlap... 
    # A known lower bound: rank >= (n*m*p) / (max degree), but that's loose.
    # For 3x3x8: lower bound is at least 24 (from m*p = p*n = 24).
    
    # Try a structured ansatz based on Kronecker product ideas
    # Since 8 = 3*2 + 2, maybe we can exploit that 8 = 2^3
    
    # Let's try: factor the p=8 dimension.
    # If we can decompose the 8 columns into groups, we might save rank.
    
    # Alternative: use the fact that 3x3 matrix multiplication has rank 23 (Landsberg)
    # and then tensor with 1x8 (rank 8) gives 23*8=184, which is worse than 72.
    # Or 3x3 (rank 23) * 1x1 for each of 8: still 23*8.
    
    # Better: think of it as (3x3 x 3x1) tensor (1x1 x 1x8)? No.
    # 3x3 x 3x8 = (3x3 x 3x1) * 8 copies? That's rank 23*8 = 184.
    
    # What about: 3x3 x 3x8. Think of B as 3x8 = (3x4) * 2 horizontally?
    # C = A * B where A is 3x3, B is 3x8.
    # C[:, 0:4] = A * B[:, 0:4]  (rank <= rank(3x3x3) * ... no)
    # C[:, 4:8] = A * B[:, 4:8]
    # Total rank <= rank(3x3x3) + rank(3x3x3) = 2*23 = 46? No, that's not how it works.
    # The rank of the concatenation tensor is at most the sum, so <= 23+23=46.
    # But wait, 3x3 x 3x4: the rank of multiplying 3x3 by 3x4.
    # Schoolbook for 3x3x4: 3*3*4=36. Can we do better?
    # 3x3x3 has rank 23, 3x3x1 has rank 9 (schoolbook, and 9 is optimal for 1-column).
    # 3x3x4: schoolbook 36. Lower bound max(9,12,12)=12.
    # Actually for 3x3xn, the rank is known to be 9n for n=1, and for larger n
    # it's at most 9n but can be less. For 3x3x4, is it less than 36?
    
    # Let me just try a numerical search with a structured ansatz.
    
    # Approach: use the fact that for 3x3 x 3x8, we can view it as
    # computing 8 outputs C[i,k] = sum_j A[i,j]*B[j,k].
    # For each k, we compute C[:,k] = A * B[:,k].
    # The tensor for all k is the 8-fold "stack" of 3x3x1 tensors.
    # If we use a shared decomposition for the A part...
    
    # Key idea: U depends on (i,j), V on (j,k), W on (k,i).
    # If we fix the pattern for a "base" 3x3x1 multiplication (rank 9 schoolbook),
    # and then for each k we reuse the same U structure but with V_k and W_k,
    # we get rank 9*8 = 72 (same as schoolbook).
    
    # To save rank, we need the U rows for different k to overlap.
    # I.e., U[i*m+j, c] should be non-zero for multiple k values sharing the same c.
    # But U doesn't depend on k! U is indexed by (i,j) only.
    # So U[i*m+j, c] is the same for all k.
    # V[j*p+k, c] depends on k.
    # W[k*n+i, c] depends on k.
    
    # So the constraint is: for each (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "wrong" combinations (which don't exist since all (i,j,k) are valid),
    # there are no additional constraints.
    
    # Wait, I need to be more careful. The tensor T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition must satisfy:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    # 
    # There are NO constraints for "invalid" (i,j,k) because all triples are valid.
    # So we just need: for all i in [0,3), j in [0,3), k in [0,8):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # That's 3*3*8 = 72 equations in r*(9+24+24) = 57r unknowns (with r=rank).
    # Plus the constraint that entries are integers or half-integers.
    #
    # For r < 72, we have more unknowns than equations potentially, so solutions may exist.
    
    # Let's try a greedy/structured approach:
    # Start with schoolbook and try to merge columns.
    
    # Actually, let me try a simpler structured ansatz:
    # Assume U has only 0s and 1s, V has only 0s and 1s, W has only 0s and 1s.
    # Then the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] is 1 only if all three are 1.
    # So we need: for each (i,j,k), the number of c where all three are 1 is exactly 1.
    # This is equivalent to finding a 1-design or combinatorial structure.
    
    # This is a 3-partite 3-uniform hypergraph problem:
    # Parts: I = {(i,j) : i in [0,3), j in [0,3)} (9 vertices)
    #         J = {(j,k) : j in [0,3), k in [0,8)} (24 vertices)  
    #         K = {(k,i) : k in [0,8), i in [0,3)} (24 vertices)
    # Edges: for each c, we have a hyperedge connecting one vertex from each part.
    # Constraint: for each (i,j,k), exactly one hyperedge contains 
    #   the vertex (i,j) from I, the vertex (j,k) from J, and the vertex (k,i) from K.
    
    # This is equivalent to finding a 3D permutation matrix (or Latin square generalization).
    # For the schoolbook, each edge is a single (i,j,k) triple, so we need 72 edges.
    
    # To reduce rank, we need to find edges that cover multiple (i,j,k) triples.
    # But a single edge (u, v, w) where u=(i,j), v=(j,k), w=(k,i) only covers
    # the single triple (i,j,k) if i,j,k are consistent. If they're not consistent,
    # it covers no valid triple.
    
    # Wait, that's not right. Let me reconsider.
    
    # U[i*m+j, c] = 1 means the c-th column has a "1" at position (i,j).
    # V[j*p+k, c] = 1 means the c-th column has a "1" at position (j,k).
    # W[k*n+i, c] = 1 means the c-th column has a "1" at position (k,i).
    
    # The product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if and only if
    # all three are 1. This happens when the c-th column has 1s at positions
    # (i,j) in U, (j,k) in V, and (k,i) in W.
    
    # So for a fixed c, the set of (i,j,k) triples covered is:
    #   {(i,j,k) : U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1}
    
    # For the schoolbook, each c covers exactly one triple.
    
    # To reduce rank, we need to find columns that cover multiple triples.
    # But the constraint is that the total coverage is exactly 1 for each triple.
    
    # This is a set cover problem with exact coverage constraint.
    
    # Let me try a different approach: use the fact that the problem is small enough
    # to try a numerical optimization.
    
    # For now, return None to use schoolbook.
    return None


def _local_search(n, m, p, deadline):
    """Local search to find lower-rank decomposition."""
    rank_sb = n * m * p
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing columns that are redundant
    # A column c is redundant if removing it and adjusting others still works.
    # This is hard to check directly. Instead, try a different approach.
    
    # Approach: use the fact that for 3x3x8, we can potentially use
    # a decomposition based on the structure of the 3x3 part.
    
    # Known result: 3x3x3 has rank 23 (Landsberg 1993).
    # For 3x3x8, we can think of it as 3x3x3 * (1x1x8) in some sense,
    # but that gives rank 23*8=184 which is worse.
    
    # Better: view 3x3x8 as computing 8 outputs, each a 3x3x1 multiplication.
    # The 3x3x1 tensor has rank 9 (schoolbook is optimal for this case).
    # But we can share structure across the 8 outputs.
    
    # Let me try a constructive approach:
    # U[i*m+j, c] depends only on (i,j), so it's the same for all k.
    # V[j*p+k, c] and W[k*n+i, c] depend on k.
    
    # For a fixed c, let u_c[i,j] = U[i*m+j, c], v_c[j,k] = V[j*p+k, c], w_c[k,i] = W[k*n+i, c].
    # The constraint is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    # This is a 3-tensor decomposition problem. Let me try to find a low-rank
    # solution using a structured ansatz.
    
    # Ansatz: use rank-3 blocks for the i dimension.
    # Since n=3, we can use a 3x3x3 decomposition (rank 23) and then
    # extend to 8 columns.
    
    # Actually, let me just try a numerical optimization approach.
    # Set up the problem as minimizing ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
    # subject to the entries being in {0, 1/2, 1}.
    
    # For now, return None to use schoolbook.
    return None


def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 72
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by merging columns
    # Strategy: for each pair of columns, check if they can be merged
    # (i.e., their product patterns don't conflict)
    
    # First, try a structured approach:
    # The key constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # With binary U,V,W, each column c "covers" a set of (i,j,k) triples.
    # We need each triple to be covered exactly once.
    
    # Try greedy column merging with backtracking
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank_sb
    
    # Precompute the "signature" of each column: which (i,j,k) it covers
    def get_signature(U_mat, V_mat, W_mat, col, n, m, p):
        sig = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if U_mat[i*m+j, col] * V_mat[j*p+k, col] * W_mat[k*n+i, col] != 0:
                        sig.add((i, j, k))
        return sig
    
    # Try to merge columns greedily
    merged = True
    while merged and time.time() < deadline - 5:
        merged = False
        for c1 in range(current_rank):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, current_rank):
                # Check if columns c1 and c2 can be merged
                # They can be merged if for every (i,j,k), at most one of them
                # has all three entries non-zero
                sig1 = get_signature(current_U, current_V, current_W, c1, n, m, p)
                sig2 = get_signature(current_U, current_V, current_W, c2, n, m, p)
                if not sig1.isdisjoint(sig2):
                    # They overlap, can't simply merge
                    continue
                # Try merging: new column = element-wise product? No.
                # Actually, we need: for each (i,j,k), the sum stays 1.
                # If sig1 and sig2 are disjoint, we can't merge them into one
                # column because a single column would need to cover both sets,
                # but the product U*V*W for a single column is either 0 or 1
                # for each (i,j,k), so it can only cover a "box" structure.
                # 
                # Alternative: try to find a new column that covers sig1 ∪ sig2
                # This requires U[i*m+j,new]=1 for all (i,j) in sig1∪sig2's i,j pairs,
                # V[j*p+k,new]=1 for all (j,k) in sig1∪sig2's j,k pairs,
                # W[k*n+i,new]=1 for all (k,i) in sig1∪sig2's k,i pairs,
                # AND for all (i,j,k) NOT in sig1∪sig2, at least one of U,V,W is 0.
                
                # This is complex. Let me try a different approach:
                # Use a local search to minimize rank directly.
                break
    
    # Try a more direct approach: parameterized search
    # For small ranks, try to find solutions with entries in {0, 1/2, 1}
    
    # Lower bound: rank >= max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound: for fixed (i,j), the equation
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
    # means that the vector (V[j*p+k,c]*W[k*n+i,c])_k dotted with U[i*m+j,:] = 1 for all i
    # This gives constraints on the rank.
    
    # Let me try a numerical approach: set up the equations and solve
    # For rank r, we have r*(n*m + m*p + p*n) variables and n*m*p equations.
    # For r=53: 53*(9+24+24)=2808 variables, 72 equations. Very underdetermined.
    # But we need integer/half-integer solutions.
    
    # Try a randomized local search starting from schoolbook
    best_solution = _local_search(n, m, p, deadline)
    if best_solution is not None:
        return best_solution
    
    # Return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
