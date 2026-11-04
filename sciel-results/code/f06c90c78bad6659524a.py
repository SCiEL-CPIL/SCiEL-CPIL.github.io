"""
Strategy: Greedy Exact Cover over a library of overlapping structured decompositions.
Unlike the partition-based approach which restricts to disjoint blocks, this strategy
considers all possible 2x2x2 Strassen blocks and 1x1x1 schoolbook terms as "tools".
We use a greedy set cover algorithm to select a minimal set of these tools such that
their weighted sum exactly equals the matrix multiplication tensor.

This allows overlapping blocks (e.g., a Strassen block on indices (0,1) and another on (1,2))
to be combined, potentially yielding a lower rank than any disjoint partition.
"""
import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

def _get_strassen_terms():
    """
    Returns the 7 rank-1 terms of the standard Strassen decomposition for 2x2x2.
    Each term is a tuple (u_vec, v_vec, w_vec) of length 4, representing the
    coefficients for the 4 basis elements in each dimension.
    The tensor equation is:
    (A00 A01; A10 A11) * (B00 B01; B10 B11) = (C00 C01; C10 C11)
    
    The terms correspond to:
    1. (A00+A11)(B00+B11) -> C00, C11
    2. (A10+A11)B00 -> C10
    3. A00(B10-B11) -> C01
    4. A11(B01-B00) -> C10
    5. (A00+A01)B11 -> C01
    6. (A10-A00)(B00+B01) -> C10
    7. (A01-A11)(B10+B11) -> C01
    
    Let's verify the standard Strassen formulas:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B10 - B11)
    M4 = A11 * (B01 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M6 - M7
    
    We need to express these as rank-1 tensors u_i v_j w_k.
    The tensor is T = sum_{i,j,k} x_i y_j z_k e_i e_j e_k.
    For a 2x2x2 block, the indices are local 0,1.
    
    Term 1: M1
    u = [1, 0, 0, 1] (A00, A11)
    v = [1, 0, 0, 1] (B00, B11)
    w: C00 gets +1, C11 gets +1.
    C indices: (k,i). C00 is k=0,i=0. C11 is k=1,i=1.
    w = [1, 0, 0, 1]
    
    Term 2: M2
    u = [0, 0, 1, 1] (A10, A11)
    v = [1, 0, 0, 0] (B00)
    w: C10 gets +1. k=1, i=0.
    w = [0, 0, 1, 0] -> index k*n+i. n=2. k=1, i=0 -> 1*2+0 = 2.
    w = [0, 0, 1, 0]
    
    Term 3: M3
    u = [1, 0, 0, 0] (A00)
    v = [0, 0, 1, -1] (B10 - B11)
    w: C01 gets +1. k=0, i=1.
    w = [0, 1, 0, 0]
    
    Term 4: M4
    u = [0, 0, 0, 1] (A11)
    v = [-1, 1, 0, 0] (B01 - B00)
    w: C10 gets +1. k=1, i=0.
    w = [0, 0, 1, 0]
    
    Term 5: M5
    u = [1, 1, 0, 0] (A00 + A01)
    v = [0, 0, 0, 1] (B11)
    w: C01 gets +1. k=0, i=1.
    w = [0, 1, 0, 0]
    
    Term 6: M6
    u = [-1, 0, 1, 0] (A10 - A00)
    v = [1, 1, 0, 0] (B00 + B01)
    w: C10 gets +1. k=1, i=0.
    w = [0, 0, 1, 0]
    
    Term 7: M7
    u = [0, 1, 0, -1] (A01 - A11)
    v = [0, 0, 1, 1] (B10 + B11)
    w: C01 gets -1, C11 gets -1.
    C01: k=0, i=1. C11: k=1, i=1.
    w = [0, -1, 0, -1]
    """
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, -1, 0, -1]),
    ]

def _generate_candidate_terms(n, m, p):
    """
    Generate a list of candidate terms.
    Each term is a dict with 'U', 'V', 'W' vectors (sparse or dense, but small enough for dense)
    and a 'mask' indicating which (i,j,k) indices it covers (for efficiency in set cover, 
    though we will use full vector comparison for correctness).
    """
    candidates = []
    strassen_terms = _get_strassen_terms()
    
    # 1. Schoolbook terms (1x1x1)
    # These are rank 1 terms that cover exactly one (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = np.zeros(n * m)
                V = np.zeros(m * p)
                W = np.zeros(p * n)
                U[i * m + j] = 1
                V[j * p + k] = 1
                W[k * n + i] = 1
                candidates.append((U, V, W))
                
    # 2. Strassen terms (2x2x2)
    # For every possible 2x2x2 sub-block
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                for term in strassen_terms:
                    u_coeffs, v_coeffs, w_coeffs = term
                    U = np.zeros(n * m)
                    V = np.zeros(m * p)
                    W = np.zeros(p * n)
                    
                    # Fill U
                    for di in range(2):
                        for dj in range(2):
                            U[(i0 + di) * m + (j0 + dj)] = u_coeffs[di * 2 + dj]
                            
                    # Fill V
                    for dj in range(2):
                        for dk in range(2):
                            V[(j0 + dj) * p + (k0 + dk)] = v_coeffs[dj * 2 + dk]
                            
                    # Fill W
                    for di in range(2):
                        for dk in range(2):
                            W[(k0 + dk) * n + (i0 + di)] = w_coeffs[di * 2 + dk]
                            
                    candidates.append((U, V, W))
                    
    return candidates

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Target tensor is the identity-like structure for matrix multiplication
    # T[i,j,k] = 1 if this is a valid multiplication term?
    # Actually, the tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all i,j,k.
    # Wait, the tensor is defined by the bilinear map.
    # (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is in U* x V* x W*.
    # T(u, v, w) = sum_{i,j,k} u_i v_j w_k T_{ijk}.
    # For matrix multiplication, T_{ijk} = 1 if the indices align?
    # No, the standard tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # So T_{ijk} = 1 for all i,j,k.
    
    # We want to find coefficients c_t such that sum_t c_t U_t \otimes V_t \otimes W_t = T.
    # Since we are using "structured" terms where the entries are integers/half-integers,
    # and the target is integer, we can assume c_t = 1 for all selected terms?
    # Not necessarily. Strassen terms have coefficients like -1.
    # However, in the "Set Cover" formulation, we are summing the tensors directly.
    # If we select a term, we add its full tensor contribution.
    # The question is: can we form the target tensor as a sum of a subset of these candidates?
    # Yes, because the candidates are linearly independent in terms of their "support" in a greedy sense?
    # Actually, no. They are not linearly independent.
    # But if we use a greedy algorithm that picks the term that maximizes the "alignment" with the residual,
    # and the residual becomes zero, we have a solution.
    
    # Let's represent the residual tensor as a vector of length n*m*p.
    # Residual[i, j, k] = 1 initially.
    # When we add a term (U, V, W), we add the tensor U \otimes V \otimes W to the sum.
    # The contribution to index (i,j,k) is U[i,j] * V[j,k] * W[k,i].
    
    # We want sum_{t in S} (U_t \otimes V_t \otimes W_t) = T_target.
    
    # Pre-compute the tensor contribution of each candidate as a flat vector of length n*m*p.
    # This might be memory heavy if we have many candidates, but n*m*p = 96.
    # Number of candidates:
    # Schoolbook: 96
    # Strassen: (3-1)*(4-1)*(8-1)*7 = 2*3*7*7 = 294
    # Total ~ 390 candidates.
    # Each vector is length 96.
    # 390 * 96 floats is small.
    
    target = np.ones(n * m * p)
    
    candidates = _generate_candidate_terms(n, m, p)
    
    # Convert candidates to flat tensor vectors
    candidate_tensors = []
    for U, V, W in candidates:
        # Compute U \otimes V \otimes W flattened
        # Index mapping:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # Target index: i*m*p + j*p + k ? 
        # Let's define a consistent flattening for the tensor T_{ijk}.
        # Let's use idx = i * (m*p) + j * p + k.
        
        tensor_vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j] * V[j * p + k] * W[k * n + i]
                    if val != 0:
                        tensor_vec[i * (m * p) + j * p + k] = val
        candidate_tensors.append(tensor_vec)
    
    # Greedy Set Cover / Residual Reduction
    # We want to select a subset of candidate_tensors such that their sum equals target.
    # Since coefficients are not 1, this is a linear system.
    # However, if we restrict to "exact" matches, we can use a greedy approach:
    # 1. Start with residual = target.
    # 2. Find the candidate that best reduces the residual (e.g., minimizes L2 norm of residual - candidate).
    # 3. Subtract it.
    # 4. Repeat until residual is 0 or no improvement.
    
    # But this is for non-negative coefficients. Here we have negative entries.
    # A better approach for exact integer/half-integer decompositions:
    # Use a greedy algorithm that picks the term with the highest "overlap" with the residual,
    # then solves for the coefficient? No, coefficients must be 1 (we are summing tensors).
    # Wait, the problem asks for a decomposition. It doesn't say coefficients must be 1.
    # "integer/half-integer entries" refers to U, V, W.
    # The decomposition is sum_t U_t \otimes V_t \otimes W_t.
    # This implies the coefficient for each term in the sum is 1.
    # So we are looking for a subset S such that sum_{t in S} T_t = T_target.
    
    # This is a Subset Sum problem in a high-dimensional space.
    # Greedy:
    residual = target.copy()
    selected_indices = []
    remaining_candidates = list(range(len(candidate_tensors)))
    
    # To make greedy work, we should probably sort candidates by "magnitude" or "density"
    # or just iterate.
    
    # A more robust greedy:
    # At each step, pick the candidate c that minimizes ||residual - c||_2.
    # If min ||residual - c|| < ||residual||, subtract c and add to selected.
    # Else, try adding c? No, we are summing.
    # If we can't reduce the norm by subtracting, maybe we need to add?
    # But target is positive. Candidates can be negative.
    
    # Let's try a different heuristic:
    # We know the schoolbook terms form a basis.
    # We can start with the schoolbook decomposition (rank 96) and try to "merge" terms.
    # Or, use the greedy approach on the residual.
    
    # Let's try the greedy reduction.
    # To avoid getting stuck, we can use a "best fit" selection.
    
    # Optimization: Pre-compute norms?
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline:
            break
        if np.allclose(residual, 0, atol=1e-9):
            break
        if not remaining_candidates:
            break
            
        best_idx = None
        best_norm = np.linalg.norm(residual)
        
        # Heuristic: Only consider candidates that have non-zero overlap with residual
        # Or just iterate through all remaining.
        # 390 candidates * 96 dims is fast.
        
        for idx in remaining_candidates:
            # Try subtracting
            new_res = residual - candidate_tensors[idx]
            norm = np.linalg.norm(new_res)
            if norm < best_norm - 1e-9:
                best_norm = norm
                best_idx = idx
            # Also try adding? If residual is positive, adding a positive term makes it worse.
            # But if candidate has negative parts, adding might help?
            # Let's stick to subtraction for now, as target is positive and schoolbook terms are positive.
            # Strassen terms have mixed signs.
            # If we subtract a term with negative parts, it adds to the residual in those spots.
            # This is correct behavior for set cover.
            
        if best_idx is not None:
            residual = residual - candidate_tensors[best_idx]
            selected_indices.append(best_idx)
            remaining_candidates.remove(best_idx)
        else:
            # No single term reduces the norm.
            # This suggests we need a combination or we are stuck.
            # For exact cover, if we can't reduce, we might need to backtrack or use a different strategy.
            # Let's break and see what we have.
            break
            
    # Verify the solution
    if np.allclose(residual, 0, atol=1e-9):
        # We have a valid decomposition!
        # Construct U, V, W
        rank = len(selected_indices)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for col, idx in enumerate(selected_indices):
            U[:, col] = candidates[idx][0]
            V[:, col] = candidates[idx][1]
            W[:, col] = candidates[idx][2]
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
