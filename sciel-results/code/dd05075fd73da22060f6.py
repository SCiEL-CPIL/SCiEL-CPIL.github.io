"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _vec_to_frac_list(vec):
    """Convert a numpy array of floats to a list of Fractions."""
    result = []
    for v in vec:
        # Try to find a simple fraction
        v_float = float(v)
        if abs(v_float) < 1e-12:
            result.append(Fraction(0, 1))
            continue
        # Try denominators up to 100
        found = False
        for den in range(1, 101):
            num = v_float * den
            num_rounded = round(num)
            if abs(num - num_rounded) < 1e-9:
                result.append(Fraction(int(num_rounded), den))
                found = True
                break
        if not found:
            # Use a high-precision fraction
            result.append(Fraction(v_float).limit_denominator(10**6))
    return result

def construct():
    n, m, p = DIMENSIONS
    # Tensor dimensions: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12
    
    rng = np.random.default_rng(42)
    
    # We'll greedily build a basis of rank-1 slices.
    # Each slice is (u, v, w) where u in R^(n*m), v in R^(m*p), w in R^(p*n)
    # The tensor T = sum_c u_c ⊗ v_c ⊗ w_c
    # We need T to equal the matrix multiplication tensor.
    
    # The matrix multiplication tensor T_{i,j,k} = 1 if the entry (i,k) of A*B depends on A[i,j]*B[j,k]
    # Specifically, (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k (it's all-ones in the appropriate sense)
    # Actually, the tensor is: T_{i,j,k} = 1 if we're computing entry (i,k) using A[i,j] and B[j,k]
    # In the standard formulation, the multiplication tensor has entries T_{i,j,k} = δ_{j,j} = 1
    # Wait, let me reconsider. The tensor for matrix multiplication A(n×m) × B(m×p) → C(n×p) is:
    # T: R^{n×m} ⊗ R^{m×p} → R^{n×p}
    # T(A, B) = A × B
    # As a 3-tensor: T_{(i,k), j} ... no.
    
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has indices (i, j, k) with T[i,j,k] = 1
    # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    
    # For a rank-1 decomposition: C[i,k] = sum_c U[i*m+j_c, c] ... no.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The reconstruction is: for each c, we add U[:,c] ⊗ V[:,c] ⊗ W[:,c] to the tensor
    # Then C[i,k] = sum_{j} sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, W is indexed by (k*n+i), so W[k*n+i, c].
    
    # Let me verify: the tensor T[i,j,k] should satisfy:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need T[i,j,k] = 1 for all i,j,k.
    
    # So the condition is: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 48 equations in rank variables.
    
    # Approach: generate random (u, v, w) triples, check if adding them helps satisfy the constraints.
    # Better: use the algebraic approach - the set of all valid rank-1 tensors forms a variety.
    # We want to find a set of rank-1 tensors whose sum equals T.
    
    # Let's use a different approach: 
    # The tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # We want T = sum_c u_c ⊗ v_c ⊗ w_c
    # 
    # Fix two indices and think about it as a matrix.
    # For fixed j: the matrix M_j[i,k] = T[i,j,k] = 1 (all ones, n×p matrix)
    # M_j = sum_c u_c[j] * (v_c * w_c^T) ... no.
    # 
    # Actually, T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is not a simple tensor product because W depends on both k and i.
    
    # Let me re-examine the W indexing. W[k*n+i, c].
    # So for fixed i and k, W[k*n+i, c] is a scalar.
    
    # Let's think of it differently. Define:
    # For each c, define matrices:
    # A_c[i,j] = U[i*m+j, c]  (n×m)
    # B_c[j,k] = V[j*p+k, c]  (m×p)  
    # C_c[i,k] = W[k*n+i, c]  (n×p)
    # 
    # Then the condition is: for all i,k:
    # C[i,k] = sum_c sum_j A_c[i,j] * B_c[j,k] * C_c[i,k]
    # That doesn't look right either.
    
    # Let me re-read the problem. The tensor decomposition means:
    # The multiplication tensor T can be written as sum of rank-1 tensors.
    # A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{p*n}
    # T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # And we need T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    
    # So we need: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    # This is a system of 48 equations. Let's set up the linear system if we fix the structure.
    # But the variables are the entries of u_c, v_c, w_c, which are nonlinear.
    
    # The randomized algebraic approach: 
    # 1. Start with an empty set of slices.
    # 2. Generate random (u, v, w).
    # 3. Check if this slice is "new" by seeing if it adds a new dimension to the span of 
    #    the bilinear maps associated with the slices.
    # 4. Keep adding until the span covers the full tensor space.
    
    # The key insight: the tensor T defines a bilinear map from R^{n*m} × R^{m*p} → R^{p*n}
    # T(u, v) = sum_{i,j,k} u[i*m+j] v[j*p+k] T[i,j,k] e_{k*n+i}
    # Since T[i,j,k] = 1, T(u, v) = sum_{i,j,k} u[i*m+j] v[j*p+k] e_{k*n+i}
    
    # For a rank-1 slice (a, b, c), the bilinear map it induces is:
    # B_{a,b,c}(u, v) = (a·u) (b·v) c
    # This is a rank-1 bilinear map.
    
    # The tensor T has rank r if and only if the bilinear map T can be written as sum of r rank-1 bilinear maps.
    
    # To check if a set of rank-1 slices spans the tensor space, we can check if their induced
    # bilinear maps span the space of bilinear maps that agree with T on a dense set.
    
    # Practical approach: 
    # - Generate many random rank-1 slices.
    # - For each, compute its "signature" as a vector in some representation.
    # - Use Gaussian elimination to find a maximal linearly independent set.
    # - The rank is the size of this set (if it spans the tensor).
    
    # How to represent a rank-1 slice as a vector? 
    # The slice (a, b, c) defines a bilinear map. We can evaluate it on basis vectors.
    # B_{a,b,c}(e_{i*m+j}, e_{j'*p+k}) = a[i*m+j] b[j'*p+k] c[k*n+i] if j=j' else 0
    # Wait, the bilinear map T: R^{n*m} × R^{m*p} → R^{p*n}
    # T(u, v)[k*n+i] = sum_j u[i*m+j] v[j*p+k] T[i,j,k] = sum_j u[i*m+j] v[j*p+k]
    
    # For a rank-1 slice (a, b, c):
    # B(u, v)[k*n+i] = (sum_{i',j} a[i'*m+j] u[i'*m+j]) * (sum_{j',k'} b[j'*p+k'] v[j'*p+k']) * c[k*n+i]
    # That's not right. A rank-1 tensor a⊗b⊗c gives:
    # (a⊗b⊗c)(u, v) = (a·u)(b·v) c
    # So B(u,v) = (a·u)(b·v) c, which is a vector in R^{p*n}.
    
    # The full tensor T gives: T(u,v)[k*n+i] = sum_j u[i*m+j] v[j*p+k]
    
    # We need sum_c (a_c·u)(b_c·v) c_c = T(u,v) for all u,v.
    
    # This is equivalent to: the set of vectors {c_c * (a_c ⊗ b_c)} spans the same space as T in the 
    # tensor product R^{p*n} ⊗ (R^{n*m} ⊗ R^{m*p})^*.
    
    # Simpler: just solve the system numerically.
    # Variables: for each c, a_c (12 entries), b_c (16 entries), c_c (12 entries).
    # Constraints: 48 equations.
    # This is underdetermined for small rank.
    
    # Let's try rank = 12, 16, 20, etc. and see if we can find a solution.
    # Use a greedy randomized approach:
    
    best_rank = None
    best_solution = None
    
    for target_rank in [12, 13, 14, 15, 16, 18, 20, 24, 32, 48]:
        for trial in range(50):
            rng = np.random.default_rng(42 + target_rank * 100 + trial)
            
            # Initialize with random slices
            slices = []
            for c in range(target_rank):
                a = rng.standard_normal(dim_U)
                b = rng.standard_normal(dim_V)
                cc = rng.standard_normal(dim_W)
                # Normalize to avoid numerical issues
                a = a / (np.linalg.norm(a) + 1e-12)
                b = b / (np.linalg.norm(b) + 1e-12)
                cc = cc / (np.linalg.norm(cc) + 1e-12)
                slices.append((a, b, cc))
            
            # Build the linear system. The constraints are nonlinear, so we can't directly use linear algebra.
            # Instead, we can use the fact that if we fix a and b, the constraints become linear in c.
            # But that's still complicated.
            
            # Alternative: use the moment matrix approach.
            # The tensor T has a flattening. Consider the flattening that groups (i,j) vs k:
            # M[i*m+j, k*n+i]... this is messy due to the W indexing.
            
            # Let's just try to verify if a given decomposition works by checking all 48 constraints.
            
            def check_decomposition(slices):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for a, b, cc in slices:
                                total += a[i*m+j] * b[j*p+k] * cc[k*n+i]
                            if abs(total - 1.0) > 1e-6:
                                return False
                return True
            
            if check_decomposition(slices):
                best_rank = target_rank
                best_solution = slices
                break
        if best_rank is not None:
            break
    
    if best_solution is None:
        # Fallback to schoolbook
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
    
    rank = len(best_solution)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (a, b, cc) in enumerate(best_solution):
        for idx in range(dim_U):
            U[idx][c] = _vec_to_frac_list(a)[idx]
        for idx in range(dim_V):
            V[idx][c] = _vec_to_frac_list(b)[idx]
        for idx in range(dim_W):
            W[idx][c] = _vec_to_frac_list(cc)[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
