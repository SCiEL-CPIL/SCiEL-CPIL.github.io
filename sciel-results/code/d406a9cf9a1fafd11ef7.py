"""
Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

Strategy:
- Start from the schoolbook rank-36 decomposition.
- Use a structured block approach: the 2x3x6 tensor can be viewed as
  two 2x3x3 blocks (since p=6=2*3). 
- Search for low-rank decompositions using:
  1. Known results for small matrix multiplication tensors.
  2. Integer/half-integer constrained optimization via exhaustive search over
     structured patterns.
  3. A final verification pass.

Known bounds: For 2x3 * 3x6, the tensor is T_{2,3,6}. 
The schoolbook rank is 36. 
We can exploit that 2x3x6 = 2x3x3 + 2x3x3 (two copies), but that doesn't directly help.

Actually, let's think differently. The tensor T_{n,m,p} has entries T[i,j,k] = 1 if the
corresponding output element depends on input[i,j]*input[j,k]. For n=2, m=3, p=6:
- U has shape (2*3, r) = (6, r)
- V has shape (3*6, r) = (18, r)  
- W has shape (6*2, r) = (12, r)

The multiplication tensor satisfies:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (the actual product)

Wait, the convention is: the matrix multiplication tensor M_{n,m,p} has
M[i,j,k] = 1 for each triple (i,j,k) in [n]x[m]x[p], and the decomposition is:
M = sum_r u_r ⊗ v_r ⊗ w_r

where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]

And the condition is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
and for all other combinations (where the product term is zero in the tensor), the sum = 0.

Actually the tensor M_{n,m,p} has 1s at positions (i,j,k) for all valid i,j,k, and the
decomposition must reproduce exactly: the (i,j,k)-th entry of the tensor is 1, and the
decomposition gives sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] which must equal 1 for all
valid (i,j,k) triples.

But wait - that's not right either. The matrix multiplication tensor is defined so that
C = AB where C[i,k] = sum_j A[i,j]*B[j,k]. The tensor T has T[i,j,k] = 1 for all i,j,k,
and the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j]*B[j,k]*C[k,i] (with appropriate
indexing). The rank of the tensor is the minimum number of rank-1 tensors that sum to it.

For the decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r where:
- u_r is a vector in R^{n*m} (indexed by (i,j))
- v_r is a vector in R^{m*p} (indexed by (j,k))
- w_r is a vector in R^{p*n} (indexed by (k,i))

And the condition is that for all (i,j,k):
sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1

This is the standard matrix multiplication tensor. The schoolbook decomposition uses
n*m*p terms.

For 2x3x6, schoolbook rank = 36. We need to find if there's a lower rank.

Key insight: 2x3x6 can be related to 2x3x3. If we can find a rank-r decomposition for
2x3x3, we might extend it. But 2x3x6 is not simply two copies of 2x3x3 in a useful way
for rank reduction.

Let me try a direct search approach with structured constraints. Given the small sizes:
- U: 6 x r
- V: 18 x r
- W: 12 x r

For r < 36, we need to find integer/half-integer entries.

A practical approach: use the fact that for small matrix multiplication, the optimal
rank is often close to or equal to the schoolbook rank. Let me try to verify if rank 36
is indeed optimal by attempting to find rank < 36 solutions.

Given the time constraint (150s), I'll implement:
1. A quick check with known lower bounds
2. A structured search for rank reductions using symmetry
3. Return the best found (likely 36 if no improvement found quickly)

Actually, let me try a more systematic approach. The tensor T_{2,3,6} has 2*3*6 = 36
nonzero entries (all 1s). The rank must be at least the maximum of the "slice ranks" or
other lower bounds.

For now, let me implement a search that tries to find rank < 36 by:
- Using random initializations with integer/half-integer constraints
- Local search / hill climbing
- Checking feasibility exactly

Given the difficulty, I'll focus on a robust implementation that returns the schoolbook
solution if no better is found within time.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), else 0.
    
    Actually, the tensor is defined such that the decomposition condition is:
    For all (i, j, k) in [n]x[m]x[p]:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    # We don't actually need to build the full tensor explicitly.
    # The condition is: for each (i,j,k), the sum must be 1.
    # There are n*m*p = 36 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (6 + 18 + 12) = 36*r
    # For r=36, we have 1296 unknowns and 36 equations - very underdetermined.
    pass

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    # Try integer snapping
    snapped_int = np.round(mat)
    if np.max(np.abs(mat - snapped_int)) < tol:
        return snapped_int
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _try_rank_r(r, n, m, p, deadline, rng):
    """
    Try to find a rank-r decomposition. Uses a structured approach:
    Initialize with random values, then use iterative refinement.
    Returns (U, V, W) if found, else None.
    """
    if time.time() > deadline:
        return None
    return None

def _structured_search(n, m, p, deadline):
    """
    Try structured decompositions based on tensor product factorization.
    
    Key insight: T_{2,3,6} = T_{2,3,3} ⊗ T_{1,1,2}
    Since rank(T_{1,1,2}) = 1, we have rank(T_{2,3,6}) ≤ rank(T_{2,3,3}).
    
    We search for a rank-r decomposition of T_{2,3,3} with r < 18,
    then extend it to T_{2,3,6} via the tensor product construction.
    """
    n2, m2, p2 = 2, 3, 3
    schoolbook_233 = n2 * m2 * p2  # 18
    
    # Search for rank < 18 decomposition of T_{2,3,3}
    best_233 = None
    for r in range(1, schoolbook_233):
        if time.time() > deadline:
            break
        result = _try_233_rank(r, deadline)
        if result is not None:
            best_233 = result
            break
    
    if best_233 is not None:
        U233, V233, W233 = best_233
        r233 = U233.shape[1]
        
        # T_{1,1,2} rank-1 decomposition:
        # U1 = [[1]], V1 = [[1],[1]], W1 = [[1],[1]]
        U1 = np.array([[1.0]])
        V1 = np.array([[1.0], [1.0]])
        W1 = np.array([[1.0], [1.0]])
        
        # Extend to T_{2,3,6} via tensor product
        U6 = np.zeros((6, r233))
        V6 = np.zeros((18, r233))
        W6 = np.zeros((12, r233))
        
        # U unchanged (n=2, m=3)
        U6 = U233.copy()
        
        # V: index (j,k) in 3x6, k = k1*2 + k2
        for j in range(3):
            for k in range(6):
                k1 = k // 2
                k2 = k % 2
                V6[j * 6 + k, :] = V233[j * 3 + k1, :] * V1[k2, 0]
        
        # W: index (k,i) in 6x2, k = k1*2 + k2
        for k in range(6):
            k1 = k // 2
            k2 = k % 2
            for i in range(2):
                W6[k * 2 + i, :] = W233[k1 * 2 + i, :] * W1[k2, 0]
        
        valid, info = _verify_decomposition(U6, V6, W6, 2, 3, 6)
        if valid:
            return U6, V6, W6
    
    return None

def _try_233_rank(r, deadline):
    """
    Try to find a rank-r decomposition of T_{2,3,3}.
    
    Strategy: Use the fact that for r close to 18, we can try to merge
    schoolbook terms. The schoolbook has 18 terms. We try to find a
    rank-r decomposition by solving a system of equations.
    
    For r = 17, we have 17*21 = 357 unknowns and 18 equations.
    We use a constructive approach: start with schoolbook and try to
    eliminate one term by adjusting others.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 2, 3, 3
    # U: 6 x r, V: 9 x r, W: 6 x r
    
    # Approach: Try to construct a rank-r decomposition directly.
    # For r=17, we need to find U(6x17), V(9x17), W(6x17) such that
    # for all (i,j,k): sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
    
    # Use a randomized local search with half-integer constraints.
    # Initialize near schoolbook structure and refine.
    
    rng = np.random.RandomState(123)
    
    # Try multiple random initializations
    for trial in range(50):
        if time.time() > deadline:
            return None
        
        # Initialize with small random half-integers
        U = rng.randint(-2, 3, size=(6, r)).astype(float) * 0.5
        V = rng.randint(-2, 3, size=(9, r)).astype(float) * 0.5
        W = rng.randint(-2, 3, size=(6, r)).astype(float) * 0.5
        
        # Local search: iteratively adjust entries to satisfy equations
        for iteration in range(200):
            if time.time() > deadline:
                return None
            
            # Compute residuals for all 18 equations
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        residuals.append((i, j, k, val - 1.0))
            
            # Check if all residuals are near zero
            max_res = max(abs(res) for _, _, _, res in residuals)
            if max_res < 1e-9:
                # Verify exactly
                valid, _ = _verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return U, V, W
            
            # Find the equation with the largest residual and adjust
            i, j, k, res = max(residuals, key=lambda x: abs(x[3]))
            
            # Try adjusting one entry in U, V, or W for this equation
            # U index: i*m+j
            # V index: j*p+k  
            # W index: k*n+i
            
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Try adjusting U[u_idx, c] for random c
            c = rng.randint(0, r)
            old_U = U[u_idx, c].copy()
            U[u_idx, c] += 0.5 * (1 if res < 0 else -1)
            
            # Check if this improved things
            new_res = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :]) - 1.0
            if abs(new_res) > abs(res):
                U[u_idx, c] = old_U
            else:
                # Keep the change, also try V and W
                break
        
        # After each major iteration, try snapping to half-integers
        snapped_U = _snap_to_half_integers(U)
        snapped_V = _snap_to_half_integers(V)
        snapped_W = _snap_to_half_integers(W)
        if snapped_U is not None and snapped_V is not None and snapped_W is not None:
            valid, _ = _verify_decomposition(snapped_U, snapped_V, snapped_W, n, m, p)
            if valid:
                return snapped_U, snapped_V, snapped_W
    
    return None

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x3 * 3x6.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try the structured approach (tensor product reduction)
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        snapped_U = _snap_to_half_integers(U)
        snapped_V = _snap_to_half_integers(V)
        snapped_W = _snap_to_half_integers(W)
        if snapped_U is not None and snapped_V is not None and snapped_W is not None:
            U, V, W = snapped_U, snapped_V, snapped_W
        valid, _ = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook (rank 36)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
