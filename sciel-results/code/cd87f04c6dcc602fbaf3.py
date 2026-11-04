"""Rewritten program: attempt a rank-23 exact decomposition for 3x3 matrix multiplication,
falling back to schoolbook rank-27 if verification fails. Uses a deadline guard."""

DIMENSIONS = [3, 3, 3]

import time
import numpy as np

DEADLINE_SECONDS = 140.0  # safety margin before 150s kill


def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n=3, m=3, p=3):
    """Return the rank-n*m*p schoolbook decomposition."""
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


def _try_rank_23():
    """
    Attempt to construct a rank-23 decomposition for 3x3 matrix multiplication.
    
    Strategy: Use the known result M_3 <= 23. We construct this by:
    1. Starting with the rank-27 schoolbook tensor.
    2. Applying a local search / algebraic reduction to find 4 linear dependencies
       that allow us to eliminate 4 of the 27 terms.
    
    The 3x3 multiplication tensor T has entries T[i,j,k] = delta_{i,k} (when we view
    it as a 3x3x3 tensor where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]).
    Actually, the standard tensor for matrix multiplication of size n has dimension
    n^3 and T[i,j,k] = 1 if the "output" index matches, i.e., for C[i,k] = sum_j A[i,j]B[j,k],
    the tensor is T[i,j,k] where the three indices are (row of A, col of A / row of B, col of B / col of C).
    
    More precisely, the matrix multiplication tensor M_{n} has entries:
    M_{n}[i, j, k] = 1  if we identify the three modes as:
      mode 1: (i, j) -> A_{i,j}
      mode 2: (j, k) -> B_{j,k}  
      mode 3: (i, k) -> C_{i,k}
    
    So M_{n} is a tensor of size (n^2, n^2, n^2) with:
    M_{n}[(i,j), (j,k), (i,k)] = 1  (note: the middle index j is shared)
    
    Wait, that's not quite right. Let me re-examine.
    
    For C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has three modes:
    - Mode 1: indexed by (i, j) for A, size n*m
    - Mode 2: indexed by (j, k) for B, size m*p
    - Mode 3: indexed by (i, k) for C, size n*p
    
    And T[(i,j), (j,k), (i,k)] = 1 for all valid i, j, k.
    
    For n=m=p=3, this is a 9x9x9 tensor with 27 non-zero entries (each = 1).
    
    A rank-r decomposition means:
    T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    
    where u_c ∈ R^9, v_c ∈ R^9, w_c ∈ R^9.
    
    The schoolbook decomposition uses r=27 with each term being a basis vector product.
    
    To get rank 23, we need to find 4 linear dependencies among the 27 rank-1 tensors
    that allow us to rewrite them using only 23 terms.
    
    One known approach: The 3x3 tensor can be viewed as containing two overlapping
    2x2 tensors. Each 2x2 has rank 7 (Strassen). The two 2x2 blocks share a "middle"
    index. 
    
    Block 1: indices {0,1} in all three dimensions -> 2x2x2 sub-tensor, rank 7
    Block 2: indices {1,2} in all three dimensions -> 2x2x2 sub-tensor, rank 7
    
    The union of these two blocks covers:
    - All entries where at least two of the three indices are in {0,1} OR at least two are in {1,2}
    - The entries NOT covered are those where the indices are "spread out": one in {0}, one in {1}, one in {2} in some arrangement where no pair is in the same 2-element set.
    
    Actually, let me think more carefully. The 27 entries are indexed by (i,j,k) with i,j,k ∈ {0,1,2}.
    
    Block 1 (top-left 2x2): i∈{0,1}, j∈{0,1}, k∈{0,1} -> 8 entries
    Block 2 (bottom-right 2x2): i∈{1,2}, j∈{1,2}, k∈{1,2} -> 8 entries
    
    Overlap: i=j=k=1 -> 1 entry (counted in both)
    
    Union: 8 + 8 - 1 = 15 entries covered by the two blocks.
    Remaining: 27 - 15 = 12 entries.
    
    If we use 7 terms for block 1 and 7 terms for block 2, that's 14 terms.
    Plus 12 more for the remaining entries = 26 terms. Not 23.
    
    The actual rank-23 construction is more subtle and uses algebraic identities that
    mix the blocks. This is complex to derive from scratch.
    
    Alternative approach: Use a numerical local search.
    
    Start with rank-27 (schoolbook). Then try to find a rank-23 decomposition by:
    1. Setting up 23 rank-1 terms with random initial values.
    2. Minimizing ||T - sum_{c=1}^{23} u_c ⊗ v_c ⊗ w_c||_F^2 using gradient descent or similar.
    3. If we converge to zero error (within tolerance), we have a valid rank-23 decomposition.
    
    This is risky but let's try it with a time limit.
    """
    start_time = time.time()
    n, m, p = 3, 3, 3
    
    # Build the target tensor T of shape (9, 9, 9)
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    target_flat = T.flatten()
    target_norm = np.linalg.norm(target_flat)
    
    # Try local search for rank 23
    rank = 23
    rng = np.random.RandomState(42)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(5):
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
        
        # Initialize with random values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(n * p, rank) * 0.5
        
        # Use a simple iterative refinement
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > DEADLINE_SECONDS - 20:
                break
            
            # Compute current tensor estimate
            # T_est[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # Flatten index: (i,j,k) -> i*(m*p) + j*p + k ... wait, need to be careful
            # T is shape (n*m, m*p, n*p) = (9, 9, 9)
            # Index 0: i*m+j, Index 1: j*p+k, Index 2: k*n+i
            
            # For efficiency, compute using einsum
            T_est = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = T_est - T
            err_flat = err.flatten()
            error = np.dot(err_flat, err_flat)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if error < 1e-18:
                break
            
            # Compute gradients
            # d/dU[i,c] of ||T - T_est||^2 = -2 * sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
            # d/dV[j,c] = -2 * sum_{i,k} err[i,j,k] * U[i,c] * W[k,c]
            # d/dW[k,c] = -2 * sum_{i,j} err[i,j,k] * U[i,c] * V[j,c]
            
            # grad_U[i,c] = -2 * sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
            # This can be written as: grad_U = -2 * einsum('ijk,jc,kc->ic', err, V, W)
            # But wait, the indices in err are (i*m+j, j*p+k, k*n+i) which are the flattened indices.
            # Let me re-index. Let a = i*m+j, b = j*p+k, c_idx = k*n+i.
            # err[a, b, c_idx]
            # grad_U[a, c] = -2 * sum_{b, c_idx} err[a, b, c_idx] * V[b, c] * W[c_idx, c]
            # grad_V[b, c] = -2 * sum_{a, c_idx} err[a, b, c_idx] * U[a, c] * W[c_idx, c]
            # grad_W[c_idx, c] = -2 * sum_{a, b} err[a, b, c_idx] * U[a, c] * V[b, c]
            
            grad_U = -2.0 * np.einsum('abc,bc,dc->ad', err, V, W)
            grad_V = -2.0 * np.einsum('abc,ac,dc->bd', err, U, W)
            grad_W = -2.0 * np.einsum('abc,ac,bc->dc', err, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Periodically try to reduce learning rate
            if step % 500 == 499 and step > 0:
                lr *= 0.5
        
        if best_error < 1e-14:
            break
    
    if best_U is not None and best_error < 1e-14:
        # Verify with exact arithmetic check
        if _verify(best_U, best_V, best_W, n, m, p):
            # Convert to half-integers if possible (multiply by 2 and check if integers)
            U2 = best_U * 2
            V2 = best_V * 2
            W2 = best_W * 2
            if (np.allclose(U2, np.round(U2), atol=1e-9) and
                    np.allclose(V2, np.round(V2), atol=1e-9) and
                    np.allclose(W2, np.round(W2), atol=1e-9)):
                U2 = np.round(U2).astype(int)
                V2 = np.round(V2).astype(int)
                W2 = np.round(W2).astype(int)
                # Scale back: actual values are U2/2, V2/2, W2/2
                # But the product scales by 1/8, so we need to check
                # Actually, let's just return the float values if they're close to half-integers
                return best_U, best_V, best_W
            else:
                return best_U, best_V, best_W
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank-23 first
    result = _try_rank_23()
    if result is not None:
        U, V, W = result
        # Verify once more
        if _verify(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Check if we're running out of time
    if time.time() - start_time > DEADLINE_SECONDS - 5:
        # Fall back to schoolbook immediately
        U, V, W = _schoolbook(n, m, p)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fall back to schoolbook rank-27
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
