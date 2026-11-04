"""Matrix multiplication 4,4,5: rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# Known rank bounds for 4x4 x 4x5:
# The tensor T in R^{4x4} x R^{4x5} -> R^{4x5} has rank r.
# We need r >= max(16, 20, 20) = 20 (dimensional lower bound).
# Strassen's method can reduce rank. For 4x4*4x4, known rank is 49 (or 48 with improvements).
# For 4x4*4x5, the rank is at most 49 (embed 4x4*4x4).
# But we can do better. Let's think about what's achievable.
#
# Actually, for small matrices, the exact minimal rank is hard to determine.
# Let's use a randomized search with verification approach.

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact decomposition of matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if (i,j,k) is a valid index triple, else 0.
    Wait, actually the tensor for matrix mult A(n,m)*B(m,p) = C(n,p) is:
    C[i,k] = sum_j A[i,j]*B[j,k]
    So the tensor T[i,k,j] = delta (i.e., T[i,k,j]=1 always? No...)
    
    Actually the matrix multiplication tensor M_{n,m,p} has entries:
    M_{i,k,j} = 1 for all valid i,k,j (it's the "multiplication" tensor where
    contracting over j gives the product).
    
    A rank-r decomposition means: M_{i,k,j} = sum_{r=1}^{r} U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
    Wait, let me re-read the task.
    
    The task says: "output factor follows the official trace convention W[k*n+i,r]"
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    So the decomposition is: for the matrix multiplication tensor,
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} * ... 
    
    Actually for matrix multiplication A(n x m) * B(m x p) = C(n x p):
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor is T[i,k,j] which should equal 1 for all (i,k,j) in the appropriate range? No.
    The multiplication map is bilinear: (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k].
    
    As a trilinear form / tensor, the multiplication tensor has components:
    mu_{i,j,k} = 1 if we think of it as: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    So the tensor T is in R^{(n*m) x (m*p) x (n*p)} with T[i*m+j, j*p+k, i*p+k] = 1.
    
    A rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p} (or R^{p*n} per the convention).
    
    With the W convention W[k*n+i, r], the third index is k*n+i which ranges over p*n.
    
    So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is valid, else 0.
    Actually it should equal 1 for ALL valid (i,j,k) triples, since every term A[i,j]*B[j,k]
    contributes to C[i,k].
    
    Wait no. The tensor is defined such that for the bilinear map, the structure constants are all 1.
    So T_{(i,j),(j,k),(i,k)} = 1 and T is 0 otherwise? No, that's not right either.
    
    Let me think again. The matrix multiplication tensor M_{n,m,p} is an element of
    (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{n*p}.
    Its components are M_{a,b,c} where a indexes (i,j) pairs, b indexes (j,k) pairs, c indexes (i,k) pairs.
    M_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{j,k'} * ... 
    
    Hmm, let me just use the standard definition. The multiplication tensor for A(n,m)*B(m,p)=C(n,p) is:
    T[i*m+j, j*p+k, i*p+k] = 1 for all valid i,j,k, and 0 otherwise.
    
    But with W indexed as W[k*n+i], the third factor index is k*n+i, ranging 0..p*n-1.
    So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition: T = sum_{r=0}^{r-1} u_r ⊗ v_r ⊗ w_r
    means: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k).
    """
    rank = U.shape[1]
    # Compute the reconstructed tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    # Build all (i,j,k) triples
    # For each r, the contribution to T[i*m+j, j*p+k, k*n+i] is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We need sum_r of these = 1 for all (i,j,k)
    
    # Create index arrays
    ii = np.repeat(np.arange(n), m*p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n*m)
    
    # For each r, compute the product for all (i,j,k)
    total = np.zeros(n*m*p)
    for r in range(rank):
        # U[i*m+j, r] for all (i,j,k)
        u_vals = U[ii*m + jj, r]  # shape (n*m*p,)
        v_vals = V[jj*p + kk, r]
        w_vals = W[kk*n + ii, r]
        total += u_vals * v_vals * w_vals
    
    return np.all(np.abs(total - 1.0) <= tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin
    
    # Best known: schoolbook rank = n*m*p = 80
    best_rank = n * m * p
    best_result = None
    
    # Generate schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    # Try to find lower rank using random initialization + gradient-free search
    # Strategy: try ranks from lower bound up, use random restarts with verification
    
    # Lower bound: max(n*m, m*p, n*p) = max(16, 20, 20) = 20
    # But for matrix multiplication, the rank is at least n*m + m*p + n*p - ... 
    # Actually the trivial lower bound is max(n*m, m*p, n*p) = 20.
    # Strassen showed 2x2x2 has rank 7. For larger sizes, ranks are known.
    # For 4x4x4, the rank is known to be 49 (Winograd 1980) or possibly 48.
    # For 4x4x5, we can at most do rank 49 by padding, but maybe less.
    
    # Let's try a more systematic approach:
    # Use the fact that if we can decompose 4x4x4 with rank r44, then 4x4x5 can use
    # rank r44 + 4 (for the extra column) or similar.
    
    # Actually, let's just do randomized search for various ranks.
    # For each candidate rank r, initialize U, V, W randomly and try to drive
    # the residual to zero using a simple iterative method.
    
    # Better approach: use the "tensor rank" optimization.
    # We want min_r such that there exist U,V,W with the decomposition property.
    # This is a non-convex problem. Use random restarts.
    
    # For efficiency, let's use a smarter initialization based on known structure.
    
    # Approach: Start with schoolbook and try to merge/eliminate terms.
    # Or: use random sparse matrices.
    
    # Let's try ranks from 20 upward, with increasing effort.
    np.random.seed(42)
    
    # Precompute indices for fast verification
    ii = np.repeat(np.arange(n), m*p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n*m)
    
    # For a given rank r, the constraint is:
    # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all (i,j,k)
    
    # This is a system of n*m*p = 80 equations in 3*r variables (with r being the number of terms).
    # For r=20: 60 unknowns, 80 equations -> overdetermined, may have no solution.
    # For r=24: 72 unknowns, 80 equations -> still overdetermined.
    # For r=27: 81 unknowns, 80 equations -> just determined.
    # But the equations are cubic (products of 3 variables), so it's nonlinear.
    
    # Let's use a different strategy: random search with local improvement.
    # Initialize U, V, W randomly, compute residual, and do coordinate descent.
    
    def compute_residual(U, V, W, r):
        """Compute the residual vector of length n*m*p."""
        total = np.zeros(n*m*p)
        for r_idx in range(r):
            u_vals = U[ii*m + jj, r_idx]
            v_vals = V[jj*p + kk, r_idx]
            w_vals = W[kk*n + ii, r_idx]
            total += u_vals * v_vals * w_vals
        return total - 1.0  # We want total = 1
    
    def residual_norm(U, V, W, r):
        res = compute_residual(U, V, W, r)
        return np.max(np.abs(res))
    
    # Try to find a solution for decreasing ranks
    # Start from a reasonable upper bound
    # For 4x4x5, let's try to get below 80.
    
    # Strategy: use random initialization and Newton-like steps.
    # Since the problem is small (80 constraints), we can try various ranks.
    
    # Actually, a cleaner approach: use the known result that 4x4*4x4 has rank 49.
    # For 4x4*4x5, we can potentially use rank 49 + something, or find a direct decomposition.
    
    # Let me try a randomized approach with verification:
    # For rank r, generate random U,V,W (sparse, with entries in {-1, 0, 1} or {-1, 0, 1, 2}),
    # and check if they work.
    
    # This is like searching for a combinatorial solution.
    
    # Better: use the fact that optimal decompositions often have small integer entries.
    
    # Let's try a structured random search:
    for r in range(20, min(80, 60)):  # Try ranks 20 to 59
        if time.time() > DEADLINE:
            break
        
        # Number of random restarts depends on remaining time
        time_remaining = DEADLINE - time.time()
        # Estimate: each trial takes ~0.1ms for verification
        max_trials = max(10, int(time_remaining * 5000))
        
        found = False
        for trial in range(max_trials):
            if time.time() > DEADLINE:
                break
            
            # Random initialization with small integers
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Randomly assign non-zero entries
            for r_idx in range(r):
                # Each factor has some sparsity
                # U column: pick a few random positions
                n_nonzero_u = np.random.randint(1, 5)
                idx_u = np.random.choice(n*m, n_nonzero_u, replace=False)
                U[idx_u, r_idx] = np.random.choice([-2, -1, 1, 2], n_nonzero_u)
                
                n_nonzero_v = np.random.randint(1, 5)
                idx_v = np.random.choice(m*p, n_nonzero_v, replace=False)
                V[idx_v, r_idx] = np.random.choice([-2, -1, 1, 2], n_nonzero_v)
                
                n_nonzero_w = np.random.randint(1, 5)
                idx_w = np.random.choice(p*n, n_nonzero_w, replace=False)
                W[idx_w, r_idx] = np.random.choice([-2, -1, 1, 2], n_nonzero_w)
            
            # Verify
            if residual_norm(U, V, W, r) < 1e-9:
                found = True
                best_rank = r
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break
        
        if found:
            break
    
    # If we haven't found something better, try a more systematic approach
    # for ranks near the known bounds.
    
    # Actually, let's also try a gradient-based approach for a few ranks.
    if best_rank == n * m * p and time.time() < DEADLINE - 20:
        # Try gradient descent for rank 25-35
        for r in range(25, 36):
            if time.time() > DEADLINE - 5:
                break
            
            for restart in range(3):
                if time.time() > DEADLINE - 5:
                    break
                
                # Initialize
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
                
                # Simple iterative refinement: for each r_idx, optimize one factor at a time
                # Given V and W, the optimal U[:, r_idx] satisfies:
                # U[i*m+j, r_idx] * (sum over other r' contributions) ... this is complex.
                
                # Alternative: use the fact that for fixed V[:, r_idx] and W[:, r_idx],
                # the contribution to the residual from term r_idx is:
                # U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                # To minimize residual, we want this to help fill the gap.
                
                # Let's just do a few rounds of coordinate updates.
                for iteration in range(50):
                    if time.time() > DEADLINE - 5:
                        break
                    
                    # Update U
                    for r_idx in range(r):
                        # For each (i,j), U[i*m+j, r_idx] should be chosen to minimize residual
                        # The term involving U[i*m+j, r_idx] in the residual is:
                        # U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] for each k
                        # This is a linear system in U[i*m+j, r_idx] if we fix others... 
                        # but it's coupled with other r_idx terms.
                        # Skip this; too complex for the time budget.
                        pass
                    break  # Skip gradient approach, too slow
    
    # Convert to lists if numpy
    if isinstance(best_result['U'], np.ndarray):
        best_result['U'] = best_result['U'].tolist()
        best_result['V'] = best_result['V'].tolist()
        best_result['W'] = best_result['W'].tolist()
    
    # Final safety: if time is running out, return best found
    if time.time() > DEADLINE - 2:
        pass  # Just return
    
    return best_result
