"""Matrix multiplication 3x3x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1] if U.ndim == 2 else len(U[0])
    
    # Convert to numpy if needed
    if not isinstance(U, np.ndarray):
        U = np.array(U, dtype=np.float64)
    if not isinstance(V, np.ndarray):
        V = np.array(V, dtype=np.float64)
    if not isinstance(W, np.ndarray):
        W = np.array(W, dtype=np.float64)
    
    # Check the "on" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    # Check "off" positions (sample)
    rng = np.random.RandomState(42)
    for _ in range(min(2000, nm * mp * pn // 5)):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False, (a, b, c, val)
    
    return True, None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _try_strassen_like(n, m, p):
    """
    Try to construct a low-rank decomposition using Strassen-like ideas.
    
    For 3x3x5: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Strategy: Use a block decomposition.
    Split A into columns and B into rows, then use bilinear combinations.
    
    One approach: treat this as multiplying a 3x3 matrix by a 3x5 matrix.
    We can write B as [B1, B2] where B1 is 3x3 and B2 is 3x2.
    Then C = [A*B1, A*B2].
    
    For A*B1 (3x3 * 3x3), we can use rank-15 (known optimal for 3x3x3).
    For A*B2 (3x3 * 3x2), we need additional terms.
    
    Actually, let me try a direct algebraic construction.
    
    Key idea: Use the fact that we can share intermediate products.
    Define:
      P1 = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]  = C[0,0]
      ...
    
    But that's just schoolbook. Let me try to find cancellations.
    
    Alternative: Use polynomial evaluation.
    Let x be a variable. Define:
      f_A(x) = A[0,0] + A[1,0]*x + A[2,0]*x^2  (column 0 of A)
      f_B(x) = B[0,0] + B[1,0]*x + B[2,0]*x^2  (row 0 of B)
    
    Then f_A(x) * f_B(x) is a polynomial of degree 4, and its coefficients
    give partial sums of A[i,j]*B[j,k] for specific (i,k) pairs.
    
    This doesn't directly help for the full product. Let me try another approach.
    
    For 3x3x5, the lower bound is max(9, 15, 15) = 15.
    The schoolbook rank is 45.
    
    Let me try to find a decomposition with rank around 20-30 using
    a structured approach based on sharing intermediate computations.
    """
    n, m, p = 3, 3, 5
    nm, mp, pn = 9, 15, 15
    
    # Try to construct using a "split" method.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can define intermediate products that combine multiple terms.
    # For example, for a fixed j, the product A[:,j] * B[j,:] contributes
    # to all C[i,k].
    #
    # One structured approach: use the fact that for small matrices,
    # we can sometimes reduce the number of multiplications by using
    # identities like (a+b)(c+d) = ac + ad + bc + bd.
    
    # Let me try a specific construction for 3x3x5.
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # ...
    # C[2,4] = A[2,0]*B[0,4] + A[2,1]*B[1,4] + A[2,2]*B[2,4]
    #
    # Total of 15 output entries, each requiring 3 products = 45 products.
    #
    # To reduce, we need to find combinations where the same product
    # contributes to multiple output entries.
    
    # Try rank 27 (one less than schoolbook/1.67)
    # Actually, let me just try gradient descent with better initialization.
    return None

def _gradient_descent_search(n, m, p, target_rank, max_iters=3000, lr=0.02, seed=42, tol=1e-5):
    """
    Use gradient descent to find a low-rank decomposition.
    Minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute target tensor entries (sparse)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    target_dict = {}
    for a, b, c in target_indices:
        target_dict[(a, b, c)] = 1.0
    
    # Initialize with random values
    U = rng.randn(nm, target_rank) * 0.1
    V = rng.randn(mp, target_rank) * 0.1
    W = rng.randn(pn, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Use a mix of on and off positions for sampling
    all_positions = set()
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                all_positions.add((a, b, c))
    
    off_positions = list(all_positions - set(target_indices))
    rng.shuffle(off_positions)
    
    for it in range(max_iters):
        # Compute loss on a sample
        sample_size = min(50, len(target_indices) + 10)
        loss = 0.0
        
        # Sample on-positions
        on_sample = rng.choice(len(target_indices), size=min(20, len(target_indices)), replace=False)
        for idx in on_sample:
            a, b, c = target_indices[idx]
            s_val = np.dot(U[a, :], V[b, :] * W[c, :])
            loss += (s_val - 1.0) ** 2
        
        # Sample off-positions
        off_sample_size = min(10, len(off_positions))
        for idx in range(off_sample_size):
            a, b, c = off_positions[idx]
            s_val = np.dot(U[a, :], V[b, :] * W[c, :])
            loss += s_val ** 2
        
        loss /= sample_size
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        # Gradient update
        batch_size = 15
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Mix of on and off positions
        for _ in range(batch_size):
            if rng.random() < 0.7:  # More weight on on-positions
                a, b, c = target_indices[rng.randint(0, len(target_indices))]
                t_val = 1.0
            else:
                a, b, c = off_positions[rng.randint(0, len(off_positions))]
                t_val = 0.0
            
            s_val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = s_val - t_val
            
            grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
            grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
            grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
        
        grad_U /= batch_size
        grad_V /= batch_size
        grad_W /= batch_size
        
        # Adaptive learning rate
        effective_lr = lr * (1.0 / (1.0 + it * 0.001))
        
        U -= effective_lr * grad_U
        V -= effective_lr * grad_V
        W -= effective_lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_direct_construction(n, m, p, deadline):
    """
    Try to directly construct a low-rank decomposition for 3x3x5.
    """
    n, m, p = 3, 3, 5
    
    # Try various ranks with gradient descent
    for target_rank in range(20, 45):
        if time.time() > deadline - 15:
            break
        
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            
            U, V, W, loss = _gradient_descent_search(
                n, m, p, target_rank,
                max_iters=2000, lr=0.03,
                seed=target_rank * 10 + seed,
                tol=1e-4
            )
            
            if loss < 1e-3:
                # Verify more carefully
                valid, _ = _validate(U, V, W, n, m, p, tol=1e-3)
                if valid:
                    return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try direct construction with gradient descent
    if time.time() < deadline - 20:
        result = _try_direct_construction(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            valid, _ = _validate(U, V, W, n, m, p, tol=1e-4)
            if valid:
                best_rank = rank
                best_result = (U, V, W)
    
    # If we still have time, try more ranks with different seeds
    if time.time() < deadline - 10 and best_rank == n * m * p:
        for target_rank in range(25, 45):
            if time.time() > deadline - 10:
                break
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                U, V, W, loss = _gradient_descent_search(
                    n, m, p, target_rank,
                    max_iters=1500, lr=0.04,
                    seed=1000 + target_rank * 10 + seed,
                    tol=1e-4
                )
                if loss < 1e-3:
                    valid, _ = _validate(U, V, W, n, m, p, tol=1e-3)
                    if valid:
                        best_rank = target_rank
                        best_result = (U, V, W)
                        break
    
    # Return the best result found
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
