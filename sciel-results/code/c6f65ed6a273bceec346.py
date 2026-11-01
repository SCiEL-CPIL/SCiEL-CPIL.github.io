"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_2x2x2():
        """
        Standard Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns (U, V, W) with rank 7.
        
        For matrices A (2x2), B (2x2), C = A*B:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        """
        rank = 7
        U = np.zeros((4, rank))  # 2x2 matrix entries
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_hybrid_decomposition():
        """
        Build a hybrid decomposition for 2x3 * 3x7.
        
        Strategy: Split the 3x7 matrix into blocks and use Strassen where possible.
        
        A is 2x3: [[a00, a01, a02], [a10, a11, a12]]
        B is 3x7: split into columns
        C is 2x7
        
        We can view this as computing:
        C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2]  (2x2 * 2x2, use Strassen)
        C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4]  (2x2 * 2x2, use Strassen)
        C[:, 4:6] = A[:, 0:2] * B[0:2, 4:6]  (2x2 * 2x2, use Strassen)
        C[:, 6:7] = A[:, 0:3] * B[0:3, 6:7]  (2x3 * 3x1, schoolbook)
        
        But we need to be more careful about the structure.
        
        Alternative approach: Use the fact that m=3 is small.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For each output column k, we compute:
        C[:,k] = A * B[:,k]
        
        This is a 2x3 * 3x1 multiplication, which has rank 3 (schoolbook) 
        or potentially less with clever combinations.
        
        Total rank if independent: 7 * 3 = 21, but we can share structure.
        
        Let me try a different approach: exploit the small n=2 dimension.
        
        For fixed j, the contribution to C is:
        C[:,k] += A[:,j] * B[j,k]
        
        So C = sum_j (A[:,j] outer B[j,:])
        
        This gives rank m=3 if we could factor each term, but each term 
        A[:,j] outer B[j,:] is a 2x7 matrix of rank 1, so the sum of 3 
        such terms has rank at most 3... but that's for the output matrix, 
        not for the tensor decomposition rank.
        
        The tensor rank is different. Let me think more carefully.
        
        The multiplication tensor T has T[i,j,k] = 1 if the indices are 
        consistent with matrix multiplication.
        
        For 2x3*3x7:
        T[a,b,c] = 1 where a = i*m+j, b = j'*p+k, c = k'*n+i'
        and j = j' (the inner dimension must match).
        
        So T[i*m+j, j*p+k, k*n+i] = 1
        
        The rank of this tensor is what we want to minimize.
        
        Let me try to construct a decomposition by grouping terms.
        
        Key insight: For a fixed k, the terms involve:
        - i in {0,1}
        - j in {0,1,2}
        - The output index is k*n+i = 7*i+k
        
        For fixed k, we have 2*3 = 6 terms.
        These can potentially be combined using Strassen-like tricks.
        
        Let's try: for each k, use a rank-2 or rank-3 decomposition of the 
        2x3 inner product with the k-th column of B.
        
        Actually, let me try a completely different approach: 
        use the known result that the rank of matrix multiplication 
        M_{2,3,7} is at most 39 (from the cell findings).
        
        I'll construct a decomposition with rank 39 by combining:
        - 7 blocks, each handling one output column
        - Each block has rank 5-6 using Strassen-like combinations
        
        For output column k, C[:,k] = A * B[:,k]
        where A is 2x3 and B[:,k] is 3x1.
        
        This is equivalent to computing 2 inner products of length 3.
        The standard way uses 3 multiplications per output, total 6 per column.
        
        But we can do better by sharing multiplications across columns.
        
        Let me try: group columns in pairs and use Strassen.
        
        For columns 2k and 2k+1 (k=0,1,2):
        C[:, 2k:2k+2] = A * B[:, 2k:2k+2]
        
        This is 2x3 * 3x2, which can be done with rank 5 using 
        a variant of Strassen.
        
        For the last column (column 6):
        C[:, 6] = A * B[:, 6], rank 3.
        
        Total: 3 * 5 + 3 = 18? That seems too optimistic.
        
        Let me be more careful. The tensor rank for M_{2,3,2} is 8 
        (it's the same as M_{2,2,2} essentially, since the extra column 
        just adds one more output).
        
        Actually M_{2,3,2} has rank at most 2*3*2 = 12, and potentially 
        less. Let me just try to construct it directly.
        
        I'll use a systematic approach: enumerate all 42 terms and try 
        to combine them into rank-1 tensors with half-integer coefficients.
        """
        
        # Start with schoolbook (rank 42) and try to merge terms
        rank = 42
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Now try to merge pairs of terms that differ only in one index
        # to create Strassen-like combinations
        
        # For Strassen on 2x2 blocks:
        # Terms (i=0,j=0,k=0) and (i=1,j=1,k=1) can be combined:
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This replaces 4 terms with 1 term (net saving of 3)
        
        # Let me try to find such combinations systematically
        
        # Group terms by (j,k) pairs
        # For fixed j, we have terms for i=0 and i=1
        # For fixed k, we have terms for i=0 and i=1
        
        # Strassen combination: 
        # Replace {(i1,j,k1), (i2,j,k2), (i1,j',k1), (i2,j',k2)} 
        # with combinations
        
        # This is complex. Let me try a different approach:
        # Use known Strassen decomposition for sub-blocks.
        
        # Split the 2x3 matrix into a 2x2 block and a 2x1 block
        # Split the 3x7 matrix into a 2x7 block and a 1x7 block
        
        # C = A1*B1 + A2*B2
        # where A1 is 2x2, B1 is 2x7, A2 is 2x1, B2 is 1x7
        
        # A1*B1: 2x2 * 2x7, can be done with rank 7*2 = 14 (schoolbook)
        # or less with Strassen: 7 * (rank of 2x2*2x2) / 4 * ... 
        # Actually, 2x2 * 2x7 = sum of 7 (2x2*2x1) multiplications
        # Each 2x2*2x1 has rank 3, so total 7*3 = 21
        # But we can do better by sharing: 2x2*2x7 has rank at most 14
        
        # A2*B2: 2x1 * 1x7, rank 2 (each of the 2 outputs needs 1 mult)
        
        # Total: 14 + 2 = 16? Still seems optimistic.
        
        # Let me just try to verify if rank 39 works by construction.
        
        # I'll use the following strategy:
        # - 7 output columns
        # - For each pair of columns (0,1), (2,3), (4,5): use rank 5
        # - For column 6: use rank 3
        # - Plus some shared terms: total rank = 3*5 + 3 + extra = ?
        
        # Actually, let me just return schoolbook for now and rely on 
        # the optimization to find a better decomposition.
        
        return U, V, W
    
    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)
    
    def compute_gradients(U, V, W):
        """Compute gradients of tensor_error w.r.t. U, V, W efficiently."""
        r = U.shape[1]
        R = compute_R(U, V, W)
        E = R - target  # error tensor
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            M_U = np.einsum('abc,c->ab', E, W[:, c])
            grad_U[:, c] = 2.0 * (M_U @ V[:, c])
            
            N_V = np.einsum('abc,a->bc', E, U[:, c])
            grad_V[:, c] = 2.0 * (N_V @ W[:, c])
            
            UV_outer = np.outer(U[:, c], V[:, c])  # shape (nm, mp)
            grad_W[:, c] = 2.0 * np.einsum('abc,ab->c', E, UV_outer)
        
        return grad_U, grad_V, grad_W
    
    def optimize(U, V, W, max_iters=500, lr_init=0.1):
        """Optimize U, V, W to minimize tensor error using gradient descent with momentum."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        # Momentum
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.9
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            # Update velocities
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            # Update
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                # Decay learning rate
                lr *= 0.995
            else:
                # Roll back and reduce learning rate
                lr *= 0.5
                if lr < 1e-10:
                    break
        
        return U, V, W, prev_error
    
    def polish(U, V, W, max_iters=1000, lr_init=0.01):
        """High-precision polish pass with tighter tolerance and more iterations."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.95
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                lr *= 0.99
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        return U, V, W, prev_error
    
    def try_strassen_hybrid():
        """
        Try to construct a rank-39 decomposition using Strassen on 2x2 blocks.
        
        Approach:
        - Split A (2x3) into A_left (2x2) and A_right (2x1)
        - Split B (3x7) into B_left (2x7) and B_right (1x7)
        - C = A_left * B_left + A_right * B_right
        
        For A_left * B_left (2x2 * 2x7):
        - Split B_left into 7 columns, each 2x1
        - For each pair of columns, use a rank-5 decomposition
        - Total: 3 pairs * 5 + 1 single * 3 = 18
        
        For A_right * B_right (2x1 * 1x7):
        - Rank 2 (one multiplication per output row)
        
        Total estimated rank: 18 + 2 = 20 (optimistic)
        
        In practice, we'll use schoolbook for most and Strassen for some blocks.
        """
        
        # Let's try a concrete construction
        # Use Strassen for the 2x2 * 2x2 subproblems
        
        U = np.zeros((nm, 39))
        V = np.zeros((mp, 39))
        W = np.zeros((pn, 39))
        
        col = 0
        
        # Handle columns 0-5 in pairs using Strassen-like combinations
        for k_pair in range(3):
            k1 = 2 * k_pair
            k2 = 2 * k_pair + 1
            
            # For output columns k1 and k2:
            # C[i,k1] = sum_j A[i,j] * B[j,k1]
            # C[i,k2] = sum_j A[i,j] * B[j,k2]
            
            # Use Strassen on the 2x2 block formed by rows 0,1 of A 
            # and columns k1,k2 of B
            
            # M1 = (A00+A11)(B0k1+B0k2)  -- but B is 3x7, not 2x2
            # This doesn't directly work because the inner dimension is 3, not 2.
            
            # Let me try a different grouping.
            
            # Actually, for the 2x3 * 3x2 subproblem:
            # C = A * B where A is 2x3, B is 3x2
            # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
            
            # We can use the following 7-term decomposition (Strassen-like):
            # M1 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0])
            # M2 = (A[0,0]-A[0,1]) * (B[1,0]+B[2,0])
            # M3 = (A[0,1]-A[0,2]) * (B[0,0]+B[1,0])
            # M4 = A[0,0] * B[0,0]
            # M5 = A[0,1] * B[1,0]
            # M6 = A[0,2] * B[2,0]
            # ... this is getting complicated.
            
            # Let me just use schoolbook for now and try to optimize.
            for j in range(m):
                for i in range(n):
                    U[i*m+j, col] = 1
                    V[j*p+k1, col] = 1
                    W[k1*n+i, col] = 1
                    col += 1
                    if col >= 39:
                        break
                if col >= 39:
                    break
            if col >= 39:
                break
        
        if col >= 39:
            return U, V, W
        
        # Fill remaining with schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= 39:
                        break
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W
    
    def try_rank(rank, num_attempts=5):
        """Try to find a valid decomposition for a given rank using multiple strategies."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Strategy 1: Strassen hybrid
        if rank == 39:
            U_init, V_init, W_init = try_strassen_hybrid()
            U_opt, V_opt, W_opt, final_error = optimize(U_init, V_init, W_init, max_iters=200, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # Strategy 2: Random init + gradient descent
        for attempt in range(min(3, num_attempts)):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(hash((rank, attempt, 1)) % 2**32)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=150, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # Strategy 3: Start from schoolbook and try to reduce rank
        if rank < n * m * p and time.time() < deadline - 25:
            U_sb, V_sb, W_sb = schoolbook()
            U_red = U_sb[:, :rank]
            V_red = V_sb[:, :rank]
            W_red = W_sb[:, :rank]
            
            U_opt, V_opt, W_opt, final_error = optimize(U_red, V_red, W_red, max_iters=150, lr_init=0.05)
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # If we have a best candidate, try to polish and round
        if best_U is not None and best_error < 1.0:
            U_pol, V_pol, W_pol, _ = polish(best_U, best_V, best_W, max_iters=300, lr_init=0.005)
            U_pol = round_to_half_integer(U_pol)
            V_pol = round_to_half_integer(V_pol)
            W_pol = round_to_half_integer(W_pol)
            
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol
        
        return None, None, None
    
    best_result = None
    best_rank = 42
    
    # Try ranks from low to high, prioritizing 39 (known to work)
    for rank in [39, 40, 41]:
        if time.time() > deadline - 25:
            break
        
        U, V, W = try_rank(rank, num_attempts=5)
        
        if U is not None:
            best_result = (U, V, W)
            best_rank = rank
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
