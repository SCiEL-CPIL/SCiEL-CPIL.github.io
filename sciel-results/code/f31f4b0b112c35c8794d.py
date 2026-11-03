"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    deadline = time.time() + 115  # Safety margin before 120s limit (evaluator kills at 150s)
    
    # Build the target tensor
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # a in 0..5, b in 0..26, c in 0..17
    target = np.zeros((num_a, num_b, num_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 54) as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(num_a)]
        V = [[0] * rank for _ in range(num_b)]
        W = [[0] * rank for _ in range(num_c)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def compute_residual(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W) - target
    
    def compute_error(U, V, W):
        res = np.einsum('ar,br,cr->abc', U, V, W) - target
        return np.sum(res**2)
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact within tolerance"""
        res = np.einsum('ar,br,cr->abc', U, V, W) - target
        return np.max(np.abs(res)) < tol
    
    def solve_for_U(V, W, target, R):
        """Solve for U given V, W using least squares"""
        # M[bc, r] = V[b,r] * W[c,r]
        M = np.zeros((num_b * num_c, R))
        for b in range(num_b):
            for c in range(num_c):
                M[b * num_c + c, :] = V[b, :] * W[c, :]
        
        U = np.zeros((num_a, R))
        for a in range(num_a):
            rhs = target[a, :, :].flatten()
            U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return U
    
    def solve_for_V(U, W, target, R):
        """Solve for V given U, W using least squares"""
        M = np.zeros((num_a * num_c, R))
        for a in range(num_a):
            for c in range(num_c):
                M[a * num_c + c, :] = U[a, :] * W[c, :]
        
        V = np.zeros((num_b, R))
        for b in range(num_b):
            rhs = target[:, b, :].transpose(1, 0).flatten()
            V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return V
    
    def solve_for_W(U, V, target, R):
        """Solve for W given U, V using least squares"""
        M = np.zeros((num_a * num_b, R))
        for a in range(num_a):
            for b in range(num_b):
                M[a * num_b + b, :] = U[a, :] * V[b, :]
        
        W = np.zeros((num_c, R))
        for c in range(num_c):
            rhs = target[:, :, c].flatten()
            W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return W
    
    def alternating_minimization(R, n_iters=150, n_restarts=5, seed=42):
        """Run alternating minimization for rank R"""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        rng = np.random.RandomState(seed)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = rng.randn(num_a, R) * 0.1
            V = rng.randn(num_b, R) * 0.1
            W = rng.randn(num_c, R) * 0.1
            
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(V, W, target, R)
                V = solve_for_V(U, W, target, R)
                W = solve_for_W(U, V, target, R)
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-14:
                    break
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-10:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    def structured_init(R):
        """Create structured initializations based on Kronecker product ideas"""
        inits = []
        
        # Initialization 1: Block diagonal structure
        # Try to exploit the 2x3x9 = 3 parallel 2x3x3 structure
        if R >= 18:
            # Each 2x3x3 block gets 6 columns
            cols_per_block = R // 3
            U = np.zeros((num_a, R))
            V = np.zeros((num_b, R))
            W = np.zeros((num_c, R))
            
            for block in range(3):
                # For block b (columns b*cols_per_block to (b+1)*cols_per_block-1)
                start_col = block * cols_per_block
                end_col = start_col + cols_per_block
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b_idx = j * p + (block * 3 + k)  # k offset by block
                            c = k * n + i
                            
                            col = start_col + (i * m + j) * p + k  # This doesn't work cleanly
                            # Let's use a simpler structured init
                
                # Simpler: just use random init for this block
                U[:, start_col:end_col] = np.random.randn(num_a, cols_per_block) * 0.1
                V[:, start_col:end_col] = np.random.randn(num_b, cols_per_block) * 0.1
                W[:, start_col:end_col] = np.random.randn(num_c, cols_per_block) * 0.1
            
            inits.append((U, V, W))
        
        # Initialization 2: Pure random
        U = np.random.randn(num_a, R) * 0.1
        V = np.random.randn(num_b, R) * 0.1
        W = np.random.randn(num_c, R) * 0.1
        inits.append((U, V, W))
        
        # Initialization 3: Sparse random (mostly zeros with some ones)
        U = np.zeros((num_a, R))
        V = np.zeros((num_b, R))
        W = np.zeros((num_c, R))
        for r in range(R):
            # Pick random non-zero positions
            for a in range(num_a):
                if np.random.rand() < 0.3:
                    U[a, r] = np.random.choice([-1, 1])
            for b in range(num_b):
                if np.random.rand() < 0.2:
                    V[b, r] = np.random.choice([-1, 1])
            for c in range(num_c):
                if np.random.rand() < 0.3:
                    W[c, r] = np.random.choice([-1, 1])
        inits.append((U, V, W))
        
        return inits
    
    def try_rank(R, n_iters=100, n_restarts=3):
        """Try to find rank-R decomposition"""
        if time.time() > deadline - 5:
            return None
        
        # Try multiple initializations
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Method 1: Standard alternating minimization
        U, V, W, err = alternating_minimization(R, n_iters=n_iters, n_restarts=n_restarts)
        if U is not None:
            best_err = err
            best_U, best_V, best_W = U, V, W
            if err < 1e-10:
                return (U, V, W)
        
        # Method 2: Try structured initializations
        inits = structured_init(R)
        for idx, (U0, V0, W0) in enumerate(inits):
            if time.time() > deadline - 5:
                break
            
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            for it in range(n_iters):
                if time.time() > deadline - 5:
                    break
                
                U = solve_for_U(V, W, target, R)
                V = solve_for_V(U, W, target, R)
                W = solve_for_W(U, V, target, R)
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-14:
                    break
            
            if best_err < 1e-10:
                return (best_U, best_V, best_W)
        
        if best_err < 1e-10:
            return (best_U, best_V, best_W)
        return None
    
    # Try ranks from low to high
    # The theoretical minimum for 2x3x9 is at least 18 (from 3 parallel 2x3x3)
    # But it could be lower due to shared structure
    # Try: 18, 20, 22, 24, 27, 30, 36, 42, 48, 54
    ranks_to_try = [18, 20, 22, 24, 27, 30, 36, 42, 48, 54]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R, n_iters=80, n_restarts=3)
        
        if result is not None:
            U, V, W = result
            # Final verification
            if verify(U, V, W, tol=1e-8):
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
