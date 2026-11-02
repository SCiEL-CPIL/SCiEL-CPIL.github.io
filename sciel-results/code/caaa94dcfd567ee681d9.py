"""Matrix multiplication 2,2,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if (i % m == j // p) and (k % n == i // m):
                    T[i, j, k] = 1.0

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    expected = T[i, j, k]
                    s = 0.0
                    for r in range(R):
                        s += U[i][r] * V[j][r] * W[k][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def verify_numpy(U, V, W):
        """Fast verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_approx = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        return np.allclose(T_approx, T, atol=1e-9)

    def build_rank14():
        """Standard rank-14: two independent Strassen 2x2 blocks."""
        R = 14
        U = [[0]*R for _ in range(nm)]
        V = [[0]*R for _ in range(mp)]
        W = [[0]*R for _ in range(pn)]
        
        # Left block (B cols 0,1): Strassen terms r=0..6
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[5][0]=1
        W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        U[0][2]=1
        V[1][2]=1; V[5][2]=-1
        W[2][2]=1; W[3][2]=1
        U[3][3]=1
        V[4][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        U[0][4]=1; U[1][4]=1
        V[5][4]=1
        W[0][4]=-1; W[2][4]=1
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        U[1][6]=1; U[3][6]=-1
        V[4][6]=1; V[5][6]=1
        W[0][6]=1
        
        # Right block (B cols 2,3): Strassen terms r=7..13
        U[0][7]=1; U[3][7]=1
        V[2][7]=1; V[7][7]=1
        W[4][7]=1; W[7][7]=1
        U[2][8]=1; U[3][8]=1
        V[2][8]=1
        W[5][8]=1; W[7][8]=-1
        U[0][9]=1
        V[3][9]=1; V[7][9]=-1
        W[6][9]=1; W[7][9]=1
        U[3][10]=1
        V[6][10]=1; V[2][10]=-1
        W[4][10]=1; W[5][10]=1
        U[0][11]=1; U[1][11]=1
        V[7][11]=1
        W[4][11]=-1; W[6][11]=1
        U[2][12]=1; U[0][12]=-1
        V[2][12]=1; V[3][12]=1
        W[7][12]=1
        U[1][13]=1; U[3][13]=-1
        V[6][13]=1; V[7][13]=1
        W[4][13]=1

        return {"U": U, "V": V, "W": W}

    def try_rank13_structured():
        """
        Try to find rank-13 by merging one pair of Strassen terms that share
        the same A-factor pattern. We'll try several structural modifications
        and use least_squares to find exact solutions.
        """
        if time.time() > deadline - 30:
            return None

        R = 13
        T_flat = T.flatten()

        # Strategy: Start from rank-14 and try to merge two terms that have
        # the same or similar A-factor support. We'll try different merge patterns.
        
        # Pattern 1: Merge M1 and N1 (both use a00+a11 on A side)
        # M1: U=[1,0,0,1], V=[1,0,0,0,0,1,0,0], W=[1,0,0,1,0,0,0,0]
        # N1: U=[1,0,0,1], V=[0,0,1,0,0,0,0,1], W=[0,0,0,0,1,0,0,1]
        # Merged term would need V to have both b00+b11 AND b02+b13, which isn't a single product.
        # So we can't directly merge. Instead, let's try a different approach.
        
        # Better approach: Use a parameterized family of rank-13 decompositions
        # where we allow one "extra" degree of freedom to be absorbed.
        
        # Let's try: take the 14 Strassen terms and see if one can be expressed
        # as a linear combination of the others. We'll set up a system where
        # the 13th term is determined by the constraint that the tensor is exact.
        
        # Actually, let's try a direct optimization approach with structured initialization.
        
        np.random.seed(123)
        
        # Try multiple different structured initializations
        initializations = []
        
        # Init 1: Random sparse
        for _ in range(20):
            U_init = np.random.choice([-1, 0, 1], size=(4, R))
            V_init = np.random.choice([-1, 0, 1], size=(8, R))
            W_init = np.random.choice([-1, 0, 1], size=(8, R))
            initializations.append((U_init, V_init, W_init))
        
        # Init 2: Based on rank-14 with one term modified
        r14 = build_rank14()
        U14 = np.array(r14['U'], dtype=float)
        V14 = np.array(r14['V'], dtype=float)
        W14 = np.array(r14['W'], dtype=float)
        
        # Try dropping one term and reoptimizing
        for drop_idx in range(14):
            if time.time() > deadline - 5:
                break
            U_d = np.delete(U14, drop_idx, axis=1)
            V_d = np.delete(V14, drop_idx, axis=1)
            W_d = np.delete(W14, drop_idx, axis=1)
            # Add a small random perturbation to the last column
            U_p = U_d.copy()
            V_p = V_d.copy()
            W_p = W_d.copy()
            U_p[:, -1] += np.random.randn(4) * 0.1
            V_p[:, -1] += np.random.randn(8) * 0.1
            W_p[:, -1] += np.random.randn(8) * 0.1
            initializations.append((U_p, V_p, W_p))
        
        def residuals(x):
            U_flat = x[:4*R]
            V_flat = x[4*R:4*R+8*R]
            W_flat = x[4*R+8*R:]
            U_mat = U_flat.reshape(4, R)
            V_mat = V_flat.reshape(8, R)
            W_mat = W_flat.reshape(8, R)
            T_approx = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
            return (T_approx - T).flatten()
        
        for U_init, V_init, W_init in initializations:
            if time.time() > deadline - 5:
                break
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500)
            except Exception:
                continue
            
            if result.cost < 1e-10:
                U_sol = result.x[:4*R].reshape(4, R)
                V_sol = result.x[4*R:4*R+8*R].reshape(8, R)
                W_sol = result.x[4*R+8*R:].reshape(8, R)
                
                # Try rounding to integers
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                
                if verify_numpy(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try rounding to half-integers
                U_half = np.round(2 * U_sol) / 2
                V_half = np.round(2 * V_sol) / 2
                W_half = np.round(2 * W_sol) / 2
                
                if verify_numpy(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        
        return None

    def try_rank13_algebraic():
        """
        Algebraic approach: The tensor for 2x2 * 2x4 multiplication has a specific
        structure. Let's analyze if rank 13 is possible by checking the border rank
        or using a different decomposition strategy.
        
        Key insight: The two column blocks of B are not independent - they share
        the same A matrix. This means there might be a way to use fewer than
        14 multiplications.
        
        Let's try a decomposition where we use 7 multiplications for the first
        two columns and 6 for the last two, sharing one "partial" result.
        """
        if time.time() > deadline - 20:
            return None
        
        R = 13
        T_flat = T.flatten()
        
        np.random.seed(456)
        
        # Try a structured approach: use 7 terms for left block, 6 for right block
        # with one term shared between them.
        
        # Let's parameterize: 
        # Terms 0-6: left block (standard Strassen-like)
        # Terms 7-12: right block (6 terms, one shared with left)
        
        # We'll try to find a 6-term decomposition for the right block that
        # reuses one A-factor from the left block.
        
        def residuals_structured(x):
            # x contains 13*4 + 13*8 + 13*8 = 52 + 104 + 104 = 260 variables
            U_flat = x[:4*R]
            V_flat = x[4*R:4*R+8*R]
            W_flat = x[4*R+8*R:]
            U_mat = U_flat.reshape(4, R)
            V_mat = V_flat.reshape(8, R)
            W_mat = W_flat.reshape(8, R)
            T_approx = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
            return (T_approx - T).flatten()
        
        # Try different random seeds and initializations
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed * 1000 + 789)
            
            U_init = np.random.randn(4, R)
            V_init = np.random.randn(8, R)
            W_init = np.random.randn(8, R)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals_structured, x0, method='lm', max_nfev=300)
            except Exception:
                continue
            
            if result.cost < 1e-10:
                U_sol = result.x[:4*R].reshape(4, R)
                V_sol = result.x[4*R:4*R+8*R].reshape(8, R)
                W_sol = result.x[4*R+8*R:].reshape(8, R)
                
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                
                if verify_numpy(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        return None

    # Try rank-13 first with structured approach
    result = try_rank13_structured()
    if result is not None:
        return result
    
    # Try algebraic approach
    result = try_rank13_algebraic()
    if result is not None:
        return result
    
    # Fall back to rank-14
    return build_rank14()
# EVOLVE-BLOCK-END
