"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition correctness."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = np.sum(Ua[a] * Va[b] * Wa[c])
                    if abs(val - 1.0) > tol:
                        return False
        # Check sparsity: entries that should be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    k = a % m
                    k2 = b // p
                    j = b % p
                    i2 = c % n
                    j2 = c // n
                    if i != i2 or k != k2 or j != j2:
                        val = np.sum(Ua[a] * Va[b] * Wa[c])
                        if abs(val) > tol:
                            return False
        return True
    
    # Known correct rank-14 Strassen solution
    def make_rank14():
        rank = 14
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1 (cols 0,1 of B): Strassen on A * B[:,0:2]
        U[0][0] = 1; U[3][0] = 1   # M1: a00+a11
        U[2][1] = 1; U[3][1] = 1   # M2: a10+a11
        U[0][2] = 1                # M3: a00
        U[3][3] = 1                # M4: a11
        U[0][4] = 1; U[1][4] = 1   # M5: a00+a01
        U[2][5] = 1; U[0][5] = -1  # M6: a10-a00
        U[1][6] = 1; U[3][6] = -1  # M7: a01-a11
        
        # Block 2 (cols 2,3 of B): Strassen on A * B[:,2:4]
        U[0][7] = 1; U[3][7] = 1
        U[2][8] = 1; U[3][8] = 1
        U[0][9] = 1
        U[3][10] = 1
        U[0][11] = 1; U[1][11] = 1
        U[2][12] = 1; U[0][12] = -1
        U[1][13] = 1; U[3][13] = -1
        
        V[0][0] = 1; V[5][0] = 1   # b00+b11
        V[0][1] = 1                # b00
        V[1][2] = 1; V[5][2] = -1  # b01-b11
        V[4][3] = 1; V[0][3] = -1  # b10-b00
        V[5][4] = 1                # b11
        V[0][5] = 1; V[1][5] = 1   # b00+b01
        V[4][6] = 1; V[5][6] = 1   # b10+b11
        
        V[2][7] = 1; V[7][7] = 1   # b02+b13
        V[2][8] = 1                # b02
        V[3][9] = 1; V[7][9] = -1  # b03-b13
        V[6][10] = 1; V[2][10] = -1 # b12-b02
        V[7][11] = 1               # b13
        V[2][12] = 1; V[3][12] = 1 # b02+b03
        V[6][13] = 1; V[7][13] = 1 # b12+b13
        
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        W[1][1] = 1; W[1][3] = 1
        W[2][2] = 1; W[2][4] = 1
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
        W[5][8] = 1; W[5][10] = 1
        W[6][9] = 1; W[6][11] = 1
        W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-12 by structured search
    # Key idea: merge the two Strassen decompositions by combining
    # certain B-forms so that some multiplications are shared.
    
    # Strategy: Use 4 "shared" products where the A-form is the same
    # for both blocks, but the B-form combines entries from both blocks.
    # Then 4 more products for block 1 and 4 more for block 2.
    
    # Actually, let's try a direct numerical approach:
    # Parameterize U, V with a structured ansatz and solve for W.
    
    def try_structured_rank12():
        """
        Structured ansatz for rank 12:
        - Terms 0-3: shared A-forms, combined B-forms (spanning both blocks)
        - Terms 4-7: block-1 specific
        - Terms 8-11: block-2 specific
        
        For shared terms, the A-form uses only A entries (same for both blocks),
        and the B-form uses entries from both B-blocks.
        """
        rank = 12
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Shared terms (0-3): These handle cross-block contributions
        # Term 0: A-form = (a00 + a11), B-form spans both blocks
        # We need to be careful: the B-form for a shared term must
        # produce correct contributions to BOTH output blocks.
        
        # Let's use a different structured approach:
        # For the 2x2*2x2 problem, Strassen uses 7 products.
        # For 2x2*2x4, we can think of it as:
        # C_left = A * B_left (2x2 * 2x2)
        # C_right = A * B_right (2x2 * 2x2)
        # 
        # Rank 12 = 7 + 7 - 2, meaning we share 2 products.
        # The shared products must have A-forms that are valid for both
        # and B-forms that combine entries from both blocks.
        
        # Let's try: use Strassen for block 1 with terms 0-6,
        # and for block 2, reuse some of those A-forms with modified B-forms.
        
        # Actually, let's try a cleaner approach:
        # Use 4 terms that are "diagonal" type (involving a00 and/or a11 only)
        # shared between blocks, and handle the "off-diagonal" terms separately.
        
        # Term 0: a00 * (b00 + b02)  -> contributes to C[0,0] via b00 and C[0,2] via b02
        U[0][0] = 1
        V[0][0] = 1; V[2][0] = 1
        W[0][0] = 1  # C[0,0] gets a00*b00
        W[4][0] = 1  # C[0,2] gets a00*b02
        
        # Term 1: a11 * (b10 + b12)  -> contributes to C[1,0] via b10 and C[1,2] via b12
        U[3][1] = 1
        V[4][1] = 1; V[6][1] = 1
        W[1][1] = 1  # C[1,0] gets a11*b10
        W[5][1] = 1  # C[1,2] gets a11*b12
        
        # Term 2: a10 * (b00 + b02)  -> contributes to C[1,0] via b00 and C[1,2] via b02
        U[2][2] = 1
        V[0][2] = 1; V[2][2] = 1
        W[1][2] = 1  # C[1,0] gets a10*b00
        W[5][2] = 1  # C[1,2] gets a10*b02
        
        # Term 3: a01 * (b10 + b12)  -> contributes to C[0,0] via b10 and C[0,2] via b12
        U[1][3] = 1
        V[4][3] = 1; V[6][3] = 1
        W[0][3] = 1  # C[0,0] gets a01*b10
        W[4][3] = 1  # C[0,2] gets a01*b12
        
        # Now we've handled the "diagonal" contributions for columns 0 and 2.
        # We still need to handle columns 1 and 3, and the off-diagonal corrections.
        
        # For column 1: C[0,1] = a00*b01 + a01*b11
        #                C[1,1] = a10*b01 + a11*b11
        # For column 3: C[0,3] = a00*b03 + a01*b13
        #                C[1,3] = a10*b03 + a11*b13
        
        # We can use Strassen-like combinations for the remaining terms.
        # Terms 4-7: handle column 1
        # Term 4: (a00+a11) * (b01+b11) -> helps with C[0,1] and C[1,1]
        U[0][4] = 1; U[3][4] = 1
        V[1][4] = 1; V[5][4] = 1
        # This gives: a00*b01 + a00*b11 + a11*b01 + a11*b11
        # C[0,1] needs a00*b01 + a01*b11
        # C[1,1] needs a10*b01 + a11*b11
        
        # This approach is getting complex. Let me try a numerical solve instead.
        
        return None  # placeholder
    
    # Let's use a numerical optimization approach
    def numerical_rank12_search():
        """
        Use scipy to find a rank-12 decomposition.
        We'll parameterize and use least-squares / nonlinear optimization.
        """
        from scipy.optimize import least_squares
        
        rank = 12
        # Variables: U (4x12), V (8x12), W (8x12) = 4*12 + 8*12 + 8*12 = 288 variables
        # Constraints: for each valid (i,k,j), sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1
        #              for each invalid (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # Build the constraint list
        valid_triples = []
        invalid_triples = []
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    k = a % m
                    k2 = b // p
                    j = b % p
                    i2 = c % n
                    j2 = c // n
                    if i == i2 and k == k2 and j == j2:
                        valid_triples.append((a, b, c))
                    else:
                        invalid_triples.append((a, b, c))
        
        # Total constraints: n*m*p valid + (n*m*m*p*p*n - n*m*p) invalid
        # That's 16 valid + (4*8*8 - 16) = 16 + 240 = 256 constraints
        
        def residuals(x):
            U = x[:4*rank].reshape(4, rank)
            V = x[4*rank:4*rank+8*rank].reshape(8, rank)
            W = x[4*rank+8*rank:4*rank+16*rank].reshape(8, rank)
            
            res = []
            for (a, b, c) in valid_triples:
                val = np.sum(U[a] * V[b] * W[c])
                res.append(val - 1.0)
            for (a, b, c) in invalid_triples:
                val = np.sum(U[a] * V[b] * W[c])
                res.append(val)
            return res
        
        # Initial guess: use the rank-14 solution, zero out 2 columns
        U14, V14, W14 = make_rank14()["U"], make_rank14()["V"], make_rank14()["W"]
        U14 = np.array(U14, dtype=float)
        V14 = np.array(V14, dtype=float)
        W14 = np.array(W14, dtype=float)
        
        # Try different initializations
        np.random.seed(42)
        
        best_result = None
        best_score = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                # Start from rank-14, zero out last 2 columns
                U_init = U14[:, :rank].copy()
                V_init = V14[:, :rank].copy()
                W_init = W14[:, :rank].copy()
            else:
                # Add noise to rank-14 truncated
                noise_scale = 0.1
                U_init = U14[:, :rank].copy() + np.random.randn(4, rank) * noise_scale
                V_init = V14[:, :rank].copy() + np.random.randn(8, rank) * noise_scale
                W_init = W14[:, :rank].copy() + np.random.randn(8, rank) * noise_scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000, xtol=1e-12)
                res_norm = np.max(np.abs(result.fun))
                
                if res_norm < 1e-8:
                    U_sol = result.x[:4*rank].reshape(4, rank)
                    V_sol = result.x[4*rank:4*rank+8*rank].reshape(8, rank)
                    W_sol = result.x[4*rank+8*rank:4*rank+16*rank].reshape(8, rank)
                    
                    # Convert to lists
                    U_list = U_sol.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_sol.tolist()
                    
                    # Verify
                    if verify(U_list, V_list, W_list, tol=1e-8):
                        if best_result is None or res_norm < best_score:
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_score = res_norm
                        
                        if best_score < 1e-10:
                            break
            except Exception:
                continue
        
        return best_result
    
    # Try structured approaches first (fast)
    if time.time() < deadline - 30:
        # Try a specific structured rank-12 ansatz
        # Based on the idea of sharing "diagonal" Strassen terms
        pass
    
    # Use numerical search
    result = numerical_rank12_search()
    
    if result is not None:
        return result
    
    # Fallback to rank-14
    return make_rank14()
# EVOLVE-BLOCK-END
