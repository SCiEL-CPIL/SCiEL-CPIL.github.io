"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.reshape(nm * mp, pn)  # (216, 24)
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def build_strassen_decomp():
        """Build the Strassen-based rank-68 decomposition."""
        U_t = []
        V_t = []
        W_t = []
        
        def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
            a = (row_off_a, col_off_a)
            b = (row_off_a, col_off_a + 1)
            c = (row_off_a + 1, col_off_a)
            d = (row_off_a + 1, col_off_a + 1)
            e = (row_off_b, col_off_b)
            f = (row_off_b, col_off_b + 1)
            g = (row_off_b + 1, col_off_b)
            h = (row_off_b + 1, col_off_b + 1)
            
            C00 = (row_off_a, col_off_b)
            C01 = (row_off_a, col_off_b + 1)
            C10 = (row_off_a + 1, col_off_b)
            C11 = (row_off_a + 1, col_off_b + 1)
            
            # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
            U_t.append([(a, 1), (d, 1)])
            V_t.append([(e, 1), (h, 1)])
            W_t.append([(C00, 1), (C11, 1)])
            
            # m2 = (c+d)*e -> C10 += 1, C11 -= 1
            U_t.append([(c, 1), (d, 1)])
            V_t.append([(e, 1)])
            W_t.append([(C10, 1), (C11, -1)])
            
            # m3 = a*(f-h) -> C01 += 1, C11 += 1
            U_t.append([(a, 1)])
            V_t.append([(f, 1), (h, -1)])
            W_t.append([(C01, 1), (C11, 1)])
            
            # m4 = d*(g-e) -> C00 += 1, C10 += 1
            U_t.append([(d, 1)])
            V_t.append([(g, 1), (e, -1)])
            W_t.append([(C00, 1), (C10, 1)])
            
            # m5 = (a+b)*h -> C00 -= 1, C01 += 1
            U_t.append([(a, 1), (b, 1)])
            V_t.append([(h, 1)])
            W_t.append([(C00, -1), (C01, 1)])
            
            # m6 = (c-a)*(e+f) -> C11 += 1
            U_t.append([(c, 1), (a, -1)])
            V_t.append([(e, 1), (f, 1)])
            W_t.append([(C11, 1)])
            
            # m7 = (b-d)*(g+h) -> C00 += 1
            U_t.append([(b, 1), (d, -1)])
            V_t.append([(g, 1), (h, 1)])
            W_t.append([(C00, 1)])
        
        def direct_term(i, j, k):
            U_t.append([((i, j), 1)])
            V_t.append([((j, k), 1)])
            W_t.append([((i, k), 1)])
        
        # Apply Strassen to all 2x2 B-blocks for A[0:2, 0:2]
        for b_col in range(0, 8, 2):
            strassen_decomp(0, 0, 0, b_col)
        
        # Direct terms for A[0:2, 2] * B[2, :]
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
        
        # Direct terms for A[2, 0:2] * B[0:2, :]
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
        
        # Direct terms for A[2, 2] * B[2, :]
        for k in range(8):
            direct_term(2, 2, k)
        
        rank = len(W_t)
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
        
        return U, V, W, rank
    
    def verify(Ua, Va, Wa):
        """Verify decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Start with Strassen rank-68 decomposition
    U_sb, V_sb, W_sb, sb_rank = build_strassen_decomp()
    
    if not verify(U_sb, V_sb, W_sb):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_fb = np.zeros((nm, schoolbook_rank))
        V_fb = np.zeros((mp, schoolbook_rank))
        W_fb = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1.0
                    V_fb[j*p+k, col] = 1.0
                    W_fb[k*n+i, col] = 1.0
                    col += 1
        return {"U": [[float(x) for x in row] for row in U_fb],
                "V": [[float(x) for x in row] for row in V_fb],
                "W": [[float(x) for x in row] for row in W_fb]}
    
    best_rank = sb_rank
    best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    # New Strategy: Search for low-rank 3x3x2 decomposition and tile it
    # 3x3x8 can be seen as 4 independent 3x3x2 multiplications
    # Rank(3x3x8) <= 4 * Rank(3x3x2)
    
    def find_rank_3x3x2():
        """Search for low-rank decomposition of 3x3x2 tensor."""
        n2, m2, p2 = 3, 3, 2
        nm2 = n2 * m2 # 9
        mp2 = m2 * p2 # 6
        pn2 = p2 * n2 # 6
        
        T2 = np.zeros((nm2, mp2, pn2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        T2_flat = T2.reshape(nm2 * mp2, pn2)
        
        best_r2 = 17 # Current known block rank
        best_U2, best_V2, best_W2 = None, None, None
        
        # Try ranks from 16 down to 10
        for r in range(16, 9, -1):
            if time.time() > deadline - 20:
                break
            
            for attempt in range(50):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(12345 + r * 100 + attempt)
                
                # Random dense initialization
                U2 = np.random.randn(nm2, r)
                V2 = np.random.randn(mp2, r)
                
                # Scale to keep numbers manageable
                U2 /= np.linalg.norm(U2, axis=0, keepdims=True) + 1e-9
                V2 /= np.linalg.norm(V2, axis=0, keepdims=True) + 1e-9
                
                K2 = np.zeros((nm2 * mp2, r))
                for c in range(r):
                    K2[:, c] = np.outer(U2[:, c], V2[:, c]).flatten()
                
                try:
                    W2_sol, residuals, rank_K, sv = np.linalg.lstsq(K2, T2_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K2 @ W2_sol - T2_flat).max()
                
                if residual < 1e-9:
                    # Check if W2_sol can be represented with integers/half-integers
                    # The problem allows integer/half-integer entries.
                    # We can multiply W2_sol by 2 and check if it's close to integers.
                    W2_check = W2_sol * 2
                    if np.allclose(W2_check, np.round(W2_check), atol=1e-6):
                        W2_final = np.round(W2_check) / 2
                        
                        # Verify
                        recon2 = np.zeros((nm2, mp2, pn2))
                        for c in range(r):
                            recon2 += np.einsum('a,b,c->abc', U2[:,c], V2[:,c], W2_final[:,c])
                        if np.abs(recon2 - T2).max() < 1e-8:
                            best_r2 = r
                            best_U2 = U2.copy()
                            best_V2 = V2.copy()
                            best_W2 = W2_final.copy()
                            break
            
            if best_r2 < 17:
                break
        
        if best_U2 is not None:
            return best_U2, best_V2, best_W2, best_r2
        return None, None, None, 17
    
    # Strategy: Find rank-21 decomposition for 3x3x4, then embed to 3x3x8 with rank 42
    def find_rank_3x3x4():
        """Search for low-rank decomposition of 3x3x4 tensor."""
        n4, m4, p4 = 3, 3, 4
        nm4 = n4 * m4  # 9
        mp4 = m4 * p4  # 12
        pn4 = p4 * n4  # 12
        
        T4 = np.zeros((nm4, mp4, pn4))
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    T4[i*m4+j, j*p4+k, k*n4+i] = 1.0
        T4_flat = T4.reshape(nm4 * mp4, pn4)  # (108, 12)
        
        best_r4 = 25  # Start with a safe upper bound
        best_U4, best_V4, best_W4 = None, None, None
        
        # Try ranks from 21 down to 15
        for r in range(21, 14, -1):
            if time.time() > deadline - 20:
                break
            
            for attempt in range(100):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(99999 + r * 1000 + attempt)
                
                # Random dense initialization with structured patterns
                U4 = np.random.randn(nm4, r)
                V4 = np.random.randn(mp4, r)
                
                # Try different sparsity patterns
                sparsity = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                U4[np.random.random((nm4, r)) < (1 - sparsity)] = 0
                V4[np.random.random((mp4, r)) < (1 - sparsity)] = 0
                
                # Ensure no all-zero columns
                for c in range(r):
                    if np.all(U4[:, c] == 0):
                        U4[np.random.randint(nm4), c] = 1.0
                    if np.all(V4[:, c] == 0):
                        V4[np.random.randint(mp4), c] = 1.0
                
                # Scale to keep numbers manageable
                U4 /= (np.linalg.norm(U4, axis=0, keepdims=True) + 1e-9)
                V4 /= (np.linalg.norm(V4, axis=0, keepdims=True) + 1e-9)
                
                K4 = np.zeros((nm4 * mp4, r))
                for c in range(r):
                    K4[:, c] = np.outer(U4[:, c], V4[:, c]).flatten()
                
                try:
                    W4_sol, residuals, rank_K, sv = np.linalg.lstsq(K4, T4_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K4 @ W4_sol - T4_flat).max()
                
                if residual < 1e-9:
                    # Check if W4_sol can be represented with integers/half-integers
                    W4_check = W4_sol * 2
                    if np.allclose(W4_check, np.round(W4_check), atol=1e-6):
                        W4_final = np.round(W4_check) / 2
                        
                        # Verify
                        recon4 = np.zeros((nm4, mp4, pn4))
                        for c in range(r):
                            recon4 += np.einsum('a,b,c->abc', U4[:,c], V4[:,c], W4_final[:,c])
                        if np.abs(recon4 - T4).max() < 1e-8:
                            best_r4 = r
                            best_U4 = U4.copy()
                            best_V4 = V4.copy()
                            best_W4 = W4_final.copy()
                            break
            
            if best_r4 < 25:
                break
        
        if best_U4 is not None:
            return best_U4, best_V4, best_W4, best_r4
        return None, None, None, 25
    
    U4, V4, W4, r4 = find_rank_3x3x4()
    
    if U4 is not None:
        # Embed 3x3x4 decomposition into 3x3x8 with rank doubling
        # For each term c in 0..r4-1, create two terms: 2c and 2c+1
        # Term 2c: handles even output columns (k=0,2,4,6)
        # Term 2c+1: handles odd output columns (k=1,3,5,7)
        
        rank8 = 2 * r4
        U8 = np.zeros((nm, rank8))
        V8 = np.zeros((mp, rank8))
        W8 = np.zeros((pn, rank8))
        
        for c in range(r4):
            # Term 2c: even k
            U8[:, 2*c] = U4[:, c]
            # V mapping: V4 index j*4+b (b=0..3) -> V8 index j*8+2b
            for j in range(m):
                for b in range(4):
                    V8[j*p + 2*b, 2*c] = V4[j*4 + b, c]
            # W mapping: W4 index b*3+i (b=0..3) -> W8 index (2b)*3+i
            for b in range(4):
                for i in range(n):
                    W8[(2*b)*n + i, 2*c] = W4[b*n + i, c]
            
            # Term 2c+1: odd k
            U8[:, 2*c+1] = U4[:, c]
            # V mapping: V4 index j*4+b -> V8 index j*8+2b+1
            for j in range(m):
                for b in range(4):
                    V8[j*p + 2*b + 1, 2*c+1] = V4[j*4 + b, c]
            # W mapping: W4 index b*3+i -> W8 index (2b+1)*3+i
            for b in range(4):
                for i in range(n):
                    W8[(2*b+1)*n + i, 2*c+1] = W4[b*n + i, c]
        
        if verify(U8, V8, W8):
            if rank8 < best_rank:
                best_rank = rank8
                best_decomp = (U8.copy(), V8.copy(), W8.copy())
    
    # If the 3x3x4 embedding didn't find anything better, try the original 3x3x2 approach as fallback
    if best_rank >= 68 and time.time() < deadline - 30:
        U2, V2, W2, r2 = find_rank_3x3x2()
        
        if U2 is not None:
            rank8 = 4 * r2
            U8 = np.zeros((nm, rank8))
            V8 = np.zeros((mp, rank8))
            W8 = np.zeros((pn, rank8))
            
            for b in range(4):
                offset = b * r2
                U8[:, offset:offset+r2] = U2
                
                for j in range(m):
                    for k in range(2):
                        V8[j*p + 2*b + k, offset:offset+r2] = V2[j*2 + k, :]
                
                for k in range(2):
                    for i in range(n):
                        W8[(2*b + k)*n + i, offset:offset+r2] = W2[k*n + i, :]
            
            if verify(U8, V8, W8):
                if rank8 < best_rank:
                    best_rank = rank8
                    best_decomp = (U8.copy(), V8.copy(), W8.copy())
    
    # Original random search as last resort
    if best_rank >= 68 and time.time() < deadline - 30:
        # Original random search logic (abbreviated for safety)
        # This is kept in case the block strategy fails
        for target_rank in range(best_rank - 1, 30, -1):
            if time.time() > deadline - 15:
                break
            
            max_attempts = min(5, max(3, 120 // target_rank))
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(42 + target_rank * 1000 + attempt)
                
                U_rand = np.zeros((nm, target_rank))
                V_rand = np.zeros((mp, target_rank))
                
                for c in range(target_rank):
                    u_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nm)
                    v_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=mp)
                    u_mask = np.random.random(nm) < 0.5
                    v_mask = np.random.random(mp) < 0.5
                    U_rand[:, c] = np.where(u_mask, u_vals, 0.0)
                    V_rand[:, c] = np.where(v_mask, v_vals, 0.0)
                
                K = np.zeros((nm * mp, target_rank))
                for c in range(target_rank):
                    K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
                
                try:
                    W_sol, residuals, rank_K, sv = np.linalg.lstsq(K, T_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K @ W_sol - T_flat).max()
                
                if residual < 1e-9:
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-6):
                        W_final = np.round(W_sol.T * 2) / 2
                        
                        recon = np.zeros((nm, mp, pn))
                        for c in range(target_rank):
                            recon += np.einsum('a,b,c->abc', U_rand[:,c], V_rand[:,c], W_final[:,c])
                        if np.abs(recon - T).max() < 1e-8:
                            best_rank = target_rank
                            best_decomp = (U_rand.copy(), V_rand.copy(), W_final)
                            break
            
            if best_rank < sb_rank:
                break
    
    U_out, V_out, W_out = best_decomp
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
