import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p # 96
    
    def check_decomposition_444(U, V, W):
        """
        Check if sum_l U[i*4+j, l] V[j*4+k, l] W[k*4+i, l] == 1 for all i,j,k in 0..3
        U: (16, r), V: (16, r), W: (16, r)
        """
        r = U.shape[1]
        U_3d = U.reshape(4, 4, r)
        V_3d = V.reshape(4, 4, r)
        W_3d = W.reshape(4, 4, r)
        
        # T[i, j, k] = sum_l U[i, j, l] V[j, k, l] W[k, i, l]
        T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T, 1.0, atol=1e-9)

    def refine_444(U, V, W, max_iters=50):
        """
        ALS refinement for 4x4x4.
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(4):
                W_i_mat = np.array([W[k*4 + i, :] for k in range(4)]) # (4, r)
                for j in range(4):
                    V_j = V[j*4:(j+1)*4, :] # (4, r)
                    A = V_j * W_i_mat # (4, r)
                    b = np.ones(4)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        U_new[i*4 + j, :] = x
                    except:
                        U_new[i*4 + j, :] = U[i*4 + j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(4):
                for k in range(4):
                    A = np.zeros((4, r))
                    for i in range(4):
                        A[i, :] = U[i*4 + j, :] * W[k*4 + i, :]
                    b = np.ones(4)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        V_new[j*4 + k, :] = x
                    except:
                        V_new[j*4 + k, :] = V[j*4 + k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(4):
                for i in range(4):
                    A = np.zeros((4, r))
                    for j in range(4):
                        A[j, :] = U[i*4 + j, :] * V[j*4 + k, :]
                    b = np.ones(4)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        W_new[k*4 + i, :] = x
                    except:
                        W_new[k*4 + i, :] = W[k*4 + i, :]
            W = W_new
            
            if check_decomposition_444(U, V, W):
                # Snap to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_decomposition_444(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_decomposition_444(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    else:
                        return None
        return None

    def get_strassen_4x4x4():
        """
        Construct Strassen 4x4x4 decomposition (Rank 49).
        """
        # Base case: 2x2x2 Strassen
        # U, V, W for 2x2x2
        # Indices: i in 0,1; j in 0,1; k in 0,1
        # U[i,j,l], V[j,k,l], W[k,i,l]
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        # A00: i=0,j=0. A11: i=1,j=1.
        # B00: j=0,k=0. B11: j=1,k=1.
        # C00 += M1, C11 += M1
        U2[0, 0] = 1; U2[3, 0] = 1 # i=0,j=0 and i=1,j=1
        V2[0, 0] = 1; V2[3, 0] = 1 # j=0,k=0 and j=1,k=1
        W2[0, 0] = 1; W2[3, 0] = 1 # k=0,i=0 and k=1,i=1
        
        # M2: (A10+A11)B00
        # A10: i=1,j=0. A11: i=1,j=1.
        # B00: j=0,k=0.
        # C10 += M2
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1 # k=1? No, C10 is i=1, k=0. W[k,i]. W[0,1].
        # Wait, C10 is i=1, k=0. So W[0, 1] should be 1.
        # My indexing for W2 is [k*2 + i].
        # C10 -> i=1, k=0 -> W[0*2+1] = W[1].
        # Let's re-verify Strassen formulas carefully.
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's build the 49 terms by expanding each Mk.
        # This is complex to hardcode manually.
        # Instead, I will use the recursive structure in code.
        
        pass # Placeholder

    # Since hardcoding Strassen 4x4x4 is error-prone, I will rely on the search.
    # But I need a fallback better than 96.
    # I will try to find a rank 49 solution via search first.
    
    best_result = None
    best_rank = rank_sb
    
    # Try to find a 4x4x4 decomposition with rank < 48
    # Search ranks from 47 down to 40 (Strassen is 49, so 47 is a good target)
    for r in range(47, 39, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        trial_count = 0
        while time.time() < deadline - 20:
            if found:
                break
            if trial_count > 100:
                break
                
            np.random.seed(r * 100 + trial_count)
            # Integer initialization
            U = np.random.choice([-1, 0, 1], size=(16, r))
            V = np.random.choice([-1, 0, 1], size=(16, r))
            W = np.random.choice([-1, 0, 1], size=(16, r))
            
            result = refine_444(U, V, W, max_iters=20)
            if result is not None:
                U_res, V_res, W_res = result
                # Convert 4x4x4 solution to 3x4x8
                # U_348: First 12 rows of U_res (i=0,1,2)
                U_348 = U_res[:12, :]
                # V_348: Interleave V_res rows?
                # V_res is 4x4. Rows j*4+k.
                # V_348 is 4x8. Rows j*8+k.
                # For j in 0..3:
                #   V_348[j*8 : j*8+4] = V_res[j*4 : j*4+4] (k=0..3)
                #   V_348[j*8+4 : j*8+8] = V_res[j*4 : j*4+4] (k=4..7 maps to 0..3)
                V_348 = np.zeros((32, r))
                for j in range(4):
                    V_348[j*8 : j*8+4, :] = V_res[j*4 : j*4+4, :]
                    V_348[j*8+4 : j*8+8, :] = V_res[j*4 : j*4+4, :]
                
                # W_348: Stack W_res
                # W_res is 4x4. Rows k*4+i.
                # W_348 is 8x3. Rows k*3+i.
                # For k=0..3: W_348[k*3 : k*3+3] = W_res[k*4 : k*4+4] (i=0..2)
                # For k=4..7: W_348[k*3 : k*3+3] = W_res[(k-4)*4 : (k-4)*4+4] (i=0..2)
                W_348 = np.zeros((24, r))
                for k in range(4):
                    W_348[k*3 : k*3+3, :] = W_res[k*4 : k*4+3, :]
                    W_348[(k+4)*3 : (k+4)*3+3, :] = W_res[k*4 : k*4+3, :]
                
                # Verify 3x4x8
                def check_348(U, V, W):
                    r = U.shape[1]
                    U_3d = U.reshape(3, 4, r)
                    V_3d = V.reshape(4, 8, r)
                    W_3d = W.reshape(8, 3, r)
                    T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
                    return np.allclose(T, 1.0, atol=1e-9)
                
                if check_348(U_348, V_348, W_348):
                    return {
                        "U": U_348.tolist(),
                        "V": V_348.tolist(),
                        "W": W_348.tolist()
                    }
            trial_count += 1

    # Fallback to schoolbook
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
