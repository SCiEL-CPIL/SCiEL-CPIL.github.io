import numpy as np
import time
import random

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Check all non-zero targets and a sample of zeros
    # Targets are 1 when a==i, c==k, b==d
    # Zeros are 0 otherwise
    
    # To be fast and rigorous, we check:
    # 1. All non-zero entries (should be 1)
    # 2. All zero entries (should be 0) -- this is expensive for large tensors,
    #    but for n=2, m=3, p=6 the tensor has 2*3*3*6*2*6 = 1296 entries.
    #    This is small enough to check fully in a vectorized way or simple loop.
    
    # Vectorized approach:
    # Construct the tensor indices
    # a, b, c, d, i, k
    # U_idx = a*m + b
    # V_idx = c*p + d
    # W_idx = k*n + i
    
    # Target T is 1 if (a==i and c==k and b==d)
    
    # We can check by iterating over the 1296 terms.
    for a in range(n):
        for b in range(m):
            u_row = a * m + b
            for c in range(m):
                for d in range(p):
                    v_row = c * p + d
                    for i in range(n):
                        for k in range(p):
                            w_row = k * n + i
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            # Sum_l U[u_row, l] * V[v_row, l] * W[w_row, l]
                            s = np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # The tensor T_{a,b,c,d,i,k} = delta_{a,i} delta_{c,k} delta_{b,d}
    # This is the direct sum of tensors for each output element? No.
    # Let's look at the structure.
    # U depends on (a,b).
    # V depends on (c,d).
    # W depends on (k,i).
    
    # Key Insight:
    # The tensor is a direct sum over the "middle" indices?
    # No.
    # Let's check if it's a direct sum over d (columns of B) or k (columns of C).
    # If we fix d and k, the condition b==d and c==k implies specific rows in V.
    # Actually, notice that the indices d and k are "free" in the sense that they don't couple to a, b, c, i except via the delta.
    # Wait, d is in V, k is in W.
    # b is in U, c is in V.
    # a is in U, i is in W.
    
    # Let's check the rank of M_{2,3,6}.
    # It is known that Rank(M_{n,m,p}) is submultiplicative? No.
    # But M_{2,3,6} can be viewed as M_{2,3,3} ⊕ M_{2,3,3}?
    # If we split the 6 columns into two groups of 3.
    # Group 1: d in {0,1,2}, k in {0,1,2}.
    # Group 2: d in {3,4,5}, k in {3,4,5}.
    # 
    # In Group 1:
    # U is same for all.
    # V uses rows where d in {0,1,2}.
    # W uses rows where k in {0,1,2}.
    # 
    # In Group 2:
    # U is same for all.
    # V uses rows where d in {3,4,5}.
    # W uses rows where k in {3,4,5}.
    # 
    # Since the supports of V and W are disjoint between Group 1 and Group 2 (except U is shared, but U just provides coefficients),
    # the tensor is a direct sum of two M_{2,3,3} tensors?
    # Let's verify.
    # T = T1 + T2.
    # T1 has non-zeros only when d<3 and k<3.
    # T2 has non-zeros only when d>=3 and k>=3.
    # The supports of T1 and T2 in the V-mode are disjoint (d<3 vs d>=3).
    # The supports of T1 and T2 in the W-mode are disjoint (k<3 vs k>=3).
    # The U-mode is shared, but that's fine for direct sum if the other modes are disjoint.
    # Actually, if two tensors have disjoint supports in ANY one mode, their rank is additive?
    # Yes, Rank(T1 + T2) = Rank(T1) + Rank(T2) if supports are disjoint in one mode.
    # Here supports in V-mode are disjoint.
    # So Rank(M_{2,3,6}) = Rank(M_{2,3,3}) + Rank(M_{2,3,3}) = 2 * Rank(M_{2,3,3}).
    
    # What is Rank(M_{2,3,3})?
    # Schoolbook: 2*3*3 = 18.
    # Strassen-like?
    # M_{2,3,3} is multiplying 2x3 by 3x3.
    # This is equivalent to 3 columns of 2x3 by 3x1.
    # Rank(M_{2,3,1})?
    # M_{2,3,1} computes 2 dot products of length 3.
    # This is a bilinear form.
    # Rank(M_{2,3,1}) = 3? (Schoolbook: 3). Can it be lower?
    # No, the tensor is x_i * y_j * z_k ...
    # Actually, M_{2,3,1} is the tensor for sum_{j} A_{i,j} B_{j,k}.
    # With k=1.
    # This is 2 outputs, each is a dot product of length 3.
    # The tensor is T_{i,j,k} = delta_{i, output_i} ...
    # Let's just assume Rank(M_{2,3,3}) = 17 is a known lower bound?
    # The seed program used 17.
    # So total rank = 34.
    
    # Is there a better decomposition?
    # Maybe we can share U terms?
    # In the direct sum, U is identical for both blocks.
    # But the rank counts the number of terms.
    # If we use 17 terms for block 1 and 17 for block 2, total is 34.
    # Can we do it in fewer?
    # If we could mix the blocks, i.e., use terms that contribute to both, we might reduce rank.
    # But the supports in V and W are disjoint.
    # A term l contributes to T1 only if W[:,l] has support in k<3 AND V[:,l] has support in d<3.
    # It contributes to T2 only if W[:,l] has support in k>=3 AND V[:,l] has support in d>=3.
    # It cannot contribute to both because V and W supports are disjoint?
    # Wait, a term is a product of U_col, V_col, W_col.
    # If V_col has non-zeros in both d<3 and d>=3, and W_col has non-zeros in both k<3 and k>=3,
    # then the term contributes to both T1 and T2?
    # Let's check.
    # T_{a,b,c,d,i,k} = sum_l U_{a,b,l} V_{c,d,l} W_{k,i,l}.
    # If l has V_{c,0,l} != 0 and V_{c,3,l} != 0, and W_{0,i,l} != 0 and W_{3,i,l} != 0.
    # Then for (d=0, k=0), the term is included.
    # For (d=3, k=3), the term is included.
    # So yes, a single term can contribute to both blocks if V and W have mixed supports.
    # This means the rank is NOT necessarily additive!
    # The direct sum argument holds if we restrict to decompositions where each term is confined to one block.
    # But general decompositions can mix.
    
    # So, can we find a rank < 34 solution for M_{2,3,6}?
    # This is an open question for me right now.
    # However, the seed program found a rank 34 solution and couldn't improve.
    # Let's try to find a rank 33 or lower solution via search?
    # Search space is huge.
    
    # Alternative:
    # Maybe M_{2,3,3} has rank 16?
    # If Rank(M_{2,3,3}) = 16, then Rank(M_{2,3,6}) <= 32.
    # Let's try to find a rank 16 solution for M_{2,3,3}.
    
    # If that fails, we stick with 34.
    
    # Let's implement a search for Rank(M_{2,3,3}) = 16.
    # n=2, m=3, p=3.
    # U: 6x16, V: 9x16, W: 6x16.
    # 6*16 + 9*16 + 6*16 = 96 + 144 + 96 = 336 variables.
    # Constraints: 2*3*3*3*2*3 = 324 equations.
    # Overdetermined? 336 vars, 324 eqs. Likely solvable.
    
    # We will use a least squares solver or iterative refinement.
    # Initialize with the rank 17 solution and set the 17th column to 0? No, that's rank 16 but we need to adjust the others.
    # Better: Random init and Gauss-Newton / Levenberg-Marquardt.
    
    # Since SciPy is available, we can use scipy.optimize.least_squares.
    
    from scipy.optimize import least_squares
    
    def m233_residuals(params, n=2, m=3, p=3):
        # params is a flat vector of length (6+9+6)*16 = 336
        # Split into U, V, W
        r = 16
        U_size = 6 * r
        V_size = 9 * r
        W_size = 6 * r
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size+V_size]
        W_flat = params[U_size+V_size:]
        
        U = U_flat.reshape(6, r)
        V = V_flat.reshape(9, r)
        W = W_flat.reshape(6, r)
        
        residuals = []
        for a in range(n):
            for b in range(m):
                u_row = a * m + b
                for c in range(m):
                    for d in range(p):
                        v_row = c * p + d
                        for i in range(n):
                            for k in range(p):
                                w_row = k * n + i
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                s = np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
                                residuals.append(s - target)
        return np.array(residuals)
    
    # Try to solve for Rank 16
    found_m233_16 = False
    U233_16 = None
    V233_16 = None
    W233_16 = None
    
    # Use the seed's rank 17 solution as a warm start?
    # Or random start?
    # Random start might get stuck in local minima.
    # Let's try a few restarts.
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        U_init = np.random.randn(6, 16) * 0.5
        V_init = np.random.randn(9, 16) * 0.5
        W_init = np.random.randn(6, 16) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(m233_residuals, params_init, max_nfev=2000)
            if result.success and result.cost < 1e-10:
                # Snap to half-integers
                U_sol = result.x[:96].reshape(6, 16)
                V_sol = result.x[96:240].reshape(9, 16)
                W_sol = result.x[240:].reshape(6, 16)
                
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if verify_decomposition(U_snap, V_snap, W_snap, 2, 3, 3):
                    found_m233_16 = True
                    U233_16 = U_snap
                    V233_16 = V_snap
                    W233_16 = W_snap
                    break
        except Exception:
            pass
    
    if found_m233_16:
        # Construct M_{2,3,6} from two M_{2,3,3} blocks
        # Total rank 32
        U_comb = np.zeros((6, 32))
        V_comb = np.zeros((18, 32))
        W_comb = np.zeros((12, 32))
        
        # Block 1: d in {0,1,2}, k in {0,1,2}
        U_comb[:, :16] = U233_16
        for c in range(3):
            for d in range(3):
                V_comb[c * 6 + d, :16] = V233_16[c * 3 + d]
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :16] = W233_16[k * 2 + i]
                
        # Block 2: d in {3,4,5}, k in {3,4,5}
        U_comb[:, 16:] = U233_16
        for c in range(3):
            for d in range(3, 6):
                d_local = d - 3
                V_comb[c * 6 + d, 16:] = V233_16[c * 3 + d_local]
        for k in range(3, 6):
            k_local = k - 3
            for i in range(2):
                W_comb[k * 2 + i, 16:] = W233_16[k_local * 2 + i]
                
        if verify_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Fallback: Rank 34 solution from seed
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
