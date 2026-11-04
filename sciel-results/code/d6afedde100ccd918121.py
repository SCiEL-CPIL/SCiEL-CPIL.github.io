# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None

    # Generate schoolbook solution as fallback
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=int)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=int)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Precompute the target tensor T entries
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    # We only need T for the support, but for the linear system we need T[a,b,c] for all a,b,c
    # Actually, T is 0 for most entries.
    # We can just compute T[a,b,c] on the fly or precompute a dict.
    # Given the small size (12*24*18 = 5184), precomputing is fine.
    
    T_flat = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[(a, b, c)] = 1

    # Search for lower rank
    # We try ranks from schoolbook_rank - 1 downwards
    # To keep it fast, we limit the number of random restarts per rank
    max_restarts = 50
    
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 0), -1):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Randomly initialize U and V with small integers
            # Using {-1, 0, 1} is a good start for integer decompositions
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            
            W = np.zeros((p * n, rank), dtype=float)
            success = True
            
            # Solve for W column by column? No, W is (p*n, rank).
            # The equations are coupled in r for a fixed c.
            # For each c, we solve for W[c, :]
            for c in range(p * n):
                # Construct M_c: (n*m * m*p) x rank
                # Index (a, b) -> row. r -> col.
                # M_c[(a,b), r] = U[a, r] * V[b, r]
                
                # We can vectorize this.
                # U is (nm, rank), V is (mp, rank)
                # We need outer product of U and V?
                # U[:, r] * V[:, r] gives a vector of length nm*mp?
                # Yes, if we reshape.
                
                # Let's build M_c efficiently.
                # M_c row index is (a, b).
                # M_c[(a, b), r] = U[a, r] * V[b, r]
                
                # We can compute this by iterating r or using broadcasting.
                # U shape (nm, rank) -> U[:, :, None] (nm, rank, 1)
                # V shape (mp, rank) -> V[None, :, :] (1, mp, rank)
                # This doesn't directly give (nm, mp, rank).
                
                # Let's do it simply. nm=12, mp=24, rank<=71.
                # M_c size 288 x rank.
                M_c = np.zeros((n * m * m * p, rank))
                t_c = np.zeros(n * m * m * p)
                
                # Fill t_c
                for a in range(n * m):
                    for b in range(m * p):
                        idx = a * (m * p) + b
                        if (a, b, c) in T_flat:
                            t_c[idx] = T_flat[(a, b, c)]
                        # else 0
                
                # Fill M_c
                # M_c[idx, r] = U[a, r] * V[b, r]
                # We can do this by loops or vectorized ops.
                # Vectorized:
                # U_exp = U[:, :, None] # (nm, rank, 1)
                # V_exp = V[None, :, :] # (1, mp, rank)
                # This doesn't work directly for the product U[a,r]*V[b,r] for all a,b.
                # We want M_c[a*mp+b, r] = U[a,r] * V[b,r]
                
                # Let's use a loop over a to be safe and simple, it's small.
                for a in range(n * m):
                    for r in range(rank):
                        M_c[a * (m * p) : (a + 1) * (m * p), r] = U[a, r] * V[:, r]
                
                # Solve M_c w = t_c
                # Use least squares
                try:
                    w_c, residuals, rank_c, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                except np.linalg.LinAlgError:
                    success = False
                    break
                
                # Check if solution is close to integer
                w_c_int = np.round(w_c)
                if not np.allclose(w_c, w_c_int, atol=1e-9):
                    success = False
                    break
                    
                # Check residual
                residual = np.linalg.norm(M_c @ w_c - t_c)
                if residual > 1e-9:
                    success = False
                    break
                    
                W[c, :] = w_c_int
                
            if success:
                # Verify the full decomposition
                # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = 0
                            for r in range(rank):
                                val += U[a, r] * V[b, r] * W[c, r]
                            if val != 1:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid:
                    # Check zeros?
                    # If the linear system was satisfied exactly, zeros should be satisfied too
                    # because T_flat was 0 for those entries in t_c.
                    # So if t_c was 0 and M_c w_c = 0, then the sum is 0.
                    # So we just need to check the 1s.
                    continue
                    
                # Found a valid decomposition
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break # break restart loop
                
        if found:
            # We found a lower rank. We can try to go lower, but given time,
            # we might stop or continue. The loop continues to lower ranks.
            pass

    # Return the best found
    # Convert to lists as per interface?
    # The interface says: "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't strictly say list of lists, but the seed used lists.
    # Let's return numpy arrays or lists. The seed returned lists.
    # "return {'U': U, 'V': V, 'W': W}"
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
