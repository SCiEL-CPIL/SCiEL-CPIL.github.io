import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Index mapping
    # A is n x m = 2x2. Index a = i*m + k where i in [0,n), k in [0,m)
    # B is m x p = 2x6. Index b = k*p + j where k in [0,m), j in [0,p)
    # C is n x p = 2x6. Index c = i*p + j where i in [0,n), j in [0,p)
    # Tensor T[a, b, c] = 1 if a = i*m + k and b = k*p + j and c = i*p + j, else 0.
    # Note: k must be consistent. a gives k = a % m. b gives k = b // p. c is independent of k directly but linked via i and j.
    # Actually, T[i*m+k, k*p+j, i*p+j] = 1.
    
    # Construct the full tensor T as a 3D array (or flat list) for verification
    # Size: (n*m) * (m*p) * (p*n) = 4 * 12 * 12 = 576. Very small.
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = i * p + j
                T[a, b, c] = 1

    def verify(U, V, W, rank):
        # U: (n*m) x rank
        # V: (m*p) x rank
        # W: (p*n) x rank
        # Check if sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
        # For efficiency, we can loop over a, b and compute the vector for c.
        # Or just loop over all 576 entries. 576 * rank is tiny.
        for a in range(n * m):
            for b in range(m * p):
                # Vector W_ab[c] = sum_r U[a,r] V[b,r] W[c,r]
                # Let Ua = U[a], Vb = V[b]
                # W_ab = Vb^T (element-wise Ua) W ... no.
                # W_ab[c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
                # This is (Ua * Vb) dot W[c]
                coeff = np.array([U[a, r] * V[b, r] for r in range(rank)], dtype=np.int64)
                # Now check for all c
                # Result[c] = sum_r coeff[r] * W[c, r]
                # We can do: Result = W @ coeff (since W is p*n x rank, coeff is rank)
                res = W.dot(coeff)
                # Compare with T[a, b, :]
                if not np.array_equal(res, T[a, b, :]):
                    return False
        return True

    # Known rank 21 solution (3 blocks of Strassen rank 7)
    # This is the current score 1.0 solution.
    # We want to find rank 20 or lower.
    # Theoretical lower bound?
    # For 2x2x2, rank is 7.
    # For 2x2x6, is it 3*7 = 21?
    # There are known results for matrix multiplication tensor ranks.
    # R(2,2,2) = 7.
    # R(2,2,p) ?
    # It is known that R(2,2,4) = 14? Or 13?
    # Actually, for 2x2x2, rank is 7.
    # For 2x2x4, it might be less than 14.
    # Let's try to search for a lower rank.
    
    # Strategy:
    # 1. Try to verify a hardcoded rank 20 solution if one exists?
    #    I don't know one off the top of my head.
    # 2. Perform a random search for rank 20 with integer coefficients in {-1, 0, 1}.
    #    The space is huge, but the constraints are tight.
    #    Maybe we can use a local search starting from rank 21 and merging two terms?
    
    # Let's first return the rank 21 solution to ensure validity and score 1.0.
    # Then, if time permits, search for lower rank.
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    def generate_strassen_block(k0, k1):
        # Returns U_part (4x7), V_part (12x7), W_part (12x7) for block g
        # U is shared, so we just define the 7 columns for this block
        U_block = np.zeros((4, 7), dtype=np.int64)
        V_block = np.zeros((12, 7), dtype=np.int64)
        W_block = np.zeros((12, 7), dtype=np.int64)
        
        # Strassen for 2x2x2 with B cols k0, k1 and C cols k0, k1
        # A is 2x2: A00, A01, A10, A11
        # B is 2x2: B0k0, B0k1, B1k0, B1k1
        # C is 2x2: C0k0, C0k1, C1k0, C1k1
        
        # M1 = (A00+A11)(B0k0+B1k1)
        U_block[0, 0] = 1; U_block[3, 0] = 1
        V_block[0*p+k0, 0] = 1; V_block[1*p+k1, 0] = 1
        W_block[0*p+k0, 0] = 1; W_block[1*p+k1, 0] = 1
        
        # M2 = (A10+A11)B0k0
        U_block[2, 1] = 1; U_block[3, 1] = 1
        V_block[0*p+k0, 1] = 1
        W_block[1*p+k0, 1] = 1
        
        # M3 = A00(B1k0-B1k1)
        U_block[0, 2] = 1
        V_block[1*p+k0, 2] = 1; V_block[1*p+k1, 2] = -1
        W_block[0*p+k1, 2] = 1; W_block[1*p+k1, 2] = 1
        
        # M4 = A11(B0k1-B0k0)
        U_block[3, 3] = 1
        V_block[0*p+k1, 3] = 1; V_block[0*p+k0, 3] = -1
        W_block[0*p+k0, 3] = 1; W_block[1*p+k0, 3] = 1
        
        # M5 = (A00+A01)B1k1
        U_block[0, 4] = 1; U_block[1, 4] = 1
        V_block[1*p+k1, 4] = 1
        W_block[0*p+k0, 4] = -1; W_block[0*p+k1, 4] = 1
        
        # M6 = (A10-A00)(B0k0+B1k0)
        U_block[2, 5] = 1; U_block[0, 5] = -1
        V_block[0*p+k0, 5] = 1; V_block[1*p+k0, 5] = 1
        W_block[1*p+k1, 5] = 1
        
        # M7 = (A01-A11)(B0k1+B1k1)
        U_block[1, 6] = 1; U_block[3, 6] = -1
        V_block[0*p+k1, 6] = 1; V_block[1*p+k1, 6] = 1
        W_block[0*p+k0, 6] = 1; W_block[1*p+k1, 6] = 1
        
        return U_block, V_block, W_block

    # Build Rank 21 Solution
    U21 = np.zeros((4, 21), dtype=np.int64)
    V21 = np.zeros((12, 21), dtype=np.int64)
    W21 = np.zeros((12, 21), dtype=np.int64)
    
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        Ub, Vb, Wb = generate_strassen_block(k0, k1)
        U21[:, 7*g : 7*(g+1)] = Ub
        V21[:, 7*g : 7*(g+1)] = Vb
        W21[:, 7*g : 7*(g+1)] = Wb
    
    # Verify Rank 21
    if verify(U21, V21, W21, 21):
        best_rank = 21
        best_U, best_V, best_W = U21, V21, W21
    else:
        # Should not happen, but fallback
        best_rank = 999
        best_U = best_V = best_W = None

    # Search for lower rank
    # Try Rank 20
    # Random search with small integers
    # Since the tensor is sparse and structured, maybe a greedy merge works?
    # Or just random perturbation?
    
    # Let's try a specific heuristic:
    # The rank of M_2x2x2 is 7.
    # Is there a rank 6 solution for 2x2x2? No.
    # Can we share terms between blocks?
    # The blocks are independent in B and C columns.
    # A is shared.
    # If we could combine (A00+A11) from block 1 with something from block 2?
    # Unlikely to reduce rank significantly without breaking the identity.
    
    # Let's do a random search for rank 20 with entries in {-1, 0, 1}.
    # We will sample random sparse vectors for U, V, W.
    # This is a shot in the dark, but the time budget allows some attempts.
    
    target_rank = 20
    max_attempts = 5000
    random.seed(42)
    
    found_better = False
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        U = np.random.randint(-1, 2, size=(4, target_rank))
        V = np.random.randint(-1, 2, size=(12, target_rank))
        W = np.random.randint(-1, 2, size=(12, target_rank))
        
        # Quick check: non-zero rows
        # Verify
        if verify(U, V, W, target_rank):
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            found_better = True
            break
            
    # If rank 20 not found, try rank 21 (already have)
    # Maybe try rank 20 with a more structured approach?
    # Start from rank 21, pick 2 columns, try to merge them into 1 column with adjusted coefficients?
    # This is complex.
    
    # Return the best found
    if best_rank < 999:
        # Convert to lists for the interface
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to empty/invalid? No, just return the 21 one if it was valid.
        # If 21 was invalid, return zeros?
        return {
            "U": U21.tolist(),
            "V": V21.tolist(),
            "W": W21.tolist()
        }
