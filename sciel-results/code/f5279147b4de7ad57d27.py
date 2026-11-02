import time
import numpy as np
from itertools import product

def construct():
    """
    Attempt to find a rank-27 exact decomposition for 2x3 * 3x5 matrix multiplication
    using finite field (GF(7)) search, falling back to rank-28 if not found.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s limit with margin
    
    n, m, p = 2, 3, 5
    
    def build_rank_28():
        """Build the known rank-28 solution."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        # Block 1: j=0, all k — schoolbook 2x1x5, rank 10
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        def add_strassen_2x2x2(base_j, base_k):
            nonlocal col
            # M1
            U[0 * m + base_j, col] = 1
            U[1 * m + (base_j + 1), col] = 1
            V[base_j * p + base_k, col] = 1
            V[(base_j + 1) * p + (base_k + 1), col] = 1
            W[base_k * n + 0, col] = 1
            W[(base_k + 1) * n + 1, col] = 1
            col += 1
            # M2
            U[1 * m + base_j, col] = 1
            U[1 * m + (base_j + 1), col] = 1
            V[base_j * p + base_k, col] = 1
            W[base_k * n + 1, col] = 1
            W[(base_k + 1) * n + 1, col] = -1
            col += 1
            # M3
            U[0 * m + base_j, col] = 1
            V[base_j * p + (base_k + 1), col] = 1
            V[(base_j + 1) * p + (base_k + 1), col] = -1
            W[(base_k + 1) * n + 0, col] = 1
            W[(base_k + 1) * n + 1, col] = 1
            col += 1
            # M4
            U[1 * m + (base_j + 1), col] = 1
            V[(base_j + 1) * p + base_k, col] = 1
            V[base_j * p + base_k, col] = -1
            W[base_k * n + 0, col] = 1
            W[base_k * n + 1, col] = 1
            col += 1
            # M5
            U[0 * m + base_j, col] = 1
            U[0 * m + (base_j + 1), col] = 1
            V[(base_j + 1) * p + (base_k + 1), col] = 1
            W[base_k * n + 0, col] = -1
            W[(base_k + 1) * n + 0, col] = 1
            col += 1
            # M6
            U[1 * m + base_j, col] = 1
            U[0 * m + base_j, col] = -1
            V[base_j * p + base_k, col] = 1
            V[base_j * p + (base_k + 1), col] = 1
            W[(base_k + 1) * n + 1, col] = 1
            col += 1
            # M7
            U[0 * m + (base_j + 1), col] = 1
            U[1 * m + (base_j + 1), col] = -1
            V[(base_j + 1) * p + base_k, col] = 1
            V[(base_j + 1) * p + (base_k + 1), col] = 1
            W[base_k * n + 0, col] = 1
            col += 1
        
        add_strassen_2x2x2(1, 0)
        
        for i in range(n):
            for j_local in range(2):
                j = j_local + 1
                U[i * m + j, col] = 1
                V[j * p + 2, col] = 1
                W[2 * n + i, col] = 1
                col += 1
        
        add_strassen_2x2x2(1, 3)
        
        assert col == rank
        return U.tolist(), V.tolist(), W.tolist()
    
    def verify_decomposition(U, V, W, mod=None):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if mod:
                        if total % mod != 1 % mod:
                            return False
                    else:
                        if total != 1:
                            return False
        return True
    
    # Try to find rank-27 solution via structured search
    # Strategy: modify the rank-28 construction to see if we can save one rank
    # by adjusting the schoolbook blocks for k=2
    
    # Alternative: try a different partition
    # What if we use Strassen for j∈{1,2}, k∈{2,3} and schoolbook for k=4?
    # That would be: 10 + 7 + 7 + 2 = 26? No, 2x1x1 schoolbook is rank 2.
    # Let me try: 10 (j=0) + 7 (Strassen j={1,2},k={0,1}) + 7 (Strassen j={1,2},k={2,3}) + 2 (schoolbook j={1,2},k=4) = 26
    # But wait, that's 2x2x1 schoolbook which is rank 4, not 2.
    # 2x1x1 schoolbook would be rank 2, but we have 2x2x1 which is rank 4.
    
    # Let me try: 10 + 7 + 7 + 4 = 28 (same as before, just different k partition)
    # Or: 10 + 7 + 4 + 7 = 28
    
    # What about using a different approach for the 2x2x5 block?
    # The 2x2x5 matrix multiplication tensor has rank at most 7+4+7=18 or 7+7+4=18
    # Is there a rank-17 solution for 2x2x5?
    
    # Let me try to find a rank-17 solution for 2x2x5 over GF(7)
    # 2x2x5: n=2, m=2, p=5, so U is 4x17, V is 10x17, W is 10x17
    # Total unknowns: 17*(4+10+10) = 408, equations: 20
    
    # This is highly underdetermined, so solutions should exist. Let me try a randomized approach.
    
    def search_rank_17_2x2x5(seed=42):
        """Try to find rank-17 solution for 2x2x5 over GF(7)."""
        rng = np.random.RandomState(seed)
        mod = 7
        n2, m2, p2 = 2, 2, 5
        
        # We need U (4x17), V (10x17), W (10x17) over GF(7)
        # such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        
        # Strategy: fix U and W, solve for V, then check
        # Or use a greedy approach
        
        # Let me try: initialize with the known rank-18 solution and try to merge two terms
        # Rank-18: Strassen(7) + schoolbook(4) + Strassen(7) for k={0,1},{2},{3,4}
        # Or: Strassen(7) + Strassen(7) + schoolbook(4) for k={0,1},{2,3},{4}
        
        # Let me try the latter: Strassen for k={0,1}, Strassen for k={2,3}, schoolbook for k=4
        # That's 7+7+4 = 18
        
        # To get rank 17, I need to merge one schoolbook term with something
        # Schoolbook for k=4: terms (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
        # Each has U[i*2+j, r]=1, V[j*5+4, r]=1, W[4*2+i, r]=1
        
        # Can I absorb one of these into a Strassen term by adjusting coefficients?
        
        # Let me try a direct randomized search
        for attempt in range(500):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = rng.randint(0, mod, size=(4, 17)) % mod
            V = rng.randint(0, mod, size=(10, 17)) % mod
            W = rng.randint(0, mod, size=(10, 17)) % mod
            
            # Check if this works
            ok = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        total = 0
                        for r in range(17):
                            total += U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r]
                        if total % mod != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return U, V, W
        
        return None
    
    # Try to find rank-27 solution
    found = False
    result = None
    
    if time.time() < deadline - 10:
        # Try to find rank-17 for 2x2x5, then combine with rank-10 for 2x1x5
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            sol = search_rank_17_2x2x5(seed)
            if sol is not None:
                U22, V22, W22 = sol
                # Build full rank-27 solution
                rank = 27
                U_full = np.zeros((6, 27), dtype=int)
                V_full = np.zeros((15, 27), dtype=int)
                W_full = np.zeros((10, 27), dtype=int)
                
                # First 10 terms: j=0 schoolbook
                for i in range(2):
                    for k in range(5):
                        U_full[i*3+0, i*5+k] = 1
                        V_full[0*5+k, i*5+k] = 1
                        W_full[k*2+i, i*5+k] = 1
                
                # Next 17 terms: 2x2x5 solution
                for r in range(17):
                    for i in range(2):
                        for j in range(2):
                            U_full[(i)*3+(j+1), 10+r] = U22[i*2+j, r]
                    for j in range(2):
                        for k in range(5):
                            V_full[(j+1)*5+k, 10+r] = V22[j*5+k, r]
                    for k in range(5):
                        for i in range(2):
                            W_full[k*2+i, 10+r] = W22[k*2+i, r]
                
                # Verify over GF(7)
                if verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist(), mod=7):
                    # Now need to verify over integers/rationals
                    # The GF(7) solution may not lift to integers directly
                    # But if entries are in {0,1,...,6}, we can check over integers
                    if verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                        found = True
                        result = (U_full.tolist(), V_full.tolist(), W_full.tolist())
                        break
                    else:
                        # Try to find a rational lift
                        # For now, skip and try next seed
                        pass
    
    if not found:
        # Fall back to rank-28
        U, V, W = build_rank_28()
        result = (U, V, W)
    
    U_list, V_list, W_list = result
    
    # Final verification
    if not verify_decomposition(U_list, V_list, W_list):
        # Emergency fallback: rebuild rank-28
        U_list, V_list, W_list = build_rank_28()
    
    return {"U": U_list, "V": V_list, "W": W_list}
