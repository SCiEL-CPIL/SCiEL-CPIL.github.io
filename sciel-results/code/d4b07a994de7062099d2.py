"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    def to_list(mat):
        return mat.tolist()
    
    def verify_full(U, V, W):
        """Verify exact decomposition over all entries."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check all required entries (i,j,k) -> should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] 
                                for r in range(rank))
                    if abs(actual - 1) > 1e-9:
                        return False
        
        # Check all forbidden entries -> should be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue  # Already checked
                    actual = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(actual) > 1e-9:
                        return False
        return True
    
    def verify_mod3(U, V, W):
        """Verify decomposition modulo 3 (for search guidance)."""
        rank = U.shape[1]
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(int(U[i * m + j, r]) * int(V[j * p + k, r]) * int(W[k * n + i, r]) 
                                for r in range(rank))
                    if actual % 3 != 1 % 3:
                        return False
        # Check some forbidden entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue
                    actual = sum(int(U[a, r]) * int(V[b, r]) * int(W[c, r]) for r in range(rank))
                    if actual % 3 != 0:
                        return False
        return True
    
    # Schoolbook decomposition as baseline
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Search for lower rank using GF(3) guided approach
    def search_lower_rank(target_rank, max_attempts=5000):
        """Search for a rank decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            W = np.zeros((p * n, target_rank), dtype=np.int64)
            
            # Each rank-1 component: choose support and values in {-1, 0, 1}
            for r_idx in range(target_rank):
                # Random support sizes
                u_size = np.random.randint(1, 4)
                v_size = np.random.randint(1, 4)
                w_size = np.random.randint(1, 4)
                
                u_pos = np.random.choice(n * m, size=u_size, replace=False)
                v_pos = np.random.choice(m * p, size=v_size, replace=False)
                w_pos = np.random.choice(p * n, size=w_size, replace=False)
                
                for pos in u_pos:
                    U[pos, r_idx] = np.random.choice([-1, 1])
                for pos in v_pos:
                    V[pos, r_idx] = np.random.choice([-1, 1])
                for pos in w_pos:
                    W[pos, r_idx] = np.random.choice([-1, 1])
            
            # Quick mod-3 check first
            if not verify_mod3(U, V, W):
                continue
            
            # Full verification
            if verify_full(U, V, W):
                return U, V, W
        
        return None
    
    # Try to find rank 41 decomposition
    result = search_lower_rank(41, max_attempts=3000)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try rank 40
    result = search_lower_rank(40, max_attempts=2000)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # If all searches fail, return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
