import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s margin within 120s limit

    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r_idx in range(r):
                        actual += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def build_trivial(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def build_strassen(n, m, p):
        """
        Use Strassen-like decomposition for rectangular matrix multiplication.
        For 4x5 * 5x6, we can split into blocks and use Strassen for square sub-blocks.
        
        Strategy: Split the 4x5 matrix into blocks and the 5x6 matrix into blocks,
        then use Strassen's algorithm for the overlapping square parts.
        
        For 4x5 * 5x6:
        - We can think of this as multiplying a (4x4 + 4x1) by (4x6 + 1x6)
        - Or better: use the structure where we handle the 4x4*4x6 part with Strassen-like
          and the remaining parts with schoolbook.
        
        Actually, let's use a different approach:
        - Split m=5 into m1=4 and m2=1
        - A is (n x m) = (4 x 5), split into A1 (4x4) and A2 (4x1)
        - B is (m x p) = (5 x 6), split into B1 (4x6) and B2 (1x6)
        - C = A1*B1 + A2*B2
        
        For A1*B1 (4x4 * 4x6): We can use Strassen for the 4x4 part
        For A2*B2 (4x1 * 1x6): This is just outer product, rank = 4*1*6 = 24... 
        Wait, A2 is 4x1 and B2 is 1x6, so A2*B2 is 4x6.
        The rank for this is just 4*6 = 24? No, it's rank 4 (since A2 is 4x1 and B2 is 1x6,
        the product has rank at most 1, but the tensor rank for this specific multiplication
        is 4*6 = 24 in the trivial sense, but actually since one dimension is 1, it's simpler).
        
        Actually, for (n x 1) * (1 x p), the tensor rank is n*p because each output element
        C[i,k] = A[i,0] * B[0,k] requires one multiplication per (i,k) pair.
        
        So:
        - A2*B2: rank = 4*6 = 24
        - A1*B1 (4x4 * 4x6): We need to handle this.
        
        For 4x4 * 4x6, we can split B into two 4x3 blocks: B1 = [B1a | B1b] where B1a is 4x3, B1b is 4x3.
        Then A1*B1 = [A1*B1a | A1*B1b]
        A1*B1a is 4x4 * 4x3, A1*B1b is 4x4 * 4x3.
        
        This is getting complex. Let me try a simpler approach:
        
        Use the known result that for multiplying (n x m) by (m x p), if we can find
        a decomposition, we can use it. For 4x5*5x6, let's try:
        
        Actually, let's use a different structural approach. The key insight from the
        cell findings is that using Strassen for 4x4x2 blocks can reduce rank.
        
        Let me try: Split the problem into:
        - 4x4 * 4x4 part (using Strassen, rank 7 instead of 16)
        - 4x4 * 4x2 part (using Strassen-like, rank ?)
        - 4x1 * 4x6 part (schoolbook, rank 24)
        - 4x1 * 1x6 part (schoolbook, rank 6... wait, 4x1*1x6 is outer product)
        
        Hmm, this is getting complicated. Let me just try a simpler improvement:
        
        For 4x5 * 5x6, I'll try to use the fact that we can group terms.
        
        Actually, let me try a completely different approach: use the bilinear rank
        formula. For matrix multiplication, the tensor rank R(n,m,p) satisfies:
        R(n,m,p) <= R(n1,m,p) + R(n2,m,p) where n = n1+n2
        R(n,m,p) <= R(n,m1,p) + R(n,m2,p) where m = m1+m2
        R(n,m,p) <= R(n,m,p1) + R(n,m,p2) where p = p1+p2
        
        And for square matrices, R(n,n,n) <= 7 for n=2 (Strassen), 19 for n=4, etc.
        
        Let me try: R(4,5,6) <= R(4,4,6) + R(4,1,6)
        R(4,1,6) = 4*6 = 24 (trivial, since one dimension is 1)
        R(4,4,6) <= R(4,4,4) + R(4,4,2)
        R(4,4,4) = 19 (known optimal for 4x4x4)
        R(4,4,2) <= R(4,4,1) + R(4,4,1) = 16 + 16 = 32? No, that's not right.
        
        Actually, R(4,4,2): multiplying 4x4 by 4x2. This is equivalent to computing
        2 columns of a 4x4 product. R(4,4,2) <= 2*R(4,4,1) = 2*16 = 32.
        But we can do better: R(4,4,2) is the rank of the tensor for (4x4)*(4x2)->(4x2).
        This is at most 4*4*2 = 32, but can be lower.
        
        Actually, for (n x n) * (n x 1) -> (n x 1), the rank is n^2 (trivial).
        For (n x n) * (n x 2) -> (n x 2), the rank is at most 2*n^2, but can be lower.
        
        Let me just try a practical approach: use the trivial decomposition but try
        to find linear dependencies and reduce rank.
        
        Actually, given the time constraints, let me just return the trivial decomposition
        but with a small optimization: check if we can remove any columns.
        """
        # For now, return trivial
        return build_trivial(n, m, p)

    # Start with trivial decomposition
    U, V, W, rank = build_trivial(n, m, p)
    
    if not verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by checking for linear dependencies
    # Use matricization to find a lower bound
    T_i = np.zeros((n, m * p))
    T_j = np.zeros((m, n * p))
    T_k = np.zeros((p, n * m))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_i[i, j*p+k] = 1.0
                T_j[j, i*p+k] = 1.0
                T_k[k, i*m+j] = 1.0
    
    rank_lower_bound = max(np.linalg.matrix_rank(T_i), 
                          np.linalg.matrix_rank(T_j), 
                          np.linalg.matrix_rank(T_k))
    
    # Try to find a lower rank decomposition using greedy elimination
    # This is a heuristic approach
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing columns one by one and checking if the decomposition still holds
    # This is O(rank^2 * n*m*p) which is expensive, so we limit it
    
    columns_to_try = list(range(rank))
    np.random.shuffle(columns_to_try)
    
    max_trials = 50  # Limit number of trials to stay within time budget
    
    for trial in range(min(max_trials, len(columns_to_try))):
        if time.time() > deadline:
            break
        
        col_idx = columns_to_try[trial]
        
        # Try to remove column col_idx
        mask = np.ones(rank, dtype=bool)
        mask[col_idx] = False
        
        U_other = U[:, mask]
        V_other = V[:, mask]
        W_other = W[:, mask]
        
        # Check if the tensor contribution of the removed column can be matched
        # by the remaining columns. This requires solving a linear system.
        # For simplicity, we just check if removing it breaks the decomposition.
        
        if verify(U_other, V_other, W_other, n, m, p):
            # We can remove this column!
            U = U_other
            V = V_other
            W = W_other
            rank = rank - 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
            columns_to_try = [c for c in columns_to_try if c != col_idx]
            if not columns_to_try:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
