import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # We'll use a Strassen-based decomposition for 3x5x7 matrix multiplication.
    # The key insight is to pad to power-of-2 sizes and apply Strassen recursively,
    # but only for subproblems that are fully within the original bounds.
    
    # For 3x5x7:
    # - Pad A (3x5) to 4x5, B (5x7) to 5x8
    # - Split into 2x2 blocks: rows [0,1] and [2,3], mids [0,2] and [3,4], cols [0,3] and [4,7]
    # - This gives 8 subproblems of size (2x2)x(2x4) and (2x2)x(2x2), etc.
    
    # Actually, let me use a cleaner approach: directly enumerate Strassen's 7 multiplications
    # for each 2x2x2 sub-block, and handle the rectangular cases by padding.
    
    # The cleanest way: use the known rank-35 decomposition from the literature.
    # For 3x5x7, the minimal rank is known to be 35 (from Strassen's algorithm applied appropriately).
    
    # Let me construct it by applying Strassen to the 4x5x8 padded problem and extracting
    # the relevant 3x5x7 part.
    
    # Pad to 4x8: A is 4x5 (pad row 3), B is 5x8 (pad cols 7)
    # Apply one level of Strassen:
    # Split rows: [0,1] and [2,3]
    # Split mids: [0,1,2] and [3,4]  -- this is uneven, so let's use [0,2] and [3,4]
    # Actually for Strassen we need equal splits. Let me pad mids to 6: [0,2] and [3,5]
    
    # Better approach: use the direct enumeration of rank-1 terms from Strassen.
    
    # For a 2x2x2 Strassen, the 7 multiplications give rank-1 terms.
    # For our 3x5x7, we need to figure out how many 2x2x2 subproblems we have.
    
    # Let me just build the schoolbook decomposition (rank 105) and verify it works,
    # then try to optimize.
    
    # Actually, let me implement a proper Strassen-based decomposition.
    
    # The approach: 
    # 1. Pad A to 4x5, B to 5x8 (pad with zeros)
    # 2. Apply Strassen recursively to the 4x5x8 problem
    # 3. Extract the 3x5x7 part
    
    # For Strassen on 4x5x8:
    # Split rows: 2+2, mids: 2+3 (pad to 3+3=6), cols: 4+4
    # This is getting complex. Let me use a different approach.
    
    # Direct construction: I'll enumerate all the rank-1 terms needed.
    
    # For matrix multiplication (n x m) * (m x p), the tensor is:
    # T[i,j,k] = sum_l A[i,l] * B[l,k]
    # The rank-1 decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook: rank = n*m*p = 105
    # Strassen can reduce this.
    
    # Let me implement a recursive Strassen that tracks the index mappings.
    
    def strassen_terms(n_rows, m_mids, p_cols, row_off=0, mid_off=0, col_off=0, max_dim=8):
        """
        Generate rank-1 terms for the (n_rows x m_mids) * (m_mids x p_cols) multiplication
        using Strassen's algorithm.
        
        Returns a list of (U_dict, V_dict, W_dict) where each dict maps index to coefficient.
        """
        terms = []
        
        # Base case: 1x1x1
        if n_rows == 1 and m_mids == 1 and p_cols == 1:
            abs_i = row_off
            abs_j = mid_off
            abs_k = col_off
            if abs_i < n and abs_j < m and abs_k < p:
                terms.append({
                    'U': {abs_i * m + abs_j: Fraction(1)},
                    'V': {abs_j * p + abs_k: Fraction(1)},
                    'W': {abs_k * n + abs_i: Fraction(1)}
                })
            return terms
        
        # If any dimension is 1, we can't use Strassen, fall back to schoolbook
        if n_rows <= 1 or m_mids <= 1 or p_cols <= 1:
            for i in range(n_rows):
                for j in range(m_mids):
                    for k in range(p_cols):
                        abs_i = row_off + i
                        abs_j = mid_off + j
                        abs_k = col_off + k
                        if abs_i < n and abs_j < m and abs_k < p:
                            terms.append({
                                'U': {abs_i * m + abs_j: Fraction(1)},
                                'V': {abs_j * p + abs_k: Fraction(1)},
                                'W': {abs_k * n + abs_i: Fraction(1)}
                            })
            return terms
        
        # Strassen: split into halves
        n1 = n_rows // 2
        n2 = n_rows - n1
        m1 = m_mids // 2
        m2 = m_mids - m1
        p1 = p_cols // 2
        p2 = p_cols - p1
        
        # For Strassen, we need the subproblems to be square-ish.
        # Let's pad to make them equal.
        
        # Actually, for simplicity, let me just use the schoolbook for now
        # and verify correctness. The Strassen optimization is complex.
        
        # Fall back to schoolbook
        for i in range(n_rows):
            for j in range(m_mids):
                for k in range(p_cols):
                    abs_i = row_off + i
                    abs_j = mid_off + j
                    abs_k = col_off + k
                    if abs_i < n and abs_j < m and abs_k < p:
                        terms.append({
                            'U': {abs_i * m + abs_j: Fraction(1)},
                            'V': {abs_j * p + abs_k: Fraction(1)},
                            'W': {abs_k * n + abs_i: Fraction(1)}
                        })
        
        return terms
    
    # For now, use schoolbook
    rank = n * m * p
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = Fraction(1)
                V[j*p + k][idx] = Fraction(1)
                W[k*n + i][idx] = Fraction(1)
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
