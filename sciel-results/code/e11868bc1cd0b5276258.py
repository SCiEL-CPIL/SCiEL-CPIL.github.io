"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Generate all candidate vectors with norm squared = 4 (two non-zero entries, each +/-1)
    # These are the shortest vectors in the D_n lattice family.
    # Total count: C(11,2) * 4 = 55 * 4 = 220
    
    candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute pairwise distances squared.
    # dist^2 = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # Since all have norm^2 = 4.
    # Minimum pairwise distance >= max norm = 2.
    # So we need dist^2 >= 4.
    # 8 - 2*v_i.v_j >= 4  =>  v_i.v_j <= 2.
    # Since dot products are integers, v_i.v_j <= 2.
    
    # Compute dot product matrix
    dots = candidates @ candidates.T
    
    # Conflict: two vectors conflict if dist^2 < 4, i.e., dots > 2
    # So conflict if dots[i][j] > 2
    # Note: dots[i][i] = 4, but we only care about i != j
    
    # We want a maximum independent set in the graph where edges exist if dots[i][j] > 2.
    # This is NP-hard in general, but let's try a greedy approach first, then improve.
    
    # First, let's try a simple greedy: sort by something and pick non-conflicting.
    # Actually, let's use a better heuristic: local search.
    
    # But first, let's see if we can use the structure.
    # The candidates come in pairs: (i,j,si,sj) and (i,j,-si,-sj) are negatives.
    # dist(v, -v) = 2*||v|| = 4, so they don't conflict.
    # dist(v, w) where w has same support but different signs:
    # e.g., v = (1,1,0...), w = (1,-1,0...)
    # dot = 1-1 = 0, dist^2 = 8, no conflict.
    # v = (1,1,0...), w = (-1,1,0...)
    # dot = -1+1 = 0, dist^2 = 8, no conflict.
    # v = (1,1,0...), w = (1,1,0...) same, dot=2, dist^2=4, OK.
    # v = (1,1,0...), w = (-1,-1,0...) negative, dot=-2, dist^2=12, OK.
    
    # What about different supports?
    # v = (1,1,0...), w = (1,0,1,0...)
    # dot = 1, dist^2 = 6, OK.
    # v = (1,1,0...), w = (1,0,-1,0...)
    # dot = 1, dist^2 = 6, OK.
    # v = (1,1,0...), w = (0,1,1,0...)
    # dot = 1, dist^2 = 6, OK.
    
    # It seems like conflicts only happen when supports are identical and signs are the same?
    # Let's check: dots > 2 means dot >= 3 (integer).
    # Max dot is 2 (when vectors are identical).
    # Wait, dots[i][i] = 4, but for i != j, max dot is 2.
    # So dots[i][j] > 2 for i != j is NEVER true!
    
    # Let me re-examine.
    # v_i.v_j for i != j:
    # If same support {a,b} and same signs: dot = 1*1 + 1*1 = 2 (or (-1)(-1)+(-1)(-1)=2)
    # If same support, different signs: dot = 0 or -2
    # If different support: dot = 0, 1, or -1
    
    # So dots[i][j] <= 2 for all i != j.
    # Therefore, dist^2 = 8 - 2*dots[i][j] >= 8 - 4 = 4.
    # So ALL pairs have dist^2 >= 4!
    
    # That means we can take ALL 220 vectors!
    
    # Wait, let me double-check with a concrete example.
    # v1 = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # v2 = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # same as v1
    # dist^2 = 0, but these are the same vector. We need distinct vectors.
    
    # v1 = (1, 1, 0, ...)
    # v2 = (-1, -1, 0, ...)
    # dot = -1-1 = -2, dist^2 = 8-2(-2) = 12 >= 4. OK.
    
    # v1 = (1, 1, 0, ...)
    # v3 = (1, -1, 0, ...)
    # dot = 1-1 = 0, dist^2 = 8 >= 4. OK.
    
    # So indeed, all 220 vectors are mutually at distance >= 2.
    
    # But wait, the problem says "minimum pairwise distance is at least the maximum vector norm".
    # Max norm = 2. Min pairwise dist >= 2.
    # For v and -v: dist = 4 >= 2. OK.
    
    # So the answer is simply all 220 vectors.
    
    return candidates
# EVOLVE-BLOCK-END
