"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial.distance import cdist

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    We use a construction based on the D11 lattice structure. The key insight
    is that for vectors with norm 2 (squared norm 4), we need pairwise distances
    >= 2. The squared distance between two vectors u, v is ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v>.
    If ||u||^2 = ||v||^2 = 4, then ||u-v||^2 = 8 - 2<u,v>. We need 8 - 2<u,v> >= 4,
    i.e., <u,v> <= 2.
    
    For vectors with integer coordinates and squared norm 4, the possible types are:
    - Two coordinates are ±2, rest 0: <u,v> can be -4,-2,0,2,4
    - Four coordinates are ±1, rest 0: <u,v> can be -4,-2,0,2,4
    - One coordinate is ±2, one is ±1... no, that gives norm^2 = 5.
    
    Actually, let's think more carefully. Integer vectors with squared norm 4:
    Type A: (±2, 0, ..., 0) with one ±2 and rest 0 → norm^2 = 4. Count: 11*2 = 22
    Type B: (±1, ±1, 0, ..., 0) with two ±1's → norm^2 = 2. No, that's norm^2=2.
    
    Wait, norm^2 = 4 means sum of squares = 4.
    - (±2, 0, ..., 0): 11 * 2 = 22 vectors
    - (±1, ±1, ±1, ±1, 0, ..., 0): 4 positions chosen from 11, each ±1: C(11,4)*16 = 330*16 = 5280
    - (±1, ±1, 0, ..., 0): 2 positions, each ±1: C(11,2)*4 = 55*4 = 220, but norm^2 = 2, not 4.
    
    So for norm^2 = 4: Type A (22 vectors) and Type C (5280 vectors with four ±1's).
    
    For Type A vectors u = (±2 at position i), v = (±2 at position j):
    If i=j: <u,v> = ±4, distance^2 = 8-2(±4) = 0 or 16. If same sign, distance=0 (same vector).
    If i≠j: <u,v> = 0, distance^2 = 8 >= 4. OK.
    
    For Type C vectors (four ±1's):
    <u,v> = sum over common positions of u_i * v_i.
    If they share k positions, <u,v> can range from -k to k in steps of 2.
    We need <u,v> <= 2.
    
    Let's try a simpler approach: use the standard kissing number construction.
    The known kissing number in dimension 11 is 512 (or possibly higher, but 512 is known).
    
    Actually, let me use a different approach. Let's generate all integer vectors with
    small norms and use greedy selection.
    
    Let's try vectors with norm^2 = 2 (two ±1's) and norm^2 = 4.
    
    For norm^2 = 2 vectors (±1, ±1 at two positions):
    Max norm = sqrt(4) = 2 (if we include norm^2=4 vectors) or sqrt(2) if only norm^2=2.
    
    Let's just use norm^2 = 2 vectors (two ±1's). There are C(11,2)*4 = 220 such vectors.
    Max norm = sqrt(2).
    For two such vectors u, v: ||u-v||^2 = 2 + 2 - 2<u,v> = 4 - 2<u,v>.
    <u,v> is the sum of products at common positions. If they share 0 positions: <u,v>=0, dist^2=4 >= 2. OK.
    If they share 1 position: <u,v> = ±1, dist^2 = 4-2(±1) = 2 or 6. Both >= 2. OK.
    If they share 2 positions: <u,v> = ±2, dist^2 = 4-2(±2) = 0 or 8. If <u,v>=2, dist=0 (same vector).
    
    So all distinct norm^2=2 vectors are valid! That gives us 220 vectors.
    
    Can we add more? What about norm^2=4 vectors?
    If we mix norm^2=2 and norm^2=4, max norm = 2.
    For u with norm^2=2, v with norm^2=4: ||u-v||^2 = 2+4-2<u,v> = 6-2<u,v> >= 4, so <u,v> <= 1.
    <u,v> = sum of u_i*v_i. u has two ±1's, v has either one ±2 or four ±1's.
    
    This is getting complex. Let me just use the 220 norm^2=2 vectors and see if we can do better
    by also including some norm^2=4 vectors.
    
    Actually, let's try a different known construction. In dimension 11, the kissing number
    is at least 512. Let me use a lattice-based construction.
    
    The D11 lattice has vectors with even coordinate sum. The minimal vectors have norm^2=2.
    But we need integer vectors, not just lattice points.
    
    Let me just go with a greedy approach on a larger set of candidate vectors.
    """
    # Generate candidate vectors: all integer vectors with squared norm 2 or 4
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # We need: for all pairs i, j, ||c_i - c_j||^2 >= max_norm_sq
    # ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 <c_i, c_j>
    # So we need: ||c_i||^2 + ||c_j||^2 - 2<c_i, c_j> >= max_norm_sq
    
    # Greedy selection with randomization
    np.random.seed(42)
    selected = []
    remaining = list(range(len(candidates)))
    
    # Sort by norm descending (prefer larger norms for better packing)
    remaining.sort(key=lambda idx: -norms_sq[idx])
    
    while remaining:
        idx = remaining.pop(0)
        c = candidates[idx]
        c_norm_sq = norms_sq[idx]
        
        # Check against all selected vectors
        compatible = True
        for sel_idx in selected:
            s = candidates[sel_idx]
            s_norm_sq = norms_sq[sel_idx]
            dot = np.dot(c, s)
            dist_sq = c_norm_sq + s_norm_sq - 2 * dot
            if dist_sq < max_norm_sq:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
    
    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
