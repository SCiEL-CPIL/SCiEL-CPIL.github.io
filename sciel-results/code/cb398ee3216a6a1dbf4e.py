"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict


def _generate_candidates(max_weight):
    """Generate all nonzero integer vectors in R^D with L1 norm <= max_weight,
    excluding the zero vector. We use L1 as a proxy for norm; we only need
    a finite candidate pool."""
    candidates = []
    for w in range(1, max_weight + 1):
        # Generate compositions of w into D parts (non-negative)
        # Use stars and bars: C(w+D-1, D-1)
        for comp in _compositions(w, D):
            # For each nonzero composition, assign signs to nonzero entries
            nonzero_indices = [i for i in range(D) if comp[i] != 0]
            if not nonzero_indices:
                continue
            for signs in itertools.product([1, -1], repeat=len(nonzero_indices)):
                v = [0] * D
                for idx, si in zip(nonzero_indices, signs):
                    v[idx] = si * comp[idx]
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)


def _compositions(n, d):
    """Yield all compositions of n into d non-negative parts."""
    if d == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, d - 1):
            yield (first,) + rest


def _validate(vectors):
    """Check that min pairwise distance >= max norm. Returns (valid, k)."""
    if len(vectors) == 0:
        return False, 0
    norms = np.sqrt(np.sum(vectors * vectors, axis=1))
    max_norm = np.max(norms)
    # Compute all pairwise distances
    # dist^2 = |a|^2 + |b|^2 - 2*a.b
    gram = vectors @ vectors.T
    sq_norms = np.diag(gram)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    valid = min_dist >= max_norm - 1e-9
    return valid, len(vectors)


def _greedy_select(candidates, max_norm_sq, deadline):
    """Greedy selection: pick vectors one at a time, ensuring each new vector
    has distance >= sqrt(max_norm_sq) from all previously selected vectors."""
    n = len(candidates)
    if n == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # We'll use a bitset approach for efficiency
    # Start with all candidates as available
    available = np.ones(n, dtype=bool)
    selected = []
    selected_indices = []
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Sort by norm descending? Or just iterate. Let's try sorting by norm descending
    # so we pick large-norm vectors first (they impose tighter constraints).
    order = np.argsort(-norms_sq)
    order = order.tolist()
    
    # For speed, we track for each candidate the max inner product with any selected vector
    # Actually, we need: for candidate c and selected s, dist^2 = |c|^2 + |s|^2 - 2 c.s >= max_norm_sq
    # => c.s <= (|c|^2 + |s|^2 - max_norm_sq) / 2
    
    # Let's just do it directly but vectorized over candidates
    # Maintain a list of selected vectors as a matrix
    selected_matrix = np.array([], dtype=np.int64).reshape(0, D)
    
    count = 0
    for idx in order:
        if not available[idx]:
            continue
        if count % 500 == 0:
            if time.time() > deadline:
                break
        
        c = candidates[idx]
        c_norm_sq = norms_sq[idx]
        
        if len(selected_matrix) == 0:
            # First vector: just check its norm is <= max_norm_sq (it is by construction)
            selected.append(c)
            selected_indices.append(idx)
            selected_matrix = c.reshape(1, D)
            available[idx] = False
            count += 1
            continue
        
        # Check against all selected vectors
        # dist^2 = c_norm_sq + sel_norm_sq - 2 * c.s
        # We need dist^2 >= max_norm_sq for all selected s
        # => c.s <= (c_norm_sq + sel_norm_sq - max_norm_sq) / 2
        
        sel_norms_sq = np.sum(selected_matrix * selected_matrix, axis=1)
        inner_prods = selected_matrix @ c
        thresholds = (c_norm_sq + sel_norms_sq - max_norm_sq) / 2.0
        
        if np.all(inner_prods <= thresholds + 1e-9):
            selected.append(c)
            selected_indices.append(idx)
            selected_matrix = np.vstack([selected_matrix, c.reshape(1, D)])
            available[idx] = False
            count += 1
    
    if count == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    return np.array(selected, dtype=np.int64)


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin: 120s limit, use 120s deadline
    
    # Strategy: generate candidates with small L1 norm, then greedily select
    # The key insight: if we use vectors with norm 2 (L1=2, i.e., two ±1 entries),
    # then max_norm = 2, and we need pairwise distance >= 2.
    # dist^2 = 4 + 4 - 2*c.s = 8 - 2*c.s >= 4 => c.s <= 2
    # For vectors with two ±1 entries, c.s can be -2, -1, 0, 1, 2
    # So we need c.s <= 2, which is always true!
    # Wait, that means ALL such vectors are mutually compatible?
    # Let me recheck: if c and s each have two nonzero ±1 entries,
    # c.s is the sum of products of overlapping entries.
    # If they share 0 positions: c.s = 0
    # If they share 1 position: c.s = ±1
    # If they share 2 positions: c.s = ±2
    # So c.s <= 2 always holds. That means all 220 vectors from the seed are valid!
    # And indeed the seed returns 220 vectors with score 1.0.
    
    # To do better, we need vectors with larger norms but still satisfying the constraint.
    # If we use vectors with norm > 2, then max_norm > 2, and the constraint becomes tighter.
    # But we can mix: use some vectors with norm 2 and some with larger norms.
    
    # Alternative: use only vectors with the SAME norm. If all have norm r,
    # then we need dist >= r, i.e., dist^2 >= r^2.
    # dist^2 = 2r^2 - 2*c.s >= r^2 => c.s <= r^2/2
    # For integer vectors with norm^2 = r^2, c.s is an integer.
    # So we need c.s <= floor(r^2/2).
    
    # For r^2 = 2: c.s <= 1. But we showed c.s can be 2. So NOT all are valid!
    # Wait, I made an error. Let me recheck.
    # If c and s are two distinct vectors each with two ±1 entries:
    # If they are the same vector, c.s = 2 (norm^2). But we don't compare a vector to itself.
    # For distinct vectors:
    # - If they share both positions with same signs: c.s = 2, but they're the same vector.
    # - If they share both positions with one sign flipped: c.s = 0.
    # - If they share one position: c.s = ±1.
    # - If they share no positions: c.s = 0.
    # So for DISTINCT vectors, c.s <= 1. Thus c.s <= 1 = r^2/2. Valid!
    # So all 220 are indeed valid.
    
    # To get more, we need a larger candidate pool. Let's try vectors with L1 norm = 3.
    # These have norm^2 = 1+1+1 = 3 or 1+4 = 5 or 9.
    # Mixed norms complicate things. Let's stick with same-norm vectors.
    
    # Try norm^2 = 4: vectors like (2,0,0,...) or (1,1,1,1,0,...) or (1,1,0,0,...) with signs.
    # (2,0,...): norm^2 = 4. c.s for two such: if same direction, 4; if opposite, -4; if orthogonal, 0.
    # We need c.s <= 4/2 = 2. So (2,0,...) and (2,0,...) with same sign have c.s=4 > 2. Invalid pair.
    # But (2,0,...) and (-2,0,...) have c.s = -4 <= 2. OK.
    # (2,0,...) and (2,0,...) same: c.s=4 > 2. So we can't have both (2,0,...) and (2,0,...) — they're the same.
    # We can have (2,0,...) and (-2,0,...): c.s = -4 <= 2. OK.
    # (2,0,...) and (0,2,...): c.s = 0 <= 2. OK.
    # So for axis vectors with ±2: 2*11 = 22 vectors, all mutually compatible? 
    # (2,e_i) and (2,e_j) for i≠j: c.s=0. OK.
    # (2,e_i) and (-2,e_i): c.s=-4. OK.
    # So 22 axis vectors work. But that's less than 220.
    
    # Let's try a hybrid: keep the 220 seed vectors (norm^2=2) and add more.
    # If we add a vector v with norm^2 > 2, then max_norm increases to |v|.
    # For a seed vector s (norm^2=2) and new vector v (norm^2 = R^2 > 2):
    # dist^2 = 2 + R^2 - 2*s.v >= R^2 (since max_norm = R)
    # => 2 + R^2 - 2*s.v >= R^2 => 2 >= 2*s.v => s.v <= 1
    # So every new vector v must have |s.v| <= 1 for all seed vectors s.
    # And for two new vectors v1, v2 with norms R1, R2:
    # dist^2 >= max(R1, R2)^2
    # If R1 = R2 = R: v1.v2 <= R^2/2
    
    # This is getting complex. Let me just do a greedy over a larger candidate pool
    # and let the algorithm figure out the constraints.
    
    # Generate candidates with L1 norm up to 3
    # L1=1: 2*11 = 22 vectors (±e_i), norm^2=1
    # L1=2: 220 vectors (two ±1's), norm^2=2
    # L1=3: three ±1's (C(11,3)*8 = 165*8=1320) or one ±2 and one ±1 (11*10*4=440)
    # Total L1<=3: 22+220+1320+440 = 2002 vectors
    
    # Let's generate and greedily select.
    
    candidates = _generate_candidates(3)
    if time.time() > deadline - 5:
        # Not enough time, return seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    # Compute max norm squared from candidates
    norms_sq = np.sum(candidates * candidates, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Greedy select
    result = _greedy_select(candidates, max_norm_sq, deadline - 5)
    
    if len(result) == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
