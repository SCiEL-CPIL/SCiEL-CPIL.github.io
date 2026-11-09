import time
import random
from itertools import combinations

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute all triples as masks
TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))

# Map mask to index
MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}

# Precompute which points are in each triple
TRIPLE_POINTS = []
for t in TRIPLES:
    pts = []
    for i in range(N_POINTS):
        if t & (1 << i):
            pts.append(i)
    TRIPLE_POINTS.append(pts)

# Precompute supersets for independence number calculation
SUPERSETS = [None] * len(TRIPLES)
for i, t in enumerate(TRIPLES):
    comp = ((1 << N_POINTS) - 1) ^ t
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

# Sort masks by popcount descending for alpha calculation
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)


def _calc_alpha(triple_masks):
    """Calculate independence number using bitset marking."""
    valid = [True] * N_MASKS
    for t in triple_masks:
        idx = MASK_TO_INDEX[t]
        for s in SUPERSETS[idx]:
            valid[s] = False
    
    for s in SORTED_MASKS:
        if valid[s]:
            return POP_COUNT[s]
    return 0


def _can_add(triple_masks, new_t):
    """Check if adding new_t preserves linearity."""
    for t in triple_masks:
        if (t & new_t).bit_count() >= 2:
            return False
    return True


def _greedy_construct(seed):
    """Greedy construction with randomized ordering."""
    _random = random.Random(seed)
    triples = []
    available = TRIPLES[:]
    _random.shuffle(available)
    
    # Track which points are "used" in how many triples
    point_degree = [0] * N_POINTS
    
    for t in available:
        if _can_add(triples, t):
            triples.append(t)
            for p in TRIPLE_POINTS[MASK_TO_INDEX[t]]:
                point_degree[p] += 1
    
    return triples


def _two_opt_swap(triples, seed):
    """Systematic 2-opt: try all pairs of (remove, add) swaps."""
    _random = random.Random(seed)
    n = len(triples)
    current_alpha = _calc_alpha(triples)
    best_triples = triples[:]
    best_alpha = current_alpha
    
    # Try all possible single removals followed by best addition
    for i in range(n):
        if time.time() > 110.0:
            break
        
        # Remove triple i
        remaining = triples[:i] + triples[i+1:]
        
        # Try to add each possible triple
        for t in TRIPLES:
            if t in remaining:
                continue
            if not _can_add(remaining, t):
                continue
            
            candidate = remaining + [t]
            alpha = _calc_alpha(candidate)
            
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = candidate[:]
    
    return best_triples, best_alpha


def _pair_guided_construct(seed):
    """
    Structural approach: Build triples around point pairs.
    In a linear system, each pair of points can appear in at most one triple.
    We systematically cover pairs to minimize independence number.
    """
    _random = random.Random(seed)
    triples = []
    used_pairs = set()
    
    # All possible pairs
    all_pairs = list(combinations(range(N_POINTS), 2))
    _random.shuffle(all_pairs)
    
    for pair in all_pairs:
        if time.time() > 110.0:
            break
        
        p, q = pair
        # Find a third point r such that (p,q,r) forms a valid triple
        # and doesn't create conflicts
        candidates = []
        for r in range(N_POINTS):
            if r == p or r == q:
                continue
            t_mask = (1 << p) | (1 << q) | (1 << r)
            t_idx = MASK_TO_INDEX.get(t_mask)
            if t_idx is None:
                continue
            # Check if this triple would conflict with existing ones
            if _can_add(triples, t_mask):
                # Check if any pair in this triple is already used
                # Actually, in a linear system, we just need no two triples share 2+ points
                # So we just need _can_add to pass
                candidates.append((r, t_mask, t_idx))
        
        if candidates:
            # Choose the candidate that minimizes "conflict potential"
            # Prefer points with lower degree
            point_degree = [0] * N_POINTS
            for t in triples:
                for pt in TRIPLE_POINTS[MASK_TO_INDEX[t]]:
                    point_degree[pt] += 1
            
            # Sort by degree of r (prefer lower degree)
            candidates.sort(key=lambda x: point_degree[x[0]])
            r, t_mask, t_idx = candidates[0]
            triples.append(t_mask)
    
    return triples


def construct():
    deadline = time.time() + 110.0
    
    best_alpha = 16  # Worst case
    best_triples = []
    
    # Phase 1: Try multiple greedy constructions with different seeds
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        
        triples = _greedy_construct(seed)
        alpha = _calc_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
    
    # Phase 2: Try pair-guided construction
    for seed in range(30):
        if time.time() > deadline - 20:
            break
        
        triples = _pair_guided_construct(seed)
        alpha = _calc_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
    
    # Phase 3: 2-opt improvement on best found
    if best_triples and time.time() < deadline - 5:
        improved, improved_alpha = _two_opt_swap(best_triples, 0)
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved
    
    # Phase 4: More 2-opt iterations if time remains
    for seed in range(1, 20):
        if time.time() > deadline - 2:
            break
        
        improved, improved_alpha = _two_opt_swap(best_triples, seed)
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved
    
    # Convert to output format
    result = []
    for mask in best_triples:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
    
    return result
