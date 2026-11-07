"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_EDGES = P['edges']
_MINV = P['minimum_vertices']

# Build adjacency bitsets (full graph) for fast induced-degree computation.
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)

def _score_mask(mask):
    """Number of distinct induced degrees for vertices in mask."""
    if bin(mask).count('1') < _MINV:
        return -1
    degs = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        degs.add((_ADJ[v] & mask).bit_count())
        m ^= lsb
    return len(degs)

def _mask_to_list(mask):
    """Convert bitmask to sorted list of vertex indices."""
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
    return result

def _list_to_mask(lst):
    """Convert list of vertex indices to bitmask."""
    mask = 0
    for v in lst:
        mask |= (1 << v)
    return mask

def _random_mask(rng, minv):
    """Generate a random valid mask."""
    k = rng.randint(minv, _N)
    verts = rng.sample(range(_N), k)
    mask = 0
    for v in verts:
        mask |= (1 << v)
    return mask

def _hill_climb(mask, score, rng, deadline, best_mask, best_score, max_order=2):
    """Greedy hill climbing with 1-move and 2-move neighborhoods."""
    while time.time() < deadline:
        improved = False
        
        # Try 1-moves
        for v in range(_N):
            bit = 1 << v
            new_mask = mask ^ bit
            if bin(new_mask).count('1') < _MINV:
                continue
            s = _score_mask(new_mask)
            if s > score:
                mask = new_mask
                score = s
                improved = True
                if score > best_score:
                    best_score = score
                    best_mask = mask
                break
        
        if improved:
            continue
        
        # Try 2-moves
        found_2 = False
        for v1 in range(_N):
            if time.time() >= deadline:
                break
            for v2 in range(v1 + 1, _N):
                bit1 = 1 << v1
                bit2 = 1 << v2
                new_mask = mask ^ bit1 ^ bit2
                if bin(new_mask).count('1') < _MINV:
                    continue
                s = _score_mask(new_mask)
                if s > score:
                    mask = new_mask
                    score = s
                    improved = True
                    found_2 = True
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                    break
            if found_2:
                break
        
        if not improved:
            break
    
    return mask, score, best_mask, best_score

def _simulated_annealing(mask, score, rng, deadline, best_mask, best_score):
    """Simulated annealing to escape local optima."""
    T = 2.0
    T_min = 0.01
    cooling = 0.98
    
    while time.time() < deadline and T > T_min:
        # Generate a random neighbor
        v = rng.randrange(_N)
        new_mask = mask ^ (1 << v)
        if bin(new_mask).count('1') < _MINV:
            continue
        
        new_score = _score_mask(new_mask)
        delta = new_score - score
        
        if delta > 0:
            mask = new_mask
            score = new_score
            if score > best_score:
                best_score = score
                best_mask = mask
        else:
            # Accept with probability exp(delta / T)
            if rng.random() < math.exp(delta / T):
                mask = new_mask
                score = new_score
        
        T *= cooling
    
    return mask, score, best_mask, best_score

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    rng = random.Random(12345)

    best_mask = None
    best_score = 0

    def consider(mask):
        nonlocal best_mask, best_score
        s = _score_mask(mask)
        if s > best_score:
            best_score = s
            best_mask = mask

    # Phase 0: Exhaustive search over all size-8 subsets (C(17,8) = 24310)
    for combo in combinations(range(_N), _MINV):
        if time.time() >= deadline:
            break
        mask = _list_to_mask(combo)
        s = _score_mask(mask)
        if s > best_score:
            best_score = s
            best_mask = mask

    # Phase 1: Many diverse random initializations
    for _ in range(200):
        if time.time() >= deadline:
            break
        consider(_random_mask(rng, _MINV))

    # Phase 2: Hill climbing from best
    current_mask = best_mask
    current_score = best_score
    current_mask, current_score, best_mask, best_score = _hill_climb(
        current_mask, current_score, rng, deadline, best_mask, best_score
    )

    # Phase 3: Simulated annealing from best
    if time.time() < deadline:
        current_mask, current_score, best_mask, best_score = _simulated_annealing(
            best_mask, best_score, rng, deadline, best_mask, best_score
        )

    # Phase 4: More random restarts with hill climbing
    restarts = 0
    while time.time() < deadline and restarts < 50:
        # Random restart
        current_mask = _random_mask(rng, _MINV)
        current_score = _score_mask(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        # Hill climbing
        current_mask, current_score, best_mask, best_score = _hill_climb(
            current_mask, current_score, rng, deadline, best_mask, best_score
        )
        restarts += 1

    # Phase 5: Final polish with simulated annealing from best
    if time.time() < deadline:
        _simulated_annealing(best_mask, best_score, rng, deadline, best_mask, best_score)

    # Fallback: ensure we always have a valid result.
    if best_mask is None:
        best_mask = _list_to_mask(list(range(_MINV)))
        best_score = _score_mask(best_mask)

    # Convert mask to a list of plain Python ints.
    result = _mask_to_list(best_mask)
    result.sort()
    return list(result)
# EVOLVE-BLOCK-END
