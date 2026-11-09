"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as bitmasks for fast degree computation
_N = P['n']
_adj = [0] * _N
for _u, _v in P['edges']:
    _adj[_u] |= (1 << _v)
    _adj[_v] |= (1 << _u)

# Precompute best subset via exhaustive search over all masks of size >= 8
# There are 2^17 = 131072 masks; filtering to size >= 8 leaves ~64k candidates.
# This is fast enough to do at module load time.

def _score_mask(mask):
    """Return the number of distinct degrees in the induced subgraph defined by mask."""
    # Count bits
    # Use a loop to get degrees
    degs = []
    m = mask
    while m:
        # Get lowest set bit
        lb = m & (-m)
        v = lb.bit_length() - 1
        # Degree of v within the induced subgraph: popcount(adj[v] & mask)
        deg = (adj[v] & mask).bit_count()
        degs.append(deg)
        m ^= lb
    return len(set(degs))

# Build adjacency list as bitmasks
adj = [0] * _N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _score_mask(mask):
    """Return the number of distinct degrees in the induced subgraph defined by mask."""
    degs = set()
    m = mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        deg = (adj[v] & mask).bit_count()
        degs.add(deg)
        m ^= lb
    return len(degs)

# Exhaustive search: find the mask with maximum distinct-degree count among subsets of size >= 8
_best_mask = 0
_best_score = 0

# Iterate over all masks of size 8 to 17
for _size in range(8, _N + 1):
    for _combo in itertools.combinations(range(_N), _size):
        _mask = 0
        for _v in _combo:
            _mask |= (1 << _v)
        _sc = _score_mask(_mask)
        if _sc > _best_score:
            _best_score = _sc
            _best_mask = _mask

# Convert best mask to list of vertices
_best_vertices = [v for v in range(_N) if (_best_mask >> v) & 1]

def construct():
    """Return a list of at least 8 distinct vertices maximizing distinct degrees in induced subgraph."""
    return list(_best_vertices)

# EVOLVE-BLOCK-END
