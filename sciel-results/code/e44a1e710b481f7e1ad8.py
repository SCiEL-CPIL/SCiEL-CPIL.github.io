import time
import math
from itertools import combinations

_N = 256
_DEADLINE = 70.0  # seconds, safety margin below the 90s kill
_start = time.time()

def _elapsed():
    return time.time() - _start

# Precompute: which numbers are perfect squares (a+b max = 512)
def _is_square(n):
    r = int(math.isqrt(n))
    return r * r == n

# Build conflict graph: edge (i,j) if i+j is a perfect square (i<j), and self-loop if 2*i is a square
# Represent adjacency as bitsets (Python ints) for fast intersection
# Nodes indexed 0..255 represent values 1..256
_ADJ = [0] * _N
_SELF = [False] * _N
for i in range(_N):
    v = i + 1
    if _is_square(2 * v):
        _SELF[i] = True
    for j in range(i + 1, _N):
        if _is_square(v + (j + 1)):
            _ADJ[i] |= (1 << j)
            _ADJ[j] |= (1 << i)

# A node is "bad" (can never be in a valid set) if it has a self-loop
# We'll handle this by removing such nodes from consideration.

def _bit_count(x):
    return bin(x).count("1")

def _popcount(x):
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

def _greedy_solution(available_mask):
    """Greedy: repeatedly pick the node with fewest conflicts among available nodes."""
    avail = available_mask
    chosen = set()
    while avail:
        # pick node with minimum degree within avail
        best = -1
        best_deg = 1 << 30
        m = avail
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            deg = _popcount(_ADJ[idx] & avail)
            if deg < best_deg:
                best_deg = deg
                best = idx
            m ^= lsb
        if best < 0:
            break
        chosen.add(best)
        # remove best and all its neighbors from avail
        avail &= ~_ADJ[best]
        avail &= ~(1 << best)
    return chosen

def _greedy_clique_bound(adj_mask, node):
    """Upper bound on max independent set size in the subgraph induced by adj_mask.
    Use a greedy coloring bound on the complement... actually we want a bound on
    the independent set = clique in complement. Simpler: use a greedy upper bound
    via ordering and coloring the graph (for MIS, the standard is a coloring of
    the graph gives an upper bound on clique; for independent set we color the
    complement).
    
    For an independent set in G, it's a clique in G's complement. A greedy coloring
    of the complement gives an upper bound on the clique size.
    """
    # Build list of nodes in adj_mask
    nodes = []
    m = adj_mask
    while m:
        lsb = m & (-m)
        nodes.append(lsb.bit_length() - 1)
        m ^= lsb
    n = len(nodes)
    if n == 0:
        return 0
    # Color the complement graph greedily. In complement, two nodes are adjacent
    # if they are NOT adjacent in G.
    # We process nodes and assign smallest available color.
    # The number of colors used is an upper bound on the clique size in the
    # complement = independent set size in G.
    # To get a tight bound, order by degree in G (high degree in G = low degree in complement)
    # Actually for coloring the complement, we want a good ordering.
    # Let's just do a simple greedy coloring of the complement.
    node_set = set(nodes)
    # Order: sort by degree in G descending (so low degree in complement first)
    nodes_sorted = sorted(nodes, key=lambda x: -_popcount(_ADJ[x] & adj_mask))
    color_of = {}
    next_color = 0
    for v in nodes_sorted:
        # In complement, v is adjacent to u if u is NOT adjacent to v in G and u != v
        # Find colors used by complement-neighbors of v
        used = set()
        for u, c in color_of.items():
            if u == v:
                continue
            if not (_ADJ[v] >> u) & 1:
                # u and v are NOT adjacent in G => adjacent in complement
                used.add(c)
        # assign smallest available color
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        if c > next_color:
            next_color = c
    return next_color + 1  # number of colors

def _branch_and_bound(available_mask, current_set, current_size, best_set, best_size, deadline):
    """Branch and bound for maximum independent set."""
    nonlocal _best_set, _best_size
    if _elapsed() > deadline:
        return
    if available_mask == 0:
        if current_size > _best_size:
            _best_size = current_size
            _best_set = set(current_set)
        return
    # Upper bound: current_size + (number of available nodes)
    # Tighter: current_size + greedy_clique_bound on available
    ub = current_size + _greedy_clique_bound(available_mask, -1)
    if ub <= _best_size:
        return
    # Pick a branching node: the one with highest degree in available (to reduce branching)
    m = available_mask
    best_node = -1
    best_deg = -1
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        deg = _popcount(_ADJ[idx] & available_mask)
        if deg > best_deg:
            best_deg = deg
            best_node = idx
        m ^= lsb
    # Branch 1: include best_node
    new_avail = available_mask & ~_ADJ[best_node] & ~(1 << best_node)
    current_set.add(best_node)
    _branch_and_bound(new_avail, current_set, current_size + 1, best_set, best_size, deadline)
    current_set.discard(best_node)
    # Branch 2: exclude best_node
    _branch_and_bound(available_mask & ~(1 << best_node), current_set, current_size, best_set, best_size, deadline)

_best_set = set()
_best_size = 0

def construct():
    global _best_set, _best_size
    deadline = _start + _DEADLINE
    # Remove self-loop nodes (they can never be in a valid set)
    available_mask = 0
    for i in range(_N):
        if not _SELF[i]:
            available_mask |= (1 << i)
    # Greedy initial solution
    greedy = _greedy_solution(available_mask)
    _best_set = set(greedy)
    _best_size = len(greedy)
    # Try branch and bound to improve
    _branch_and_bound(available_mask, set(), 0, _best_set, _best_size, deadline)
    # Convert back to values
    result = sorted(i + 1 for i in _best_set)
    if not result:
        result = [1]
    return result
