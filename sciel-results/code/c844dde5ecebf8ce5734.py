"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 148  # seconds to spare before the 150s kill

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

_Q3_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for bit in (1, 2, 4):
        j = i ^ bit
        if j > i:
            _Q3_NEIGHBORS[i].append(j)
            _Q3_NEIGHBORS[j].append(i)

# Precompute: for each qv, bitmask of Q3 neighbors
_Q3_QV_NEIGHBOR_MASK = [0] * 8
for i in range(8):
    mask = 0
    for j in _Q3_NEIGHBORS[i]:
        mask |= (1 << j)
    _Q3_QV_NEIGHBOR_MASK[i] = mask

# Precompute: for each qv, list of Q3 neighbors that come before it in the backtracking order
# Backtracking order: 0, 1, 2, 3, 4, 5, 6, 7
_Q3_PRIOR_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for j in _Q3_NEIGHBORS[i]:
        if j < i:
            _Q3_PRIOR_NEIGHBORS[i].append(j)

# Q3 is bipartite: A={0,3,5,6}, B={1,2,4,7}
_Q3_PART_A = {0, 3, 5, 6}
_Q3_PART_B = {1, 2, 4, 7}

# Precompute all 240 oriented Q3 edge pairs:
# For each pair of adjacent Q3 vertices (a,b) where a<b, and for each of 16*15=240
# ordered pairs of distinct graph vertices (u,v), store:
#   - the 6 remaining Q3 vertices in a good backtracking order
#   - for each, the set of Q3 neighbors that are among {a,b} or earlier in the order
# This lets us do a very fast check: for a given adj and a candidate (u,v) mapped to (a,b),
# we check if there's a valid assignment of the 6 remaining Q3 vertices to the other 14 graph vertices.

# Actually, let's precompute the 12 undirected Q3 edges and for each, the 6 remaining vertices
# in an order that maximizes pruning.

# Q3 edges (undirected): 12 edges
# Q3 vertices: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
# Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)

# For each Q3 edge (a,b), determine the order of the other 6 vertices for backtracking.
# We want vertices with more constraints (more neighbors among {a,b} and earlier vertices) first.

def _compute_backtrack_order(a, b):
    """Compute optimal backtracking order for the 6 remaining Q3 vertices given fixed edge (a,b)."""
    remaining = [v for v in range(8) if v != a and v != b]
    
    # For each remaining vertex, count how many of its Q3 neighbors are in {a,b}
    # or in earlier positions in the order (iteratively).
    
    # Use a greedy approach: at each step, pick the vertex with the most already-placed neighbors
    order = []
    placed = {a, b}
    for _ in range(6):
        best_v = -1
        best_count = -1
        for v in remaining:
            if v in placed:
                continue
            count = sum(1 for n in _Q3_NEIGHBORS[v] if n in placed)
            if count > best_count:
                best_count = count
                best_v = v
        order.append(best_v)
        placed.add(best_v)
    
    return order

# Precompute for each of the 12 Q3 edges:
# - the 6 remaining vertices in backtrack order
# - for each position i in the order, which of {a,b} and order[0..i-1] are Q3-neighbors of order[i]
_Q3_EDGE_INFO = {}
for a, b in _Q3_EDGES:
    order = _compute_backtrack_order(a, b)
    # For each position i, precompute the list of Q3 neighbors of order[i] that are in {a,b} ∪ order[0..i-1]
    neighbor_info = []
    for i, v in enumerate(order):
        prior = set([a, b]) | set(order[:i])
        nbrs = [n for n in _Q3_NEIGHBORS[v] if n in prior]
        neighbor_info.append(nbrs)
    _Q3_EDGE_INFO[(a, b)] = (order, neighbor_info)

# Precompute all possible edge pairs and their conflict status
# Two edges (u1,v1) and (u2,v2) conflict if they can't both be in a Q3-free graph
# This happens when there exists a Q3 embedding that uses both edges
_ALL_EDGES_LIST = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_EDGE_TO_IDX = {}
for _i, (_e) in enumerate(_ALL_EDGES_LIST):
    _EDGE_TO_IDX[_e] = _i

# Precompute conflict bitmasks for each edge
# conflict_mask[e] is a bitmask of edges that conflict with edge e
_conflict_masks = [0] * _E

def _edges_conflict(e1, e2):
    """Check if two edges e1=(u1,v1) and e2=(u2,v2) conflict (can't both be in Q3-free graph).
    They conflict if there's a Q3 embedding containing both edges."""
    u1, v1 = e1
    u2, v2 = e2
    
    # Check if the two edges share a vertex or are disjoint
    shared = set([u1, v1]) & set([u2, v2])
    
    # Build a temporary graph with just these two edges and check if they're in a Q3
    # Actually, we need to check if there exists a Q3 that contains both edges
    # This is equivalent to: can we extend {e1, e2} to a full Q3?
    
    # For speed, we'll use the backtracking approach but only on the 8 vertices involved
    # or a subset thereof
    
    # Get all vertices involved
    verts = set([u1, v1, u2, v2])
    if len(verts) < 4:
        # Edges share a vertex or are the same - need careful check
        pass
    
    # Try all Q3 embeddings that could contain both edges
    # For each Q3 edge (a,b), try mapping e1 and e2 to Q3 edges
    # This is complex, so let's use a simpler approach:
    # Check if adding both edges to an empty graph creates a Q3 subgraph
    
    # Actually, the simplest correct approach: two edges conflict iff
    # there exists a Q3 that contains both as subedges.
    # We can check this by trying all ways to embed the two edges into Q3.
    
    # For each pair of Q3 edges (qa, qb) where qa != qb:
    # Try to find a graph homomorphism that maps e1->qa and e2->qb
    # and is injective on vertices
    
    # This is getting complex. Let's use a different approach:
    # Precompute by brute force for all pairs (120*119/2 = 7140 pairs)
    # For each pair, check if they can coexist in a Q3-free graph
    # Two edges conflict iff every Q3-free supergraph of {e1,e2} doesn't exist
    # which is equivalent to: {e1,e2} is a subset of some Q3
    
    # Check if {e1, e2} is a subset of some Q3:
    # Try all 8! / (8-4)! = 1680 ways to map the 4 vertices of {e1,e2} to Q3 vertices
    # Actually, we need to check if there's a Q3 embedding where e1 and e2 are both edges
    
    # For each Q3, check if there's a subgraph isomorphic to {e1, e2}
    # Since Q3 has 12 edges, we just need to check if e1 and e2 can be mapped to two distinct edges of Q3
    
    # Map e1's endpoints to two adjacent Q3 vertices, and e2's endpoints to two adjacent Q3 vertices
    # such that the mapping is injective
    
    # For e1 = (u1, v1): try all 12 directed Q3 edges (a,b) with a<b
    # For e2 = (u2, v2): try all 12 directed Q3 edges (c,d) with c<d, (c,d) != (a,b)
    # Check if the vertex mapping is injective
    
    for qa, qb in _Q3_EDGES:
        # Map u1->qa, v1->qb
        if u1 == v1 or u2 == v2:
            continue
        
        for qc, qd in _Q3_EDGES:
            if (qc, qd) == (qa, qb):
                continue
            
            # Mapping: u1->qa, v1->qb, u2->qc, v2->qd
            # Check injectivity
            mapping = {u1: qa, v1: qb, u2: qc, v2: qd}
            vals = list(mapping.values())
            if len(set(vals)) != len(vals):
                continue
            
            # Check that the Q3 edges (qa,qb) and (qc,qd) are valid (they are by construction)
            # But we also need to ensure that the non-edges in {e1,e2} are non-edges in Q3
            # i.e., if u1 and u2 are not adjacent in the graph, they shouldn't be adjacent in Q3
            # Wait, we only know e1 and e2 are edges. Other pairs may or may not be edges.
            # For conflict, we need: there exists a Q3 that CONTAINS both e1 and e2 as edges.
            # This means: we need a Q3 embedding where e1 maps to a Q3 edge and e2 maps to a Q3 edge.
            # The other vertex pairs don't matter - they can be edges or non-edges in the graph.
            # So we just need the mapping to be injective and e1, e2 to map to valid Q3 edges.
            
            return True
    
    return False

# Precompute conflict masks - this takes a while but is done once
for _i in range(_E):
    for _j in range(_i + 1, _E):
        if _edges_conflict(_ALL_EDGES_LIST[_i], _ALL_EDGES_LIST[_j]):
            _conflict_masks[_i] |= (1 << _j)
            _conflict_masks[_j] |= (1 << _i)


def _has_q3_with_edge(adj_bits, u, v):
    """
    Check if adding edge (u,v) to the current graph creates a Q3.
    Uses the precomputed edge info for fast backtracking.
    """
    full_mask = (1 << _N) - 1
    
    # Get neighbors of u and v (excluding each other)
    nbrs_u = adj_bits[u] & ~(1 << v)
    nbrs_v = adj_bits[v] & ~(1 << u)
    
    if nbrs_u.bit_count() < 2 or nbrs_v.bit_count() < 2:
        return False
    
    for (a, b) in _Q3_EDGES:
        order, neighbor_info = _Q3_EDGE_INFO[(a, b)]
        
        if _check_q3_with_fixed_edge(adj_bits, u, v, a, b, order, neighbor_info, full_mask):
            return True
        if _check_q3_with_fixed_edge(adj_bits, v, u, a, b, order, neighbor_info, full_mask):
            return True
    
    return False


def _check_q3_with_fixed_edge(adj_bits, gu, gv, a, b, order, neighbor_info, full_mask):
    """
    Check if there's a Q3 embedding where graph vertex gu maps to Q3 vertex a,
    gv maps to Q3 vertex b, and the remaining 6 Q3 vertices map to 6 distinct
    graph vertices from the remaining 14.
    
    gu, gv: graph vertices mapped to Q3 vertices a, b
    order: the 6 remaining Q3 vertices in backtracking order
    neighbor_info: for each position i, list of Q3 neighbors of order[i] in {a,b} ∪ order[0..i-1]
    """
    used_mask = (1 << gu) | (1 << gv)
    assigned = {}  # qv -> gv
    assigned[a] = gu
    assigned[b] = gv
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == 6:
            return True
        
        qv = order[idx]
        candidates = (~used_mask) & full_mask
        
        # Must be adjacent to all prior Q3 neighbors
        for nqv in neighbor_info[idx]:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv2 = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv2
            used_mask |= (1 << gv2)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv2)
            del assigned[qv]
        
        return False
    
    return backtrack(0)


def _find_q3_witness(adj_bits, u, v):
    """
    Find a specific Q3 embedding that includes edge (u,v), if one exists.
    Returns a set of 8 graph vertices forming the Q3, or None.
    """
    full_mask = (1 << _N) - 1
    
    nbrs_u = adj_bits[u] & ~(1 << v)
    nbrs_v = adj_bits[v] & ~(1 << u)
    
    if nbrs_u.bit_count() < 2 or nbrs_v.bit_count() < 2:
        return None
    
    list_u = []
    c = nbrs_u
    while c:
        lsb = c & (-c)
        list_u.append(lsb.bit_length() - 1)
        c ^= lsb
    
    list_v = []
    c = nbrs_v
    while c:
        lsb = c & (-c)
        list_v.append(lsb.bit_length() - 1)
        c ^= lsb
    
    for i in range(len(list_u)):
        a = list_u[i]
        for j in range(i+1, len(list_u)):
            b = list_u[j]
            for k in range(len(list_v)):
                c2 = list_v[k]
                for l in range(k+1, len(list_v)):
                    d = list_v[l]
                    
                    mask_excl = ((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c2) | (1 << d)) & full_mask
                    e_candidates = adj_bits[a] & adj_bits[b] & ~mask_excl
                    f_candidates = adj_bits[c2] & adj_bits[d] & ~mask_excl
                    
                    ec = e_candidates
                    while ec:
                        lsb_e = ec & (-ec)
                        e = lsb_e.bit_length() - 1
                        ec ^= lsb_e
                        
                        fc = f_candidates & adj_bits[e]
                        if fc:
                            lsb_f = fc & (-fc)
                            f = lsb_f.bit_length() - 1
                            return {u, v, a, b, c2, d, e, f}
    
    return None

def _max_independent_set_bnb(conflict_masks, edge_order, deadline, initial_solution=None):
    """
    Branch and bound for maximum independent set on the conflict graph.
    conflict_masks: list of bitmasks, conflict_masks[i] has bit j set if edges i and j conflict
    edge_order: list of edge indices in the order to consider
    deadline: time limit
    initial_solution: optional initial independent set (list of edge indices)
    """
    n = len(edge_order)
    best_set = set(initial_solution) if initial_solution else set()
    best_size = len(best_set)
    
    # Precompute: for each edge in edge_order, which other edges in edge_order conflict with it
    # Use the global conflict_masks
    
    # Greedy upper bound: sort remaining edges by degree in the induced subgraph
    def _greedy_upper_bound(candidates_mask):
        """Compute a greedy upper bound on the independent set size from candidates."""
        count = 0
        remaining = candidates_mask
        while remaining:
            # Pick the vertex with minimum degree in the remaining subgraph
            best_v = -1
            best_deg = _E + 1
            c = remaining
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                # Degree of v in remaining subgraph
                deg = (conflict_masks[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            count += 1
            # Remove v and its neighbors
            remaining &= ~(conflict_masks[best_v] | (1 << best_v))
        return count
    
    # Branch and bound
    sys.setrecursionlimit(10000)
    
    def bnb(idx, current_mask, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound
        remaining_mask = current_mask
        ub = _greedy_upper_bound(remaining_mask)
        if current_size + ub <= best_size:
            return
        
        e = edge_order[idx]
        
        # Option 1: include e
        # Check if e is compatible with all in current_set
        compatible = True
        for ce in current_set:
            if (conflict_masks[e] >> ce) & 1:
                compatible = False
                break
        
        if compatible:
            new_set = current_set | {e}
            new_mask = current_mask & ~(conflict_masks[e] | (1 << e))
            bnb(idx + 1, new_mask, new_set, current_size + 1)
        
        # Option 2: exclude e
        bnb(idx + 1, current_mask, current_set, current_size)
    
    all_mask = (1 << n) - 1
    bnb(0, all_mask, set(), 0)
    
    return list(best_set)


def _has_q3_fast(adj_bits):
    """
    Full Q3 subgraph detection using bitmask operations.
    adj_bits: list of 16 integers, each a 16-bit mask of neighbors.
    """
    assigned = [-1] * 8
    used_mask = 0
    full_mask = (1 << _N) - 1
    
    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        
        candidates = (~used_mask) & full_mask
        
        for pv in _Q3_PRIOR_NEIGHBORS[qv]:
            pgv = assigned[pv]
            candidates &= adj_bits[pgv]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(qv + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return False
    
    for gv0 in range(_N):
        assigned[0] = gv0
        used_mask = 1 << gv0
        if backtrack(1):
            return True
        used_mask = 0
        assigned[0] = -1
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    # Phase 1: Greedy edge addition with multiple orderings to get a good initial solution
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 40:
            break
        
        if trial == 0:
            edge_order = all_edges[:]
        elif trial == 1:
            edge_order = all_edges[::-1]
        else:
            random.seed(trial * 12345 + 67890)
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        
        adj = [0] * _N
        edges = []
        
        for u, v in edge_order:
            if time.time() > deadline - 30:
                break
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge(adj, u, v):
                edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    # Phase 2: Local search - add missing edges
    if time.time() < deadline - 30:
        adj = best_adj[:]
        edges = best_edges[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
        
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            for i in range(len(missing)):
                if time.time() > deadline - 20:
                    break
                u, v = missing[i]
                if (u, v) in edge_set:
                    continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not _has_q3_with_edge(adj, u, v):
                    edges.append([u, v])
                    edge_set.add((u, v))
                    missing.pop(i)
                    i -= 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    # Phase 3: Branch and bound on the conflict graph
    if time.time() < deadline - 10:
        # Get the current best solution as initial
        current_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in best_edges)
        initial_solution = [_EDGE_TO_IDX[e] for e in current_edge_set if e in _EDGE_TO_IDX]
        
        # Use all edges in the order of the initial solution first, then the rest
        edge_indices = list(range(_E))
        
        # Sort by: edges in initial solution first (in order), then others
        initial_set = set(initial_solution)
        edge_indices.sort(key=lambda x: (0 if x in initial_set else 1, x))
        
        bnb_deadline = deadline - 5
        bnb_result = _max_independent_set_bnb(_conflict_masks, edge_indices, bnb_deadline, initial_solution)
        
        if len(bnb_result) > best_count:
            # Convert back to edge list
            bnb_edges = []
            for idx in bnb_result:
                u, v = _ALL_EDGES_LIST[idx]
                bnb_edges.append([u, v])
            best_count = len(bnb_edges)
            best_edges = bnb_edges[:]
    
    # Phase 4: Final polish - try to add any remaining edges
    if time.time() < deadline - 5:
        adj = [0] * _N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        edges = best_edges[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        # Multiple passes to catch cascading additions
        for _polish_pass in range(3):
            if time.time() > deadline - 3:
                break
            improved = False
            for u, v in all_edges:
                if time.time() > deadline - 3:
                    break
                if (u, v) in edge_set:
                    continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not _has_q3_with_edge(adj, u, v):
                    edges.append([u, v])
                    edge_set.add((u, v))
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if not improved:
                break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Final verification: make sure the result is Q3-free
    if time.time() < deadline - 2:
        adj = [0] * _N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if _has_q3_fast(adj):
            # Remove edges from Q3s until Q3-free
            while _has_q3_fast(adj) and time.time() < deadline - 1:
                # Find a Q3 and remove one of its edges
                # For simplicity, just remove edges until no Q3
                for i in range(len(best_edges)):
                    u, v = best_edges[i]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if not _has_q3_fast(adj):
                        best_edges.pop(i)
                        break
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    return best_edges
# EVOLVE-BLOCK-END
