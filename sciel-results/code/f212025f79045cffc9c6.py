"""
Q3-free graph on 16 vertices: maximize edges.

Strategy:
  1. Build a dense starting graph via greedy edge addition.
  2. Repair any Q3 violations by removing edges.
  3. Local search: try to add missing edges, try swaps (remove one, add one),
     accepting moves that keep the graph Q3-free.
"""

import itertools
import random
import time

# ---------------------------------------------------------------------------
# Q3 adjacency (3-cube on vertices 0..7)
# Q3 edges: i connected to i xor 1, i xor 2, i xor 4
Q3_ADJ = {}
for i in range(8):
    Q3_ADJ[i] = sorted(set([i ^ 1, i ^ 2, i ^ 4]))

Q3_EDGES = set()
for i in range(8):
    for j in Q3_ADJ[i]:
        if i < j:
            Q3_EDGES.add((i, j))

# All 8! permutations for isomorphism check
_PERMS = list(itertools.permutations(range(8)))


def _is_q3_subgraph(adj_bits):
    """
    Check if the 8-vertex graph given by adj_bits (list of 8 ints, each a bitmask
    of 8 bits) contains a subgraph isomorphic to Q3.
    adj_bits[v] is a bitmask over the 8 local vertices.
    """
    # For each permutation p of 8 vertices, check if Q3 maps into adj_bits
    for p in _PERMS:
        # p[i] = which local vertex corresponds to Q3 vertex i
        ok = True
        for i in range(8):
            for j in Q3_ADJ[i]:
                u, v = p[i], p[j]
                if u == v:
                    ok = False
                    break
                # Check if edge (u,v) exists in adj_bits
                if not (adj_bits[u] >> v) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


def _check_q3(edges_set, n=16):
    """
    Check if the graph (set of edges on vertices 0..n-1) contains a Q3 subgraph.
    For each 8-vertex subset, check if it contains Q3.
    Returns (True, subset) or (False, None).
    """
    adj = [0] * n
    for (u, v) in edges_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    for subset in itertools.combinations(range(n), 8):
        # Build local adjacency
        sidx = {v: i for i, v in enumerate(subset)}
        local_adj = [0] * 8
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j and (adj[u] >> v) & 1:
                    local_adj[i] |= (1 << j)
        if _is_q3_subgraph(local_adj):
            return True, subset
    return False, None


def construct():
    t0 = time.time()
    deadline = t0 + 110  # safety margin

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)

    # --- Phase 1: Greedy construction ---
    # Start with K4,12 (bipartite, so Q3-free since Q3 is not bipartite... wait Q3 IS bipartite)
    # Actually Q3 is bipartite! So bipartite graphs can contain Q3.
    # K4,12: any Q3 would need 4 vertices on each side. Q3 has 4 on each side.
    # But K4,12 restricted to 4+4=8 vertices is K4,4 which has 16 edges > 12, so it contains Q3!
    # So K4,12 is NOT Q3-free. The seed used K3,13 (39 edges).
    
    # Let's use K3,13 as starting point (known Q3-free: any 8 vertices can have at most 3 on one side)
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    # Add some edges within the groups if safe
    # Actually, let's do a full greedy from scratch with a good ordering
    
    # Better: start with K3,13, then try to add all other edges greedily
    current_edges = set()
    for a in range(3):
        for b in range(3, 16):
            current_edges.add((a, b))
    
    # Try adding every other edge in random order
    other_edges = [e for e in all_edges if e not in current_edges]
    random.shuffle(other_edges)
    
    for e in other_edges:
        if time.time() > deadline - 60:
            break
        candidate = current_edges | {e}
        # Quick check: does adding e create a Q3?
        # Instead of full check, let's just do it
        has_q3, _ = _check_q3(candidate)
        if not has_q3:
            current_edges = candidate
    
    # --- Phase 2: Local search ---
    missing = [e for e in all_edges if e not in current_edges]
    
    while time.time() < deadline:
        # Try to add a random missing edge
        if missing:
            e = random.choice(missing)
            candidate = current_edges | {e}
            has_q3, _ = _check_q3(candidate)
            if not has_q3:
                current_edges = candidate
                missing.remove(e)
                continue
        
        # Try a swap: remove one edge, add another
        if missing and len(current_edges) > 0:
            e_remove = random.choice(list(current_edges))
            e_add = random.choice(missing)
            candidate = (current_edges - {e_remove}) | {e_add}
            if len(candidate) == len(current_edges):
                has_q3, _ = _check_q3(candidate)
                if not has_q3:
                    current_edges = candidate
                    missing = [e for e in all_edges if e not in current_edges]
    
    # Return edge list
    return [list(e) for e in current_edges]
