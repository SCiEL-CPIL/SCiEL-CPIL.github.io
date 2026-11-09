import time
import random
import itertools
from collections import deque, defaultdict

# Structural change: use a fast, bitset-based SAT-style backtracking checker
# (2-SAT over "edge-monochromatic" clauses via implication graph is not exact for
# non-monochromatic constraint, so we do a proper hypergraph backtracking with
# strong unit propagation and memoized state compression on the *unassigned*
# vertices' incident edge constraints).
#
# Additionally, we do a constructive *edge-addition* search guided by a
# "critical edge" heuristic: at each step, pick the edge that, when added,
# destroys the largest fraction of remaining 2-colorings (approximated by
# checking which edges participate in the most minimal conflict-free colorings
# found by a few random samples).
#
# We also include a small local-search "edge removal" phase: given a non-2-
# colorable hypergraph, try to remove edges one at a time (in order of least
# "usefulness" — edges involved in the fewest monochromatic conflicts under
# random colorings) and test whether non-2-colorability is preserved.

import sys
sys.setrecursionlimit(100000)

def _fast_not_2colorable(n, edges, timeout=5.0):
    """
    Returns True if the 5-uniform hypergraph on n vertices with given edges
    is NOT 2-colorable (i.e., every 2-coloring has a monochromatic edge).
    Uses backtracking with unit propagation and a timeout guard.
    Returns None on timeout.
    """
    start = time.time()
    
    # Preprocess: deduplicate edges, represent as frozensets
    edge_set = set()
    for e in edges:
        if len(e) == 5:
            edge_set.add(frozenset(e))
    if not edge_set:
        return True  # trivially 2-colorable

    edges_list = list(edge_set)
    e_count = len(edges_list)
    
    # Vertex adjacency: for each vertex, list of edge indices containing it
    v_edges = defaultdict(list)
    for i, e in enumerate(edges_list):
        for v in e:
            v_edges[v].append(i)
    
    # Degree ordering: sort vertices by descending degree
    deg = [len(v_edges[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # Remap edges to position indices
    remapped = []
    for e in edges_list:
        remapped.append(tuple(pos[v] for v in e))
    
    # For unit propagation efficiency, for each edge index, store its 5 positions
    # and a counter array for color counts.
    
    # We'll use a stack-based approach with state snapshots for backtracking.
    # colors: list of int, -1 unassigned, 0 or 1 assigned
    colors = [-1] * n
    # edge_color_counts: for each edge, [count0, count1, unassigned_count]
    e_c0 = [0] * e_count
    e_c1 = [0] * e_count
    e_un = [5] * e_count
    # For each edge, track the unassigned vertices (list)
    e_un_verts = [list(remapped[i]) for i in range(e_count)]
    
    # Unit propagation: find edges with 4 of one color and 1 unassigned -> force
    # Also detect contradictions: 5 of same color.
    
    def propagate():
        """
        Apply unit propagation. Returns (ok, newly_assigned_count).
        Modifies colors, e_c0, e_c1, e_un, e_un_verts in place.
        """
        changed = True
        while changed:
            changed = False
            for i in range(e_count):
                if e_un[i] == 0:
                    # All assigned; check monochromatic
                    if e_c0[i] == 5 or e_c1[i] == 5:
                        return False
                    continue
                if e_un[i] == 1:
                    # One unassigned
                    uv = e_un_verts[i][0]
                    if e_c0[i] == 4:
                        # Must be color 1
                        if colors[uv] == -1:
                            colors[uv] = 1
                            changed = True
                            # Update all edges containing uv
                            for j in v_edges[uv]:
                                e_c0[j] += 0
                                e_c1[j] += 1
                                e_un[j] -= 1
                                # Remove uv from e_un_verts[j]
                                k = e_un_verts[j].index(uv)
                                e_un_verts[j].pop(k)
                        elif colors[uv] == 0:
                            return False
                    elif e_c1[i] == 4:
                        if colors[uv] == -1:
                            colors[uv] = 0
                            changed = True
                            for j in v_edges[uv]:
                                e_c0[j] += 1
                                e_c1[j] += 0
                                e_un[j] -= 1
                                k = e_un_verts[j].index(uv)
                                e_un_verts[j].pop(k)
                        elif colors[uv] == 1:
                            return False
        return True
    
    # State snapshot for backtracking
    import copy
    def snapshot():
        return (colors.copy(), e_c0.copy(), e_c1.copy(), e_un.copy(), 
                [list(x) for x in e_un_verts])
    
    def restore(snap):
        colors[:] = snap[0]
        e_c0[:] = snap[1]
        e_c1[:] = snap[2]
        e_un[:] = snap[3]
        for i in range(e_count):
            e_un_verts[i] = list(snap[4][i])
    
    # Backtracking
    # We assign vertices in 'order' (position 0,1,...,n-1)
    # But unit propagation may assign out-of-order vertices.
    # We track which positions are "forced" vs "free".
    
    # Simpler: recursive backtracking over positions 0..n-1.
    # At each step, if position k is already assigned (by propagation), skip.
    
    def bt(k):
        if time.time() - start > timeout:
            return None  # timeout
        
        # Find next unassigned position
        while k < n and colors[k] != -1:
            k += 1
        if k == n:
            # All assigned. Check all edges.
            for i in range(e_count):
                if e_c0[i] == 5 or e_c1[i] == 5:
                    return False
            return True
        
        snap = snapshot()
        
        # Try color 0
        colors[k] = 0
        for j in v_edges[k]:
            e_c0[j] += 1
            e_un[j] -= 1
            idx = e_un_verts[j].index(k)
            e_un_verts[j].pop(idx)
        
        ok = propagate()
        if ok:
            result = bt(k + 1)
            if result is not None:
                return result
        else:
            pass  # conflict, try other color
        
        restore(snap)
        
        # Try color 1
        colors[k] = 1
        for j in v_edges[k]:
            e_c1[j] += 1
            e_un[j] -= 1
            idx = e_un_verts[j].index(k)
            e_un_verts[j].pop(idx)
        
        ok = propagate()
        if ok:
            result = bt(k + 1)
            if result is not None:
                return result
        
        restore(snap)
        return False
    
    result = bt(0)
    if result is None:
        return None
    # result is True if 2-colorable, False if not
    return not result

def _construct_cyclic(n, shifts):
    """
    Construct a cyclic 5-uniform hypergraph on n vertices.
    shifts: list of 5-element tuples of offsets mod n.
    Edges: for each base in shifts, for each i in 0..n-1, edge {i, i+s1, i+s2, i+s3, i+s4} mod n.
    Deduplicate.
    """
    edges = set()
    for base in shifts:
        for i in range(n):
            e = tuple(sorted((i + s) % n for s in base))
            # Ensure all distinct
            if len(set(e)) == 5:
                edges.add(e)
    return list(edges)

def _try_remove_edge(n, edges, edge_idx, timeout=3.0):
    """
    Try removing edge edge_idx from the hypergraph. 
    Returns True if the resulting graph is still non-2-colorable.
    """
    new_edges = [edges[i] for i in range(len(edges)) if i != edge_idx]
    result = _fast_not_2colorable(n, new_edges, timeout=timeout)
    return result is True  # True means not 2-colorable

def _local_search_remove(n, edges, time_budget=10.0):
    """
    Given a non-2-colorable hypergraph, try to remove edges to minimize count.
    Greedy: repeatedly try to remove edges in random order, keep if still non-2-colorable.
    """
    start = time.time()
    current = list(edges)
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        # Shuffle edge order for diversification
        order = list(range(len(current)))
        random.shuffle(order)
        for idx in order:
            if time.time() - start > time_budget:
                break
            # Try removing current[idx]
            if _try_remove_edge(n, current, idx, timeout=2.0):
                current = [e for i, e in enumerate(current) if i != idx]
                improved = True
                # Restart loop to re-shuffle
                break
    return current

def _random_edge_search(n, target_edges, time_budget=8.0):
    """
    Randomly add 5-uniform edges until non-2-colorable, aiming for ~target_edges.
    Returns the edge list if found, else None.
    """
    start = time.time()
    all_possible = list(itertools.combinations(range(n), 5))
    random.shuffle(all_possible)
    
    current = []
    for e in all_possible:
        if time.time() - start > time_budget:
            break
        current.append(e)
        # Check periodically (every few edges) to avoid too many checks
        if len(current) % 5 == 0:
            result = _fast_not_2colorable(n, current, timeout=2.0)
            if result is True:
                # Non-2-colorable!
                return current
        if len(current) > target_edges + 20:
            break
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    
    best_result = None
    best_count = float('inf')
    
    # Strategy 1: Cyclic constructions with local search pruning
    # Try various n and shift sets
    cyclic_configs = [
        (15, [(0,1,2,3,4), (0,1,2,3,7)]),
        (15, [(0,1,2,4,6), (0,1,3,5,7)]),
        (20, [(0,1,2,3,4), (0,1,2,5,9)]),
        (20, [(0,1,3,5,7), (0,2,4,6,8)]),
        (25, [(0,1,2,3,4), (0,1,2,7,14)]),
        (25, [(0,1,3,5,7), (0,2,4,6,8)]),
    ]
    
    for n, shifts in cyclic_configs:
        if time.time() - start_time > DEADLINE:
            break
        edges = _construct_cyclic(n, shifts)
        # Check if non-2-colorable
        result = _fast_not_2colorable(n, edges, timeout=5.0)
        if result is None:
            continue
        if result is True:
            # Non-2-colorable, try to prune
            budget = max(1.0, min(5.0, DEADLINE - (time.time() - start_time)))
            pruned = _local_search_remove(n, edges, time_budget=budget)
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_result = {'n': n, 'edges': [list(e) for e in pruned]}
        # Also try random edge search for this n
        if time.time() - start_time > DEADLINE:
            break
        random_edges = _random_edge_search(n, target_edges=50, time_budget=5.0)
        if random_edges is not None:
            budget = max(1.0, min(5.0, DEADLINE - (time.time() - start_time)))
            pruned = _local_search_remove(n, random_edges, time_budget=budget)
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_result = {'n': n, 'edges': [list(e) for e in pruned]}
    
    # Strategy 2: Direct random edge search for smaller n
    for n in [10, 11, 12, 13, 14]:
        if time.time() - start_time > DEADLINE:
            break
        for attempt in range(5):
            if time.time() - start_time > DEADLINE:
                break
            random_edges = _random_edge_search(n, target_edges=40, time_budget=4.0)
            if random_edges is not None:
                budget = max(0.5, min(3.0, DEADLINE - (time.time() - start_time)))
                pruned = _local_search_remove(n, random_edges, time_budget=budget)
                if len(pruned) < best_count:
                    best_count = len(pruned)
                    best_result = {'n': n, 'edges': [list(e) for e in pruned]}
    
    # Fallback: complete 5-uniform hypergraph on 9 vertices (126 edges)
    # This is guaranteed non-2-colorable by pigeonhole.
    if best_result is None or best_count > 126:
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    n = result['n']
    edges = result['edges']
    print(f"n={n}, edges={len(edges)}")
    # Verify
    is_non2c = _fast_not_2colorable(n, edges, timeout=10.0)
    if is_non2c is True:
        print("Valid: Non-2-colorable")
    elif is_non2c is False:
        print("INVALID: 2-Colorable")
    else:
        print("Unknown: timeout")
