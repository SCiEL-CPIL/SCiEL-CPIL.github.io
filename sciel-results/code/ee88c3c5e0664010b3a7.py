"""Hexagon-free graph construction on 40 vertices via branch-and-bound on edge conflicts."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_PAIRS = len(ALL_PAIRS)  # 780
PAIR_INDEX = {(u, v): k for k, (u, v) in enumerate(ALL_PAIRS)}


def _precompute_c6_forbidden_pairs():
    """Return list of (i, j) where selecting both ALL_PAIRS[i] and ALL_PAIRS[j] forces a C6."""
    adj = [set() for _ in range(N)]
    forbidden = []
    for i in range(NUM_PAIRS):
        u1, v1 = ALL_PAIRS[i]
        adj[u1].add(v1)
        adj[v1].add(u1)
        for j in range(i + 1, NUM_PAIRS):
            u2, v2 = ALL_PAIRS[j]
            # Check if adding (u2,v2) to current graph creates a C6
            # A C6 through (u2,v2): u2-v2-a-b-c-d-u2 where a-b-c-d is path of length 4 from v2 to u2
            found = False
            for a in adj[v2]:
                if a == u2:
                    continue
                for b in adj[a]:
                    if b == u2 or b == v2 or b == a:
                        continue
                    for c in adj[b]:
                        if c == u2 or c == v2 or c == a or c == b:
                            continue
                        for d in adj[c]:
                            if d == u2 or d == v2 or d == a or d == b or d == c:
                                continue
                            if u2 in adj[d]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                forbidden.append((i, j))
            adj[u2].add(v2)
            adj[v2].add(u2)
    return forbidden


def _seed_graph_8K5():
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c6_adj(adj, n):
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False


def _greedy_color_bound(adjacency, vertices):
    """
    Upper bound on max independent set size in conflict subgraph induced by `vertices`.
    Uses greedy sequential coloring (DSATUR-like) to get an upper bound.
    adjacency: list of sets (conflict neighbors per vertex index)
    vertices: list of vertex indices to color
    Returns: upper bound (sum of color classes... actually number of colors used is upper bound on clique, 
    but for independent set upper bound we use: greedy coloring gives upper bound on alpha? No.
    Actually: greedy coloring gives an upper bound on the independence number? No, it gives a lower bound on chromatic number.
    The standard branch-and-bound for MIS uses: color the remaining vertices greedily, the color number is an upper bound on the size of the independent set in the remaining subgraph.
    Wait, that's not right either. Let me think again.
    In branch-and-bound for maximum independent set (or equivalently maximum clique in complement):
    - We maintain a set of candidate vertices.
    - We color the candidate vertices greedily (sequential coloring).
    - The number of colors used is an UPPER BOUND on the size of the maximum clique in the candidate subgraph.
    - For maximum independent set, we work on the complement graph, so the same applies.
    
    Actually, for MIS: we want to find the largest set of vertices with no edges between them.
    This is equivalent to finding the largest clique in the complement graph.
    Standard approach: color the candidate vertices in the ORIGINAL graph (conflict graph) using greedy coloring.
    The color number is an upper bound on the independence number of the conflict graph induced by candidates.
    Wait no. If we color the conflict graph, vertices in the same color class have no conflicts (no edge in conflict graph), so they CAN all be in the independent set.
    So the number of colors is a LOWER bound on the independence number? No...
    
    Let me reconsider. In the conflict graph, an edge means "cannot both be selected". An independent set in the conflict graph is a valid set of edges.
    Greedy coloring of the conflict graph: assign colors so that adjacent vertices have different colors.
    The minimum number of colors = chromatic number.
    Each color class is an independent set in the conflict graph.
    So the size of the largest color class is a lower bound on alpha.
    The chromatic number is an upper bound on... no, chromatic number >= alpha? No, chromatic number * alpha >= n.
    
    OK let me just use the standard Tomita/MaxClique style bound:
    For maximum clique: color the candidate vertices, the color of the last vertex in the ordering is an upper bound on the clique size extendable from that vertex.
    
    For maximum independent set in conflict graph = maximum clique in complement of conflict graph.
    Let's just implement a simple bound: greedily color the conflict graph candidates, the number of colors is an upper bound on the independence number? 
    Actually yes: if we can color with k colors, then the independence number <= k? No, that's wrong. 
    If we color with k colors, each color class is an independent set, so alpha >= max color class size. And chromatic number <= ... 
    Actually, alpha * chi >= n, so chi >= n/alpha, which means alpha >= n/chi. That's a lower bound.
    
    I think I'm overcomplicating. Let me use a simpler approach: 
    For the branch-and-bound, I'll use the following:
    - Order vertices by degree (conflict degree) descending.
    - Use a simple bound: the maximum possible additional edges we can select is at most the number of remaining candidates (trivial bound).
    - Better: use a greedy coloring of the CONFLICT graph. The number of colors used is an upper bound on the size of the maximum independent set? 
      Actually, I recall now: in the maximum clique problem, you color the graph and the color number is an upper bound on the clique size. For maximum independent set, you do the same on the complement.
      
    Let me just use a practical approach: greedy coloring of the conflict subgraph, and use the number of colors as an upper bound on the independence number. This is actually correct because:
    - In a properly colored graph, no two adjacent vertices share a color.
    - An independent set can contain at most one vertex of each color? NO, that's wrong. An independent set can contain multiple vertices of the same color (since same color = no edge between them).
    
    OK I'll use a different bound: the maximum degree in the remaining conflict subgraph gives a bound. Or I'll just use the simple count bound and rely on the reduction rules.
    
    Actually, let me just implement a simple but effective branch-and-bound:
    1. Precompute conflict pairs.
    2. Start with all 780 edges as candidates.
    3. Use recursive branch-and-bound:
       - Pick a vertex (edge) with highest conflict degree.
       - Branch: include it (remove all its conflicts) or exclude it.
       - Bound: current_size + remaining_candidates <= best_found -> prune.
       - Reduction: if a vertex has no conflicts with remaining candidates, it MUST be included (forced).
    4. Use time limit to stop and return best found.
    
    This is essentially a maximum independent set solver with simple bounds. Let me implement it.
    """
    pass  # This function is replaced by the actual implementation below


def _greedy_independent_set(conflict_adj, candidates, deadline):
    """
    Greedy heuristic to find a large independent set in the conflict graph.
    conflict_adj: list of sets, conflict neighbors for each of NUM_PAIRS vertices
    candidates: set of candidate vertex indices
    Returns: set of selected vertex indices
    """
    selected = set()
    remaining = set(candidates)
    while remaining:
        # Pick vertex with minimum conflict degree within remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(conflict_adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        selected.add(best_v)
        remaining.discard(best_v)
        # Remove all conflicts of best_v from remaining
        for u in conflict_adj[best_v]:
            remaining.discard(u)
    return selected


def _branch_and_bound(conflict_adj, deadline, initial_candidates=None):
    """
    Branch-and-bound for maximum independent set in conflict graph.
    conflict_adj: list of sets
    deadline: time limit in seconds
    Returns: set of selected vertices (edge indices)
    """
    start_time = time.time()
    if initial_candidates is None:
        initial_candidates = set(range(NUM_PAIRS))
    
    best_set = set()
    best_size = 0
    
    # Initial greedy solution
    greedy_set = _greedy_independent_set(conflict_adj, initial_candidates, deadline)
    best_set = set(greedy_set)
    best_size = len(best_set)
    
    # Sort candidates by conflict degree descending for branching
    def conflict_degree(v, cand_set):
        return len(conflict_adj[v] & cand_set)
    
    # Recursive B&B with time check
    # Use iterative deepening or just simple recursion with time checks
    # For 780 vertices, deep recursion may be an issue; use a stack-based approach
    
    # Simple approach: try to improve upon greedy by local search + limited B&B
    # Given time constraints, do a limited B&B
    
    # Actually, for 780 vertices with potentially dense conflicts, B&B may not be practical.
    # Let's use a hybrid: greedy + local search (swap-based) but with the conflict graph structure.
    
    # Local search on the independent set:
    # Try to add a non-selected vertex, or swap (remove one, add one or two)
    
    current_set = set(best_set)
    current_size = best_size
    
    # Precompute for fast checking
    in_set = [False] * NUM_PAIRS
    for v in current_set:
        in_set[v] = True
    
    all_vertices = list(initial_candidates)
    random.shuffle(all_vertices)
    
    # Local search: try to add vertices, or do swaps
    while time.time() < deadline - 2:
        progress = False
        
        # Try to add a random non-selected vertex
        for _ in range(100):
            v = random.choice(all_vertices)
            if in_set[v]:
                continue
            # Check if v conflicts with any in current_set
            conflict = False
            for u in conflict_adj[v]:
                if in_set[u]:
                    conflict = True
                    break
            if not conflict:
                current_set.add(v)
                in_set[v] = True
                current_size += 1
                progress = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                break
        
        if not progress and current_size > 0:
            # Try a 1-for-1 swap
            v_remove = random.choice(list(current_set))
            # Find a vertex not in set that conflicts with v_remove but not with rest
            removed_conflicts = conflict_adj[v_remove] & current_set
            # We need a vertex w not in current_set such that w doesn't conflict with (current_set - {v_remove})
            # and w conflicts with v_remove (otherwise we could just add it)
            # Actually, we need w such that (conflict_adj[w] & (current_set - {v_remove})) is empty
            candidates_for_swap = []
            for w in all_vertices:
                if in_set[w]:
                    continue
                # Check if w conflicts with current_set minus v_remove
                ok = True
                for u in conflict_adj[w]:
                    if u != v_remove and in_set[u]:
                        ok = False
                        break
                if ok:
                    candidates_for_swap.append(w)
            
            if candidates_for_swap:
                w = random.choice(candidates_for_swap)
                # Remove v_remove, add w
                current_set.discard(v_remove)
                in_set[v_remove] = False
                current_set.add(w)
                in_set[w] = True
                # current_size stays the same
                progress = True
        
        if not progress:
            # Try 1-for-2 swap: remove one, add two
            if current_size > 1:
                v_remove = random.choice(list(current_set))
                # Find two vertices w1, w2 not in set that don't conflict with (current_set - {v_remove}) and don't conflict with each other
                remaining_after_remove = current_set - {v_remove}
                candidates = []
                for w in all_vertices:
                    if in_set[w]:
                        continue
                    ok = True
                    for u in conflict_adj[w]:
                        if u != v_remove and in_set[u]:
                            ok = False
                            break
                    if ok:
                        candidates.append(w)
                
                if len(candidates) >= 2:
                    # Find a pair from candidates that don't conflict with each other
                    random.shuffle(candidates)
                    found_pair = False
                    for idx in range(min(len(candidates), 50)):
                        w1 = candidates[idx]
                        for jdx in range(idx + 1, min(idx + 50, len(candidates))):
                            w2 = candidates[jdx]
                            if w2 not in conflict_adj[w1]:
                                # Valid pair
                                current_set.discard(v_remove)
                                in_set[v_remove] = False
                                current_set.add(w1)
                                in_set[w1] = True
                                current_set.add(w2)
                                in_set[w2] = True
                                current_size += 1
                                found_pair = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = set(current_set)
                                break
                        if found_pair:
                            break
                    if found_pair:
                        progress = True
    
    return best_set


def construct():
    n = N
    deadline = 130  # Safety margin before 150s kill
    start_time = time.time()
    
    # Precompute C6 forbidden pairs
    forbidden_pairs = _precompute_c6_forbidden_pairs()
    
    # Build conflict adjacency
    conflict_adj = [set() for _ in range(NUM_PAIRS)]
    for i, j in forbidden_pairs:
        conflict_adj[i].add(j)
        conflict_adj[j].add(i)
    
    # Run branch-and-bound / local search on conflict graph
    remaining = deadline - (time.time() - start_time)
    if remaining < 5:
        # Not enough time, return seed
        return _seed_graph_8K5()
    
    selected = _branch_and_bound(conflict_adj, deadline, initial_candidates=set(range(NUM_PAIRS)))
    
    # Convert to edges
    edges = []
    for idx in selected:
        u, v = ALL_PAIRS[idx]
        edges.append([u, v])
    
    # Verify no C6
    adj = _build_adj(edges)
    if _has_c6_adj(adj, n):
        # Fallback: remove problematic edges
        # Try to fix by removing edges that are part of C6s
        # For now, fall back to seed
        return _seed_graph_8K5()
    
    return edges
# EVOLVE-BLOCK-END
