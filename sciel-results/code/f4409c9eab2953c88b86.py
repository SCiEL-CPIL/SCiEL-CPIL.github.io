"""Hexagon-free graph construction on 40 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 40

def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_c6_adj(adj, n):
    """Check if graph with adjacency sets has a 6-cycle (u<v for uniqueness)."""
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

def _creates_c6(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    A C6 through edge (u,v): u-v-w-x-y-z-u where w,x,y,z are distinct and not u,v.
    """
    for w in adj[v]:
        if w == u:
            continue
        for x in adj[w]:
            if x == u or x == v or x == w:
                continue
            for y in adj[x]:
                if y == u or y == v or y == w or y == x:
                    continue
                for z in adj[y]:
                    if z == u or z == v or z == w or z == x or z == y:
                        continue
                    if u in adj[z]:
                        return True
    return False

def _seed_fano():
    """
    8 groups of 5 vertices. Fano plane defines which pairs of groups are connected.
    Fano plane: 7 points, 7 lines, each line has 3 points, each point on 3 lines.
    We use 8 groups: 7 Fano groups + 1 extra group.
    
    Actually, let me use a different structure. The Fano plane has 7 points.
    For 8 groups, I'll use 7 Fano groups connected according to Fano lines,
    plus 1 group connected to all others (or a subset).
    
    Fano plane lines (0-indexed points 0..6):
    Lines: {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    
    Two groups are connected if they share a line (i.e., both points are on some Fano line).
    In the Fano plane, any two distinct points are on exactly one line.
    So ALL pairs of Fano groups are connected! That gives K7 between groups.
    
    That's too many edges. Let me think differently.
    
    Alternative: Use the Fano plane as a 7-vertex graph where edges represent
    NON-connections. No, that's also not right.
    
    Let me try: 8 groups of 5. For inter-group edges, use a random bipartite-like
    structure. Actually, let me just use a good random seed with local search.
    
    Better structural seed: Start with a complete 8-partite graph with parts of size 5,
    but remove some inter-group edges to break C6s, then add intra-group edges.
    
    Actually the simplest strong seed: random graph G(40, 0.3) filtered for C6-free,
    then greedy fill. Let me just do a good local search.
    """
    # Use a structured seed: 8 cliques of 5 (K5 each) = 100 edges, C6-free.
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges

def _local_search_structured(n, seed_val, time_limit):
    """
    Local search with:
    1. Greedy fill (all non-edges sorted by degree sum)
    2. Systematic 1-1 swap (remove edge, try all non-edges)
    3. Random perturbations to escape local optima
    """
    random.seed(seed_val)
    start_time = time.time()
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    # Initialize with seed
    edges = _seed_fano()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    current_count = len(edge_set)
    best_count = current_count
    best_edges = list(edge_set)
    
    def try_add(u, v):
        nonlocal current_count
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        if _creates_c6(adj, u, v):
            return False
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        current_count += 1
        return True
    
    def greedy_fill():
        """Try to add all possible non-edges, sorted by degree sum descending."""
        nonlocal current_count, best_count, best_edges
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set and (j,i) not in edge_set]
        non_edges.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]), reverse=True)
        added_any = True
        while added_any:
            added_any = False
            for (u, v) in non_edges:
                if time.time() - start_time > time_limit:
                    return
                if (u, v) in edge_set:
                    continue
                if try_add(u, v):
                    added_any = True
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = list(edge_set)
                    non_edges.remove((u, v))
    
    def systematic_swap():
        """Try each edge removal + best non-edge addition."""
        nonlocal current_count, best_count, best_edges
        edge_list = list(edge_set)
        for (u, v) in edge_list:
            if time.time() - start_time > time_limit:
                return
            if (u, v) not in edge_set:
                continue
            # Remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            edge_set.discard((u, v))
            current_count -= 1
            
            # Try to add best non-edge
            added = False
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set and (j,i) not in edge_set]
            non_edges.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]), reverse=True)
            for (a, b) in non_edges:
                if try_add(a, b):
                    added = True
                    break
            
            if not added:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                current_count += 1
            else:
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(edge_set)
    
    def random_swap():
        """Random edge removal + random non-edge addition."""
        nonlocal current_count, best_count, best_edges
        if not edge_set:
            return
        (u, v) = random.choice(list(edge_set))
        adj[u].discard(v)
        adj[v].discard(u)
        edge_set.discard((u, v))
        current_count -= 1
        
        added = False
        for _ in range(50):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) in edge_set:
                continue
            if try_add(a, b):
                added = True
                break
        
        if not added:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            current_count += 1
        else:
            if current_count > best_count:
                best_count = current_count
                best_edges = list(edge_set)
    
    # Phase 1: Greedy fill
    greedy_fill()
    
    # Phase 2: Alternating systematic and random swaps
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget <= 0:
        return [[u, v] for u, v in best_edges], best_count
    
    # Interleave: do rounds of systematic swap, then random swaps
    while time.time() - start_time < time_limit - 2:
        # Systematic swap pass
        if time.time() - start_time < time_limit - 10:
            systematic_swap()
        # Greedy fill again
        greedy_fill()
        # Some random swaps to escape local optima
        for _ in range(20):
            if time.time() - start_time > time_limit - 2:
                break
            random_swap()
        # Final greedy fill
        greedy_fill()
    
    return [[u, v] for u, v in best_edges], best_count

def _multi_start(n, time_limit):
    """Run local search from multiple starting points."""
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    seeds = [42, 123, 7, 99, 256, 1000, 2024, 31415, 2718, 1618]
    num_seeds = min(len(seeds), max(1, (time_limit - 10) // 10))
    
    for i, seed_val in enumerate(seeds[:num_seeds]):
        if time.time() - start_time > time_limit - 15:
            break
        per_time = min(12, max(3, time_limit - 15 - (time.time() - start_time)))
        edges, count = _local_search_structured(n, seed_val, per_time)
        if count > best_count:
            best_count = count
            best_edges = edges
    
    return best_edges, best_count

def construct():
    n = N
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Multi-start local search
    edges, count = _multi_start(n, time_limit=deadline - 10)
    
    # Final verification: ensure no C6
    if count > 0:
        adj = _build_adj(edges)
        if _has_c6_adj(adj, n):
            # Fallback to seed
            return _seed_fano()
    
    return edges
# EVOLVE-BLOCK-END
