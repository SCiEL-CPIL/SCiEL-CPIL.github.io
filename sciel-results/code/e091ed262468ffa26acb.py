"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle_on_subset(vertices, used_edges, adj, deadline, min_len=3, max_len=None):
    """Find a simple cycle using exactly the given vertex set, avoiding used_edges."""
    if max_len is None:
        max_len = len(vertices)
    vset = set(vertices)
    n = len(vertices)
    
    if n < min_len:
        return None
        
    # Use Warnsdorff's heuristic: prefer neighbors with fewer available options
    def dfs(path, visited, current, depth):
        if time.time() > deadline:
            return None
        
        # If we have visited all vertices in the subset, check if we can close the cycle
        if depth == n:
            start = path[0]
            if start in adj[current] and frozenset((current, start)) not in used_edges:
                return path[:]
            return None
        
        if depth >= max_len:
            return None
        
        # Get available neighbors
        candidates = []
        for nxt in adj[current]:
            if nxt not in visited and nxt in vset:
                if frozenset((current, nxt)) not in used_edges:
                    # Warnsdorff: count available unvisited neighbors of nxt
                    avail = sum(1 for nn in adj[nxt] if nn not in visited and nn in vset and frozenset((nxt, nn)) not in used_edges)
                    candidates.append((avail, nxt))
        
        # Sort by fewest options first (Warnsdorff)
        candidates.sort(key=lambda x: x[0])
        
        for _, nxt in candidates:
            visited.add(nxt)
            path.append(nxt)
            result = dfs(path, visited, nxt, depth + 1)
            if result is not None:
                return result
            path.pop()
            visited.remove(nxt)
        return None
    
    # Try each vertex as start
    for start in sorted(vertices):
        visited = {start}
        path = [start]
        result = dfs(path, visited, start, 1)
        if result is not None:
            return result
    return None

def _pack_cycles_on_subset(vertices, adj, deadline):
    """Find maximum number of edge-disjoint cycles on the given vertex set."""
    vset = frozenset(vertices)
    n = len(vertices)
    used_edges = set()
    cycles = []
    
    # First, check if the induced subgraph has enough edges
    induced_edges = []
    for u in vertices:
        for v in adj[u]:
            if v in vset and u < v:
                induced_edges.append(frozenset((u, v)))
    
    if len(induced_edges) < n:
        return cycles
    
    # Try to find cycles greedily
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        if attempt > 0:
            # Shuffle to try different orderings
            shuffled = list(vertices)
            random.shuffle(shuffled)
            vertices = shuffled
        
        cycle = _find_cycle_on_subset(vertices, used_edges, adj, deadline)
        if cycle is None:
            break
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(frozenset((u, v)))
        cycles.append(cycle)
    
    return cycles

def construct():
    n = P['n']
    adj = _build_adj()
    
    # Deadline with safety margin
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_result = []
    best_score = 0
    
    # Strategy 1: Try Hamiltonian cycles (all 21 vertices)
    all_vertices = list(range(n))
    hami_cycles = _pack_cycles_on_subset(all_vertices, adj, deadline)
    if len(hami_cycles) > best_score:
        best_score = len(hami_cycles)
        best_result = hami_cycles
    
    # Strategy 2: Try smaller vertex subsets
    # For a subset of size k, max cycles <= floor(induced_edges / k)
    # Try various sizes from 3 to 10
    
    # First, identify promising subsets by looking at dense subgraphs
    # Compute edge density for all subsets of size 3-8
    
    # For efficiency, try specific candidate subsets
    # The graph appears to have a grid-like structure; try contiguous ranges
    
    candidate_subsets = []
    
    # Try all subsets of size 3-6 (combinatorial explosion, so be selective)
    # Size 3: C(21,3) = 1330
    # Size 4: C(21,4) = 5985
    # Size 5: C(21,5) = 20349
    # Size 6: C(21,6) = 54264
    
    # For small sizes, enumerate all
    for k in range(3, 7):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline:
                break
            vset = set(subset)
            # Check if induced subgraph has at least k edges
            edge_count = 0
            for u in subset:
                for v in adj[u]:
                    if v in vset and u < v:
                        edge_count += 1
            if edge_count >= k:
                candidate_subsets.append((k, list(subset)))
    
    # Also try some larger subsets (size 7-10) based on dense regions
    # Try contiguous blocks and other structured subsets
    for size in range(7, 11):
        if time.time() > deadline:
            break
        # Try all contiguous blocks
        for start in range(n - size + 1):
            subset = list(range(start, start + size))
            vset = set(subset)
            edge_count = 0
            for u in subset:
                for v in adj[u]:
                    if v in vset and u < v:
                        edge_count += 1
            if edge_count >= size:
                candidate_subsets.append((size, subset))
        
        # Try some random subsets
        for _ in range(100):
            if time.time() > deadline:
                break
            subset = random.sample(range(n), size)
            vset = set(subset)
            edge_count = 0
            for u in subset:
                for v in adj[u]:
                    if v in vset and u < v:
                        edge_count += 1
            if edge_count >= size:
                candidate_subsets.append((size, subset))
    
    # Sort candidates by potential (edge_count / size) descending
    # But we already filtered for edge_count >= size
    
    # Try each candidate subset
    # Limit total time
    subset_deadline = deadline
    for k, subset in candidate_subsets:
        if time.time() > subset_deadline:
            break
        # Quick upper bound: max cycles <= floor(induced_edges / k)
        vset = set(subset)
        edge_count = 0
        for u in subset:
            for v in adj[u]:
                if v in vset and u < v:
                    edge_count += 1
        max_possible = edge_count // k
        if max_possible <= best_score:
            continue
        
        # Try to pack cycles
        cycles = _pack_cycles_on_subset(subset, adj, subset_deadline)
        if len(cycles) > best_score:
            best_score = len(cycles)
            best_result = cycles
    
    # If we still have time, try more randomized approaches
    if time.time() < deadline - 5.0:
        for trial in range(200):
            if time.time() > deadline - 5.0:
                break
            # Pick a random subset of size 3-8
            k = random.randint(3, 8)
            subset = random.sample(range(n), k)
            vset = set(subset)
            edge_count = 0
            for u in subset:
                for v in adj[u]:
                    if v in vset and u < v:
                        edge_count += 1
            if edge_count < k:
                continue
            max_possible = edge_count // k
            if max_possible <= best_score:
                continue
            cycles = _pack_cycles_on_subset(subset, adj, deadline - 5.0)
            if len(cycles) > best_score:
                best_score = len(cycles)
                best_result = cycles
    
    # Validate result
    if best_result:
        vertex_set = set(best_result[0])
        for cycle in best_result:
            if set(cycle) != vertex_set:
                best_result = []
                break
        # Check edge disjointness
        if best_result:
            seen_edges = set()
            valid = True
            for cycle in best_result:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = frozenset((u, v))
                    if e in seen_edges:
                        valid = False
                        break
                    seen_edges.add(e)
                if not valid:
                    break
            if not valid:
                best_result = []
    
    # Fallback: return Hamiltonian cycles if we found any
    if not best_result and hami_cycles:
        best_result = hami_cycles
    
    # Final fallback: find any single cycle
    if not best_result:
        # Try to find any cycle of length 3+
        for k in range(3, 10):
            if time.time() > deadline:
                break
            for subset in itertools.combinations(range(n), k):
                if time.time() > deadline:
                    break
                vset = set(subset)
                edge_count = 0
                for u in subset:
                    for v in adj[u]:
                        if v in vset and u < v:
                            edge_count += 1
                if edge_count >= k:
                    cycle = _find_cycle_on_subset(list(subset), set(), adj, deadline)
                    if cycle is not None:
                        best_result = [cycle]
                        break
            if best_result:
                break
    
    # Convert to proper types
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
