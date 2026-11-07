"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _maximal_bicliques(adj):
    """Enumerate all inclusion-wise maximal bicliques."""
    bicliques = []
    n = P['n']
    
    # For each pair of vertices, try to extend to a maximal biclique
    # A biclique is defined by (L, R) where every l in L is connected to every r in R
    # We use a heuristic: start with each edge, then grow
    
    seen = set()
    
    for u, v in P['edges']:
        # Start with L={u}, R={v}
        L = {u}
        R = {v}
        
        # Grow L: add any vertex x such that x is connected to all vertices in R
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and all(r in adj[x] for r in R):
                    L.add(x)
                    changed = True
        
        # Grow R: add any vertex y such that y is connected to all vertices in L
        changed = True
        while changed:
            changed = False
            for y in range(n):
                if y not in R and all(l in adj[y] for l in L):
                    R.add(y)
                    changed = True
        
        # Normalize for deduplication
        L_frozenset = frozenset(L)
        R_frozenset = frozenset(R)
        
        # Check if this biclique is maximal and not a subset of an already found one
        # For efficiency, just check if we've seen this exact (L, R) pair
        key = (L_frozenset, R_frozenset)
        if key in seen:
            continue
        
        # Verify maximality: no vertex can be added to L or R
        # (Already done above, but let's double-check)
        is_maximal = True
        for x in range(n):
            if x not in L and all(r in adj[x] for r in R):
                is_maximal = False
                break
        if is_maximal:
            for y in range(n):
                if y not in R and all(l in adj[y] for l in L):
                    is_maximal = False
                    break
        
        if is_maximal:
            seen.add(key)
            bicliques.append((sorted(L), sorted(R)))
    
    # Also try starting with single vertices as one side
    for u in range(n):
        if not adj[u]:
            continue
        L = {u}
        R = set(adj[u])
        
        # Grow L
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and all(r in adj[x] for r in R):
                    L.add(x)
                    changed = True
        
        # Grow R
        changed = True
        while changed:
            changed = False
            for y in range(n):
                if y not in R and all(l in adj[y] for l in L):
                    R.add(y)
                    changed = True
        
        key = (frozenset(L), frozenset(R))
        if key not in seen:
            # Check maximality
            is_maximal = True
            for x in range(n):
                if x not in L and all(r in adj[x] for r in R):
                    is_maximal = False
                    break
            if is_maximal:
                for y in range(n):
                    if y not in R and all(l in adj[y] for l in L):
                        is_maximal = False
                        break
            if is_maximal:
                seen.add(key)
                bicliques.append((sorted(L), sorted(R)))
    
    return bicliques

def _greedy_partition(bicliques, edge_set):
    """Greedy set cover: repeatedly pick the biclique covering most uncovered edges."""
    uncovered = set(edge_set)
    chosen = []
    
    while uncovered:
        best_biclique = None
        best_count = -1
        best_edges = set()
        
        for L, R in bicliques:
            # Find edges in this biclique that are still uncovered
            biclique_edges = set()
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in uncovered:
                        biclique_edges.add(e)
            
            if len(biclique_edges) > best_count:
                best_count = len(biclique_edges)
                best_biclique = (L, R)
                best_edges = biclique_edges
        
        if best_biclique is None or best_count <= 0:
            # Fallback: pick a single edge
            e = next(iter(uncovered))
            u, v = e
            chosen.append(([u], [v]))
            uncovered.discard(e)
        else:
            L, R = best_biclique
            chosen.append((L, R))
            uncovered -= best_edges
    
    return chosen

def _local_search_improve(partition, edge_set, adj, time_limit=5.0):
    """Try to improve the partition by merging or replacing bicliques."""
    start_time = time.time()
    best_partition = [list(b) for b in partition]
    best_size = len(best_partition)
    
    # Convert to set of covered edges for quick check
    def partition_edges(part):
        edges = set()
        for L, R in part:
            for u in L:
                for v in R:
                    edges.add((min(u, v), max(u, v)))
        return edges
    
    current_edges = partition_edges(best_partition)
    
    # Try pairwise merges
    for i in range(best_size):
        for j in range(i + 1, best_size):
            if time.time() - start_time > time_limit:
                break
            # Try merging biclique i and j
            L1, R1 = best_partition[i]
            L2, R2 = best_partition[j]
            
            # The merged biclique would need L1∪L2 and R1∪R2 to form a valid biclique
            # Check if all cross edges exist
            merged_L = list(set(L1) | set(L2))
            merged_R = list(set(R1) | set(R2))
            
            valid = True
            for u in merged_L:
                for v in merged_R:
                    if v not in adj[u]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check if the union covers exactly the edges of i and j
            new_edges = set()
            for u in merged_L:
                for v in merged_R:
                    new_edges.add((min(u, v), max(u, v)))
            
            old_edges = set()
            for u in L1:
                for v in R1:
                    old_edges.add((min(u, v), max(u, v)))
            for u in L2:
                for v in R2:
                    old_edges.add((min(u, v), max(u, v)))
            
            if new_edges == old_edges:
                new_partition = [best_partition[k] for k in range(best_size) if k not in (i, j)]
                new_partition.append((merged_L, merged_R))
                if len(new_partition) < best_size:
                    best_partition = new_partition
                    best_size = len(best_partition)
                    current_edges = partition_edges(best_partition)
    
    return best_partition

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    adj = _build_adjacency()
    edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Phase 1: Generate candidate bicliques
    bicliques = _maximal_bicliques(adj)
    
    # Phase 2: Greedy partition
    partition = _greedy_partition(bicliques, edge_set)
    
    # Phase 3: Local search improvement
    if time.time() < deadline - 10.0:
        partition = _local_search_improve(partition, edge_set, adj, time_limit=min(5.0, deadline - time.time() - 2.0))
    
    # Convert to required format: list of [left, right]
    result = []
    for L, R in partition:
        result.append([list(L), list(R)])
    
    # Verify validity
    covered = set()
    for L, R in result:
        for u in L:
            for v in R:
                covered.add((min(u, v), max(u, v)))
    
    if covered != edge_set:
        # Fallback to seed program
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    return result
# EVOLVE-BLOCK-END
