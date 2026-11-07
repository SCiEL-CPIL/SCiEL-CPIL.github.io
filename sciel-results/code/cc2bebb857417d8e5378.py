import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_biclique(adj, left, right):
    """Check if all cross pairs between left and right are edges."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _biclique_edges(left, right):
    """Get the set of edges in this biclique."""
    edges = set()
    for u in left:
        for v in right:
            e = (min(u, v), max(u, v))
            edges.add(e)
    return edges

def _generate_candidates(adj, max_size=5):
    """Generate candidate bicliques by trying to grow small bicliques."""
    n = P['n']
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    candidates = []
    
    # Start with single edges
    for u, v in P['edges']:
        candidates.append(([u], [v]))
    
    # Try to grow bicliques
    seen = set()
    for left, right in list(candidates):
        # Try adding vertices to left
        for w in range(n):
            if w in left or w in right:
                continue
            new_left = left + [w]
            # Check if w is connected to all in right
            ok = True
            for v in right:
                if v not in adj[w]:
                    ok = False
                    break
            if ok:
                key = (tuple(sorted(new_left)), tuple(sorted(right)))
                if key not in seen:
                    seen.add(key)
                    candidates.append((new_left, list(right)))
        
        # Try adding vertices to right
        for w in range(n):
            if w in left or w in right:
                continue
            new_right = right + [w]
            # Check if w is connected to all in left
            ok = True
            for u in left:
                if w not in adj[u]:
                    ok = False
                    break
            if ok:
                key = (tuple(sorted(left)), tuple(sorted(new_right)))
                if key not in seen:
                    seen.add(key)
                    candidates.append((list(left), new_right))
    
    # Filter: keep only valid bicliques with at least 1 edge
    valid_candidates = []
    seen2 = set()
    for left, right in candidates:
        if not left or not right:
            continue
        if not _is_biclique(adj, left, right):
            continue
        # Normalize
        lkey = tuple(sorted(left))
        rkey = tuple(sorted(right))
        if (lkey, rkey) in seen2:
            continue
        seen2.add((lkey, rkey))
        valid_candidates.append((list(left), list(right)))
    
    return valid_candidates

def _greedy_cover(edge_set, candidates, adj, deadline):
    """Greedy set cover: pick biclique covering most uncovered edges."""
    remaining = set(edge_set)
    chosen = []
    
    while remaining and time.time() < deadline:
        best_bq = None
        best_count = -1
        for left, right in candidates:
            bq_edges = _biclique_edges(left, right)
            covered = len(bq_edges & remaining)
            if covered > best_count:
                best_count = covered
                best_bq = (left, right, bq_edges)
        
        if best_bq is None or best_count == 0:
            break
        
        left, right, bq_edges = best_bq
        chosen.append((left, right))
        remaining -= bq_edges
    
    # Handle any remaining edges with individual bicliques
    for u, v in remaining:
        chosen.append(([u], [v]))
    
    return chosen

def _local_search(partition, edge_set, candidates, adj, deadline):
    """Try to improve partition by replacing bicliques."""
    best = list(partition)
    best_count = len(best)
    
    # Try removing each biclique and re-covering its edges greedily
    for i in range(len(best)):
        if time.time() > deadline:
            break
        
        # Remove biclique i
        removed = best[i]
        temp = [bq for j, bq in enumerate(best) if j != i]
        
        # Edges that were covered by removed biclique
        removed_edges = _biclique_edges(removed[0], removed[1])
        
        # Find which edges are now uncovered
        covered = set()
        for left, right in temp:
            covered |= _biclique_edges(left, right)
        
        uncovered = removed_edges - covered
        
        if not uncovered:
            # Can remove this biclique entirely
            candidate_partition = temp
            if len(candidate_partition) < best_count:
                best = candidate_partition
                best_count = len(best)
            continue
        
        # Try to cover uncovered edges with one new biclique
        found = False
        for left, right in candidates:
            bq_edges = _biclique_edges(left, right)
            if uncovered <= bq_edges:
                candidate_partition = temp + [(left, right)]
                if len(candidate_partition) < best_count:
                    best = candidate_partition
                    best_count = len(best)
                found = True
                break
        
        if not found:
            # Cover with individual edges
            new_bqs = []
            for u, v in uncovered:
                new_bqs.append(([u], [v]))
            candidate_partition = temp + new_bqs
            if len(candidate_partition) < best_count:
                best = candidate_partition
                best_count = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = P['n']
    adj = _build_adj()
    
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    # Generate candidates
    candidates = _generate_candidates(adj)
    
    # Greedy cover
    partition = _greedy_cover(edge_set, candidates, adj, deadline)
    
    # Local search
    partition = _local_search(partition, edge_set, candidates, adj, deadline)
    
    # Further local search iterations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_partition = _local_search(partition, edge_set, candidates, adj, deadline)
        if len(new_partition) < len(partition):
            partition = new_partition
            improved = True
    
    # Verify and convert to proper format
    result = []
    for left, right in partition:
        result.append([list(left), list(right)])
    
    # Ensure types are correct
    final = []
    for bq in result:
        left = [int(v) for v in bq[0]]
        right = [int(v) for v in bq[1]]
        final.append([left, right])
    
    return final
# EVOLVE-BLOCK-END
