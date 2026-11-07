"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_biclique(left, right, adj):
    """Check if all cross pairs between left and right are edges."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _biclique_edges(left, right, adj):
    """Return set of edge keys covered by this biclique."""
    edges = set()
    for u in left:
        for v in right:
            if v in adj[u]:
                edges.add(_edge_key(u, v))
    return edges

def _max_biclique_from_seeds(L, R, adj, n, deadline):
    """Grow a biclique from initial sets L and R to maximal (within remaining constraints)."""
    L = set(L)
    R = set(R)
    
    # Remove any vertex that doesn't connect to all on the other side
    changed = True
    while changed:
        changed = False
        # Filter L
        new_L = set()
        for u in L:
            if all(v in adj[u] for v in R):
                new_L.add(u)
            else:
                changed = True
        L = new_L
        
        # Filter R
        new_R = set()
        for v in R:
            if all(u in adj[v] for u in L):
                new_R.add(v)
            else:
                changed = True
        R = new_R
        
        if time.time() > deadline:
            break
    
    # Try to grow
    changed = True
    while changed:
        changed = False
        
        # Try adding to R
        candidates = set()
        for u in L:
            candidates.update(adj[u])
        candidates -= L
        for v in candidates:
            if v not in R and all(u in adj[v] for u in L):
                R.add(v)
                changed = True
        
        # Try adding to L
        candidates = set()
        for v in R:
            candidates.update(adj[v])
        candidates -= R
        for u in candidates:
            if u not in L and all(v in adj[u] for v in R):
                L.add(u)
                changed = True
        
        if time.time() > deadline:
            break
    
    return L, R

def _greedy_partition(edges, adj, n, deadline, rng):
    """Greedy biclique partition with multiple anchor strategies."""
    remaining = set()
    for u, v in edges:
        remaining.add(_edge_key(u, v))
    
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Compute degrees in remaining
        rem_deg = [0] * n
        rem_adj = [set() for _ in range(n)]
        for ek in remaining:
            u, v = ek
            rem_deg[u] += 1
            rem_deg[v] += 1
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Try multiple anchors, pick best biclique
        best_bq = None
        best_size = 0
        
        # Candidate anchors: top by degree, plus some random
        candidates = [(rem_deg[i], i) for i in range(n) if rem_deg[i] > 0]
        candidates.sort(reverse=True)
        top_anchors = [c[1] for c in candidates[:30]]
        
        # Add some random anchors
        all_active = [i for i in range(n) if rem_deg[i] > 0]
        if len(all_active) > 30:
            extra = rng.sample(all_active, min(20, len(all_active) - 30))
            top_anchors.extend(extra)
        
        for anchor in top_anchors:
            if time.time() > deadline:
                break
            
            # Try different initial right sets
            neighbors = rem_adj[anchor]
            
            # Strategy 1: Start with all neighbors
            L, R = _max_biclique_from_seeds({anchor}, neighbors, adj, n, deadline)
            if L and R:
                size = len(L) * len(R)
                # Count actual edges
                actual = 0
                for u in L:
                    for v in R:
                        if v in adj[u]:
                            actual += 1
                if actual > best_size:
                    best_size = actual
                    best_bq = (L, R)
            
            # Strategy 2: Start with subset of neighbors (top by degree)
            if len(neighbors) > 3:
                neigh_list = list(neighbors)
                neigh_list.sort(key=lambda x: rem_deg[x], reverse=True)
                for subset_size in [3, 5, 8]:
                    if len(neigh_list) >= subset_size:
                        subset = set(neigh_list[:subset_size])
                        L, R = _max_biclique_from_seeds({anchor}, subset, adj, n, deadline)
                        if L and R:
                            actual = 0
                            for u in L:
                                for v in R:
                                    if v in adj[u]:
                                        actual += 1
                            if actual > best_size:
                                best_size = actual
                                best_bq = (L, R)
        
        if best_bq is None:
            # Fallback: single edge
            ek = next(iter(remaining))
            u, v = ek
            best_bq = ({u}, {v})
        
        L, R = best_bq
        # Ensure disjointness
        overlap = L & R
        if overlap:
            # Move overlapping vertices to one side
            for v in overlap:
                L.discard(v)
            if not L:
                L, R = R, L
                overlap = L & R
                for v in overlap:
                    R.discard(v)
        
        biclique_edges = _biclique_edges(L, R, adj)
        bicliques.append((list(L), list(R)))
        remaining -= biclique_edges
    
    return bicliques if remaining == set() else None

def _try_merge(bicliques, adj):
    """Try to merge compatible bicliques to reduce count."""
    if len(bicliques) <= 1:
        return bicliques
    
    result = [list(bq) for bq in bicliques]
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(result):
            j = i + 1
            while j < len(result):
                L1, R1 = result[i]
                L2, R2 = result[j]
                
                # Try merging: new biclique is (L1∪L2, R1∪R2) or variations
                # Check if (L1∪L2, R1∪R2) is valid
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                if not (new_L & new_R) and _is_biclique(new_L, new_R, adj):
                    # Verify that the merged biclique covers all edges from both
                    edges1 = _biclique_edges(set(L1), set(R1), adj)
                    edges2 = _biclique_edges(set(L2), set(R2), adj)
                    new_edges = _biclique_edges(new_L, new_R, adj)
                    
                    if edges1 | edges2 == new_edges:
                        result[i] = [list(new_L), list(new_R)]
                        result.pop(j)
                        merged = True
                        continue
                
                # Try (L1∪R2, R1∪L2) if disjoint
                new_L2 = set(L1) | set(R2)
                new_R2 = set(R1) | set(L2)
                if not (new_L2 & new_R2) and _is_biclique(new_L2, new_R2, adj):
                    edges1 = _biclique_edges(set(L1), set(R1), adj)
                    edges2 = _biclique_edges(set(L2), set(R2), adj)
                    new_edges = _biclique_edges(new_L2, new_R2, adj)
                    if edges1 | edges2 == new_edges:
                        result[i] = [list(new_L2), list(new_R2)]
                        result.pop(j)
                        merged = True
                        continue
                
                j += 1
            i += 1
    
    return result

def _validate(bicliques, adj, n, all_edges):
    """Validate that bicliques form a proper partition."""
    # Check each biclique
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        if not _is_biclique(left, right, adj):
            return False
    
    # Check partition
    covered = set()
    for left, right in bicliques:
        for u in left:
            for v in right:
                ek = _edge_key(u, v)
                if ek in all_edges:
                    if ek in covered:
                        return False  # Overlap
                    covered.add(ek)
    
    if covered != all_edges:
        return False
    
    return True

def construct():
    start_time = time.time()
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = _build_adj(edges, n)
    all_edges = set()
    for u, v in edges:
        all_edges.add(_edge_key(u, v))
    
    best_bicliques = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Multiple restarts with different random seeds
    for restart in range(100):
        if time.time() - start_time > DEADLINE:
            break
        
        # Vary the strategy slightly
        current_deadline = start_time + DEADLINE - (time.time() - start_time)
        
        bicliques = _greedy_partition(edges, adj, n, DEADLINE, rng)
        
        if bicliques is not None:
            # Merge pass
            merged = _try_merge(bicliques, adj)
            
            # Validate
            if _validate(merged, adj, n, all_edges):
                if len(merged) < best_count:
                    best_count = len(merged)
                    best_bicliques = merged
    
    # If no valid result, use star partition as fallback
    if best_bicliques is None:
        best_bicliques = []
        for u in range(n):
            neighbors = list(adj[u])
            if neighbors:
                best_bicliques.append(([u], neighbors))
    
    # Convert to proper types
    result = []
    for left, right in best_bicliques:
        left_ints = [int(x) for x in left]
        right_ints = [int(x) for x in right]
        result.append([left_ints, right_ints])
    
    return result

# EVOLVE-BLOCK-END
