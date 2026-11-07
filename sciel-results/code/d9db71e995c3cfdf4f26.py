"""Biclique partition solver using greedy edge-cover with local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_EDGES = set(frozenset(e) for e in P['edges'])

def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def validate_partition(bicliques):
    """Validate that the partition is correct."""
    covered_edges = set()
    for L, R in bicliques:
        Lset, Rset = set(L), set(R)
        if not Lset or not Rset:
            return False
        if Lset & Rset:
            return False
        for u in Lset:
            for v in Rset:
                e = frozenset((u, v))
                if e not in ALL_EDGES:
                    return False
                if e in covered_edges:
                    return False
                covered_edges.add(e)
    return covered_edges == ALL_EDGES

def find_max_right(L, remaining_edges, adj):
    """Find the largest R such that L x R ⊆ remaining_edges."""
    Lset = set(L)
    R_candidates = set(range(N)) - Lset
    valid_R = set()
    for v in R_candidates:
        valid = True
        for u in Lset:
            if frozenset((u, v)) not in remaining_edges:
                valid = False
                break
        if valid:
            valid_R.add(v)
    return valid_R

def greedy_partition(adj, deadline):
    """Greedy: repeatedly find the largest biclique covering uncovered edges."""
    remaining = set(ALL_EDGES)
    bicliques = []
    
    while remaining and time.time() < deadline:
        best_biclique = None
        best_edge_count = 0
        
        # Try L of size 1, 2, 3
        for L_size in [1, 2, 3]:
            if L_size == 1:
                candidates = [(u,) for u in range(N)]
            elif L_size == 2:
                candidates = [(u, v) for u in range(N) for v in range(u+1, N)]
            else:
                candidates = [(u, v, w) for u in range(N) for v in range(u+1, N) for w in range(v+1, N)]
            
            for L_tuple in candidates:
                L = set(L_tuple)
                # Check if all edges within L are in remaining (not required, but L x R must be)
                # Actually, we just need L x R ⊆ remaining
                R = find_max_right(L, remaining, adj)
                if R:
                    edge_count = len(L) * len(R)
                    if edge_count > best_edge_count:
                        best_edge_count = edge_count
                        best_biclique = (L, R)
            
            # If we found a good one with this L_size, we can break
            # but let's continue to try larger L for better coverage
            if best_edge_count >= 6:  # Found a decent biclique
                break
        
        if best_biclique is None:
            # Fallback: take one edge
            e = next(iter(remaining))
            u, v = list(e)
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        bicliques.append((list(L), list(R)))
        
        # Remove covered edges
        for u in L:
            for v in R:
                remaining.discard(frozenset((u, v)))
    
    return bicliques

def can_merge(b1, b2):
    """Check if two bicliques can be merged without violating constraints."""
    L1, R1 = set(b1[0]), set(b1[1])
    L2, R2 = set(b2[0]), set(b2[1])
    
    # Check disjointness
    if L1 & L2 or R1 & R2 or L1 & R2 or L2 & R1:
        return False
    
    # Check all cross edges exist
    for u in L1:
        for v in R2:
            if frozenset((u, v)) not in ALL_EDGES:
                return False
    for u in L2:
        for v in R1:
            if frozenset((u, v)) not in ALL_EDGES:
                return False
    
    return True

def merge_bicliques(bicliques, deadline):
    """Try to merge compatible bicliques."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            for j in range(i+1, n):
                if can_merge(bicliques[i], bicliques[j]):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    new_biclique = (list(set(L1) | set(L2)), list(set(R1) | set(R2)))
                    new_bicliques = []
                    for k in range(n):
                        if k != i and k != j:
                            new_bicliques.append(bicliques[k])
                    new_bicliques.append(new_biclique)
                    bicliques = new_bicliques
                    changed = True
                    break
            if changed:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    adj = build_adj()
    
    # Greedy partition
    bicliques = greedy_partition(adj, deadline)
    
    # Merge compatible bicliques
    bicliques = merge_bicliques(bicliques, deadline)
    
    # Convert to proper format
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
# EVOLVE-BLOCK-END
