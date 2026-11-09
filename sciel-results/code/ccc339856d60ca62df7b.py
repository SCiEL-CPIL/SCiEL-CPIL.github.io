import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _generate_q3_patterns():
    base_q3 = frozenset([(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)])
    patterns = set()
    from itertools import permutations
    for p in permutations(range(8)):
        mapped_edges = []
        for u, v in base_q3:
            mu, mv = p[u], p[v]
            if mu > mv:
                mu, mv = mv, mu
            mapped_edges.append((mu, mv))
        patterns.add(frozenset(mapped_edges))
    return list(patterns)

_Q3_PATTERNS = _generate_q3_patterns()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n = 16
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))
            
    # Generate all Q3s in K16
    # Represent each Q3 by (subset_tuple, pattern_index)
    # subset_tuple is 8 sorted ints.
    # pattern_index is 0..839.
    
    # To map pattern to edges:
    # We need to map canonical 0..7 to the subset's vertices.
    # Let subset = (v0, v1, ..., v7).
    # Canonical edge (a, b) maps to (subset[a], subset[b]).
    
    q3_list = []
    edge_to_q3s = {}
    
    all_subsets = list(combinations(range(n), 8))
    
    for subset in all_subsets:
        # For each pattern, add it to q3_list
        # And update edge_to_q3s
        for p_idx, pattern in enumerate(_Q3_PATTERNS):
            q3_id = len(q3_list)
            q3_list.append((subset, p_idx))
            
            for u, v in pattern:
                gu, gv = subset[u], subset[v]
                if gu > gv:
                    gu, gv = gv, gu
                e = (gu, gv)
                if e not in edge_to_q3s:
                    edge_to_q3s[e] = []
                edge_to_q3s[e].append(q3_id)
                
    # Initialize counts
    edge_counts = {e: 0 for e in edges}
    active = [True] * len(q3_list)
    
    for q3_id, (subset, p_idx) in enumerate(q3_list):
        for u, v in _Q3_PATTERNS[p_idx]:
            gu, gv = subset[u], subset[v]
            if gu > gv:
                gu, gv = gv, gu
            e = (gu, gv)
            edge_counts[e] += 1
            
    # Greedy Deletion
    # We need to find the edge with max count.
    # Instead of scanning all edges, we can just scan. 120 edges is small.
    
    while True:
        if _elapsed() > _DEADLINE:
            break
            
        # Find edge with max count
        max_edge = None
        max_count = 0
        for e in edges:
            if edge_counts[e] > max_count:
                max_count = edge_counts[e]
                max_edge = e
                
        if max_edge is None or max_count == 0:
            break
            
        # Remove max_edge
        edges.discard(max_edge)
        edge_counts[max_edge] = 0
        
        # Update counts for Q3s containing max_edge
        for q3_id in edge_to_q3s.get(max_edge, []):
            if active[q3_id]:
                active[q3_id] = False
                subset, p_idx = q3_list[q3_id]
                for u, v in _Q3_PATTERNS[p_idx]:
                    gu, gv = subset[u], subset[v]
                    if gu > gv:
                        gu, gv = gv, gu
                    e = (gu, gv)
                    if e != max_edge:
                        edge_counts[e] -= 1
                        
    return [[u, v] for (u, v) in edges]
