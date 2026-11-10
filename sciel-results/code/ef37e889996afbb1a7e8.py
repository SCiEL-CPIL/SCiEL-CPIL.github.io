import random
import itertools
import time
from collections import defaultdict, deque

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge list and mapping
edges = []
edge_map = {}
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        edges.append((u, v))
        edge_map[(u, v)] = idx
        idx += 1

# For each edge, precompute which 8-vertex subsets it belongs to
edge_to_subsets = defaultdict(list)
for combo in itertools.combinations(range(N), 8):
    combo_set = set(combo)
    for u in combo:
        for v in combo:
            if u < v:
                e_idx = edge_map[(u, v)]
                edge_to_subsets[e_idx].append(combo)

# Precompute for each 8-vertex subset, the internal edges
subset_internal = {}
for combo in itertools.combinations(range(N), 8):
    internal = []
    for i in range(8):
        for j in range(i + 1, 8):
            u, v = combo[i], combo[j]
            internal.append(edge_map[(u, v)])
    subset_internal[combo] = internal

# Precompute all Q3 patterns for each 8-vertex subset
# A Q3 on 8 vertices is a 3-regular bipartite graph with 12 edges
# Equivalent to K_{4,4} minus a perfect matching

def get_q3_patterns(vertices):
    """Get all Q3 edge sets (as frozensets of global edge indices) for these 8 vertices."""
    v_map = {v: i for i, v in enumerate(vertices)}
    patterns = set()
    
    # All bipartitions into two sets of 4
    for left_set in itertools.combinations(range(8), 4):
        left = set(left_set)
        right = set(range(8)) - left
        left_list = sorted(left)
        right_list = sorted(right)
        
        # All perfect matchings: permutations of right_list
        for perm in itertools.permutations(right_list):
            # Removed edges: (left_list[i], perm[i])
            removed_local = set()
            for i in range(4):
                u, v = left_list[i], perm[i]
                removed_local.add((min(u, v), max(u, v)))
            
            # Build the 12-edge set
            edge_set = set()
            for u in left:
                for v in right:
                    if (min(u, v), max(u, v)) not in removed_local:
                        gu, gv = vertices[u], vertices[v]
                        e_idx = edge_map[(min(gu, gv), max(gu, gv))]
                        edge_set.add(e_idx)
            
            if len(edge_set) == 12:
                patterns.add(frozenset(edge_set))
    
    return list(patterns)

# Precompute all Q3 patterns
all_q3_by_subset = {}
for combo in itertools.combinations(range(N), 8):
    pats = get_q3_patterns(combo)
    if pats:
        all_q3_by_subset[combo] = pats

# For fast conflict detection: when we add an edge, only check subsets containing that edge
# Precompute: for each edge, which Q3 patterns it participates in
edge_to_q3_patterns = defaultdict(list)
for combo, pats in all_q3_by_subset.items():
    for pat in pats:
        for e_idx in pat:
            edge_to_q3_patterns[e_idx].append(pat)

def check_q3_conflict(edge_set, added_edge):
    """Check if adding added_edge to edge_set creates a Q3. 
    Only check Q3 patterns that contain added_edge."""
    for pat in edge_to_q3_patterns[added_edge]:
        if pat.issubset(edge_set):
            return True, pat
    return False, None

def construct():
    start_time = time.time()
    timeout = 115
    
    best_edge_set = set()
    best_count = 0
    
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
        
        random.seed(restart * 7919 + 1)
        
        # Random edge ordering
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        current_set = set()
        
        # Greedy construction
        for e_idx in edge_order:
            if e_idx in current_set:
                continue
            new_set = current_set | {e_idx}
            conflict, _ = check_q3_conflict(new_set, e_idx)
            if not conflict:
                current_set = new_set
        
        # Local search phase
        no_improve = 0
        while no_improve < 30 and time.time() - start_time < timeout:
            improved = False
            
            # Try to add missing edges
            missing = [e for e in range(NUM_EDGES) if e not in current_set]
            random.shuffle(missing)
            
            for e_idx in missing[:50]:
                new_set = current_set | {e_idx}
                conflict, _ = check_q3_conflict(new_set, e_idx)
                if not conflict:
                    current_set = new_set
                    improved = True
                    break
            
            if improved:
                no_improve = 0
                if len(current_set) > best_count:
                    best_count = len(current_set)
                    best_edge_set = set(current_set)
            else:
                no_improve += 1
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_edge_set = set(current_set)
    
    result = []
    for e_idx in best_edge_set:
        result.append(list(edges[e_idx]))
    
    return result
