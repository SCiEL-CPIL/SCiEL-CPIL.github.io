"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 110.0  # safety margin before 150s kill

def _precompute_subsets():
    """Precompute all 8-subsets and their partitions."""
    n = 16
    all_subs = list(combinations(range(n), 8))
    # For each subset, precompute the 70 partitions (A, B)
    # A is a tuple of 4, B is the other 4
    partitions = []
    for s in all_subs:
        s_set = set(s)
        parts = []
        for a in combinations(s, 4):
            a_set = set(a)
            b = tuple(v for v in s if v not in a_set)
            parts.append((a, b))
        partitions.append(parts)
    return all_subs, partitions

# Precompute globally
ALL_SUBS, PARTITIONS = _precompute_subsets()

def _contains_q3_fast(adj_masks):
    """
    Check if graph contains Q3.
    adj_masks: list of 16 integers, bit i set if edge (i, j) exists? 
    No, let's use adj_masks[i] as bitmask of neighbors of i.
    """
    n = 16
    for idx, s in enumerate(ALL_SUBS):
        parts = PARTITIONS[idx]
        for a, b in parts:
            # Check if missing edges between A and B form a matching
            # Missing edge (u, v) means not adj[u][v]
            # We need to ensure:
            # 1. No vertex in A has >1 missing edge to B
            # 2. No vertex in B has >1 missing edge from A
            
            # Bitmasks of missing edges for each a in A
            # missing_mask[a_idx] is a 4-bit mask of which b in B is NOT connected
            
            # Optimization: 
            # For each a in A, count missing edges to B.
            # If any count > 1, skip partition.
            # Then check if any b is missing from > 1 a.
            
            missing_counts_b = [0, 0, 0, 0] # for b[0]..b[3]
            valid = True
            
            for i in range(4):
                a_v = a[i]
                # Mask of B vertices
                # b[0]..b[3]
                # We need to check adj[a_v][b[j]]
                
                # Build mask of neighbors of a_v in B
                # adj_masks[a_v] is full 16-bit mask.
                # We need to extract bits for b[0]..b[3]
                
                # To be fast, precompute bit positions?
                # Or just loop.
                
                missing_in_b = 0
                for j in range(4):
                    b_v = b[j]
                    if not (adj_masks[a_v] >> b_v) & 1:
                        missing_in_b |= (1 << j)
                        missing_counts_b[j] += 1
                        if missing_counts_b[j] > 1:
                            valid = False
                            break
                
                if not valid:
                    break
                    
                if missing_in_b > 1:
                    valid = False
                    break
            
            if valid:
                # Check if missing edges form a matching
                # We already checked missing_counts_b <= 1
                # And we checked missing_in_b <= 1 for each a
                # So yes, it's a matching.
                return True
    return False

def _find_q3_edge(adj_masks):
    """Find an edge that is part of a Q3, or None if Q3-free."""
    n = 16
    for idx, s in enumerate(ALL_SUBS):
        parts = PARTITIONS[idx]
        for a, b in parts:
            missing_counts_b = [0, 0, 0, 0]
            valid = True
            missing_edges = []
            
            for i in range(4):
                a_v = a[i]
                missing_in_b = 0
                for j in range(4):
                    b_v = b[j]
                    if not (adj_masks[a_v] >> b_v) & 1:
                        missing_in_b |= (1 << j)
                        missing_counts_b[j] += 1
                        missing_edges.append((a_v, b_v))
                        if missing_counts_b[j] > 1:
                            valid = False
                            break
                if not valid:
                    break
                if missing_in_b > 1:
                    valid = False
                    break
            
            if valid:
                # We found a Q3. 
                # The "missing edges" are the ones NOT in the Q3.
                # The edges IN the Q3 are all other edges between A and B.
                # We can remove any edge in the Q3.
                # Let's pick one.
                # Pick the first edge (a[0], b[0])? 
                # It must be present in G.
                # Since missing edges form a matching, (a[0], b[0]) is present 
                # UNLESS it is in the matching.
                # If it is in the matching, it's missing.
                # So we need to pick an edge that is PRESENT.
                
                # Let's just pick a random edge from the 16 possible, 
                # ensuring it's not missing.
                
                candidates = []
                for i in range(4):
                    for j in range(4):
                        if not ((missing_in_b if False else 0) >> j) & 1: # This logic is broken
                            pass
                
                # Simpler: 
                # The missing edges are in `missing_edges`.
                # Any edge in A x B not in `missing_edges` is in the Q3.
                # Pick one.
                
                # If missing_edges is empty, pick (a[0], b[0]).
                # If not, pick one that is not in missing_edges.
                
                missing_set = set(missing_edges)
                for i in range(4):
                    for j in range(4):
                        if (a[i], b[j]) not in missing_set:
                            return (a[i], b[j])
                
                # If all are missing? Impossible, max 4 missing.
                
    return None

def _adj_to_masks(adj):
    masks = [0] * 16
    for i in range(16):
        for j in range(i + 1, 16):
            if adj[i][j]:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks

def _masks_to_adj(masks):
    adj = [[False] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(i + 1, 16):
            if (masks[i] >> j) & 1:
                adj[i][j] = adj[j][i] = True
    return adj

def construct():
    start_time = time.time()
    n = 16
    deadline = start_time + DEADLINE
    
    best_adj = None
    best_count = 0
    
    # Start with K16
    adj = [[True] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = False
        
    masks = _adj_to_masks(adj)
    
    # Greedy removal
    while time.time() < deadline - 5:
        edge = _find_q3_edge(masks)
        if edge is None:
            break
        u, v = edge
        # Remove edge
        adj[u][v] = adj[v][u] = False
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        
    # Check if we succeeded
    if _contains_q3_fast(masks):
        # If we ran out of time and still have Q3, 
        # we must return a valid graph. 
        # We can't just return this.
        # We should have stopped earlier.
        # For now, if this happens, we try to remove more edges quickly?
        # Or just return the seed's K3,13?
        # Let's try to force validity by removing random edges until valid or timeout.
        while _contains_q3_fast(masks) and time.time() < deadline:
            u, v = random.sample(range(n), 2)
            if adj[u][v]:
                adj[u][v] = adj[v][u] = False
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
    
    if _contains_q3_fast(masks):
        # Fallback to a known valid graph (e.g., empty or sparse)
        # To ensure score > 0, return a small valid graph.
        adj = [[False] * n for _ in range(n)]
        masks = _adj_to_masks(adj)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
