"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _build_bitmasks(n, adj):
    """Build bitmask representation of adjacency for fast clique checking."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks


def _has_k5_fast(n, adj, masks):
    """Check if graph has a clique of size 5 using bitmask operations."""
    # For each vertex v, check if its neighborhood contains a K4
    # A K4 in neigh(v) means 4 vertices all mutually adjacent
    
    for v in range(n):
        neigh_mask = masks[v]
        # Get list of neighbors
        neigh = []
        m = neigh_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 4:
            continue
        
        # Check all 4-combinations of neigh for being a clique
        # For efficiency, use bitmask intersection
        for i in range(len(neigh)):
            for j in range(i+1, len(neigh)):
                for k in range(j+1, len(neigh)):
                    # Check if neigh[i], neigh[j], neigh[k] are mutually adjacent
                    if not (masks[neigh[i]] & (1 << neigh[j]) and 
                            masks[neigh[i]] & (1 << neigh[k]) and
                            masks[neigh[j]] & (1 << neigh[k])):
                        continue
                    # Now find a 4th vertex in neigh adjacent to all three
                    # Common neighbors of these three within neigh
                    common = neigh_mask & masks[neigh[i]] & masks[neigh[j]] & masks[neigh[k]]
                    # Check if any of these common neighbors is in neigh
                    # (it should be, since we intersected with neigh_mask)
                    if common:
                        return True
    return False


def _has_ind5_fast(n, adj, masks):
    """Check if graph has an independent set of size 5."""
    # Independent set in G is a clique in complement
    # Build complement masks
    all_mask = (1 << n) - 1
    comp_masks = [all_mask ^ masks[i] ^ (1 << i) for i in range(n)]
    
    # Now check for K5 in complement using same logic
    for v in range(n):
        neigh_mask = comp_masks[v]
        neigh = []
        m = neigh_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
        
        if len(neigh) < 4:
            continue
        
        for i in range(len(neigh)):
            for j in range(i+1, len(neigh)):
                for k in range(j+1, len(neigh)):
                    if not (comp_masks[neigh[i]] & (1 << neigh[j]) and 
                            comp_masks[neigh[i]] & (1 << neigh[k]) and
                            comp_masks[neigh[j]] & (1 << neigh[k])):
                        continue
                    common = neigh_mask & comp_masks[neigh[i]] & comp_masks[neigh[j]] & comp_masks[neigh[k]]
                    if common:
                        return True
    return False


def _is_valid_fast(n, adj, masks):
    """Check if graph is (5,5)-free using fast bitmask methods."""
    if _has_k5_fast(n, adj, masks):
        return False
    if _has_ind5_fast(n, adj, masks):
        return False
    return True


def _try_expand(n, adj, max_n, time_limit, deadline):
    """Try to expand the graph by adding vertices and flipping edges."""
    start_time = time.time()
    best_n = n
    best_adj = [s.copy() for s in adj]
    best_masks = _build_bitmasks(n, adj)
    
    current_n = n
    current_adj = [s.copy() for s in adj]
    current_masks = _build_bitmasks(n, adj)
    
    # Strategy: alternate between adding vertices and local search (edge flipping)
    add_attempts = 0
    flip_attempts = 0
    max_add_attempts = 500
    max_flip_attempts = 5000
    
    while current_n < max_n and add_attempts < max_add_attempts and flip_attempts < max_flip_attempts:
        if time.time() > deadline:
            break
        
        # Try to add a new vertex
        if add_attempts < max_add_attempts:
            new_vertex = current_n
            new_adj = [s.copy() for s in current_adj]
            new_adj.append(set())
            
            # Try multiple random connection patterns
            found = False
            for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    
                    new_adj[new_vertex] = set()
                    for i in range(current_n):
                        if random.random() < density:
                            new_adj[new_vertex].add(i)
                            new_adj[i].add(new_vertex)
                    
                    new_masks = _build_bitmasks(current_n + 1, new_adj)
                    if _is_valid_fast(current_n + 1, new_adj, new_masks):
                        current_n += 1
                        current_adj = new_adj
                        current_masks = new_masks
                        best_n = current_n
                        best_adj = [s.copy() for s in current_adj]
                        best_masks = current_masks
                        found = True
                        break
                if found:
                    break
            
            if found:
                add_attempts = 0  # Reset counter on success
            else:
                add_attempts += 1
                # If we've tried many times to add, switch to flipping
                if add_attempts > 100:
                    break
        else:
            # Try edge flipping (local search)
            flip_attempts += 1
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j:
                continue
            
            # Flip edge (i, j)
            if j in current_adj[i]:
                current_adj[i].remove(j)
                current_adj[j].remove(i)
            else:
                current_adj[i].add(j)
                current_adj[j].add(i)
            
            new_masks = _build_bitmasks(current_n, current_adj)
            if _is_valid_fast(current_n, current_adj, new_masks):
                current_masks = new_masks
                best_adj = [s.copy() for s in current_adj]
                best_masks = current_masks
            else:
                # Revert
                if j in current_adj[i]:
                    current_adj[i].remove(j)
                    current_adj[j].remove(i)
                else:
                    current_adj[i].add(j)
                    current_adj[j].add(i)
            
            # Occasionally try to add a vertex again
            if flip_attempts % 200 == 0:
                add_attempts = 0
    
    return best_n, best_adj


def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_37_edges()
    adj = _build_adjacency(n, edges)
    
    # Verify it's valid
    masks = _build_bitmasks(n, adj)
    if not _is_valid_fast(n, adj, masks):
        # Should not happen, but just in case
        return n, edges
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to expand
    expanded_n, expanded_adj = _try_expand(n, adj, 46, 280, deadline)
    if expanded_n > best_n:
        best_n = expanded_n
        best_adj = expanded_adj
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if j in best_adj[i]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
