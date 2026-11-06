import time
import random
from itertools import combinations

# Precompute Q3 masks
# Q3 vertices: 0..7
# Edges of Q3:
# 0-1, 0-2, 0-4
# 1-3, 1-5
# 2-3, 2-6
# 4-5, 4-6
# 3-7, 5-7, 6-7
# Let's generate all 840 masks programmatically to be safe.

def generate_q3_masks():
    masks = set()
    # Q3 is bipartite. Parts A={0,1,2,3}, B={4,5,6,7} is one coloring.
    # Actually, standard Q3:
    # 000, 001, 010, 011, 100, 101, 110, 111
    # Vertices 0..7.
    # Edge if Hamming distance 1.
    v = list(range(8))
    edges = []
    for i in range(8):
        for j in range(i+1, 8):
            # Check Hamming distance
            diff = bin(i ^ j).count('1')
            if diff == 1:
                edges.append((i, j))
    
    # There are 12 edges.
    # We need to find all subsets of 8 vertices in K8 that form Q3.
    # This is equivalent to finding all permutations of 0..7 that map to a valid Q3 embedding.
    # Since Q3 is vertex-transitive and edge-transitive, we can just generate all isomorphic copies.
    # Number of automorphisms is 48. Number of embeddings is 8! / 48 = 840.
    
    base_edges = set(edges)
    
    seen = set()
    for perm in set(permutations(range(8))):
        # Map base edges to new vertices
        new_edges = set()
        for u, v in base_edges:
            nu, nv = perm[u], perm[v]
            if nu > nv:
                nu, nv = nv, nu
            new_edges.add((nu, nv))
        
        # Convert to bitmask for K8 (28 edges)
        mask = 0
        for u, v in new_edges:
            # Index for edge (u,v) in K8
            # 0-1: 0, 0-2: 1, ...
            # Formula: u * 7 - (u*(u+1)//2) + v - u - 1 ? 
            # Simpler: precompute index map
            idx = edge_index[u][v]
            mask |= (1 << idx)
        
        if mask not in seen:
            seen.add(mask)
            masks.add(mask)
            
    return list(masks)

# Precompute edge indices for K8
edge_index = {}
idx = 0
for u in range(8):
    for v in range(u+1, 8):
        edge_index[(u, v)] = idx
        idx += 1

# Generate masks
# This might take a moment, but it's one-time.
# 8! = 40320 permutations. 40320 * 12 edges = ~500k ops. Fast.

# We will do this at module load or inside construct with caching.
# Since construct is called once, we can do it inside.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precomputation
    q3_masks = generate_q3_masks()
    # q3_masks is a list of 840 integers (bitmasks of 28 bits)
    
    # List of all 8-vertex subsets of 0..15
    all_8_subsets = list(combinations(range(16), 8))
    
    # Precompute for each 8-subset, the mapping from local vertex 0..7 to global vertex
    # And the edge index mapping for the induced K8
    # Actually, we can just compute the mask on the fly.
    
    def get_induced_mask(subset, graph_edges_set):
        # subset is a tuple of 8 global vertices
        # map local 0..7 to global
        local_to_global = {i: subset[i] for i in range(8)}
        global_to_local = {v: i for i, v in enumerate(subset)}
        
        mask = 0
        for u, v in graph_edges_set:
            if u in global_to_local and v in global_to_local:
                lu = global_to_local[u]
                lv = global_to_local[v]
                if lu > lv:
                    lu, lv = lv, lu
                idx = edge_index[(lu, lv)]
                mask |= (1 << idx)
        return mask

    def find_q3(graph_edges_set):
        # Iterate subsets in random order
        subsets_copy = all_8_subsets[:]
        random.shuffle(subsets_copy)
        
        for subset in subsets_copy:
            mask = get_induced_mask(subset, graph_edges_set)
            # Quick check: count bits
            if mask.bit_count() < 12:
                continue
            
            # Check if any q3 mask is a subset of mask
            for q3_mask in q3_masks:
                if (mask & q3_mask) == q3_mask:
                    # Found Q3
                    # Return the set of global edges
                    edges_in_q3 = set()
                    # We need to map the bits of q3_mask back to global edges
                    # This is inefficient. 
                    # Better: return the subset and the q3_mask, and let the caller figure out edges?
                    # Or just return one edge from it to remove.
                    
                    # Let's find one edge to remove.
                    # Iterate bits in q3_mask
                    m = q3_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        # Get local edge from idx
                        # Reconstruct local edge from idx
                        # We need an inverse of edge_index
                        # Let's precompute edge_tuples for K8
                        u, v = edge_tuples[idx]
                        gu, gv = local_to_global[u], local_to_global[v]
                        return [(gu, gv), subset, q3_mask] # Return edge to remove
                        # Actually, we just need the edge.
                    m &= m - 1
        return None

    # Precompute edge_tuples for K8
    edge_tuples = []
    for u in range(8):
        for v in range(u+1, 8):
            edge_tuples.append((u, v))
            
    # Note: edge_index and edge_tuples must be consistent.
    # edge_index[(u,v)] = idx where edge_tuples[idx] = (u,v)
    
    # Initial Graph: K16
    all_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            all_edges.add((u, v))
            
    best_edges = set()
    best_count = -1
    
    while time.time() < deadline:
        # Start with a dense random graph
        # Or start with K16 and remove edges
        current_edges = set(all_edges)
        
        # Repair Phase
        while time.time() < deadline:
            q3_info = find_q3(current_edges)
            if q3_info is None:
                break
            edge_to_remove = q3_info[0]
            current_edges.remove(edge_to_remove)
        
        if time.time() >= deadline:
            break
            
        # Optimize Phase: Add edges back
        missing_edges = all_edges - current_edges
        for u, v in missing_edges:
            if time.time() >= deadline:
                break
            # Try adding (u, v)
            current_edges.add((u, v))
            if find_q3(current_edges) is None:
                # Keep it
                pass
            else:
                # Remove it
                current_edges.remove((u, v))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
        # If we found a very good solution, we might stop early?
        # Max possible is 120. 39 is seed.
        # If best_count > 50, we are doing well.
        
    # Convert best_edges to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
