import numpy as np
import itertools
import time
import random

# Precomputation
N = 16
NUM_SUBSETS = 0
subset_vertices = []
subset_edge_indices = [] # List of lists, each list has 28 edge indices (0-119)
all_q3_masks = [] # List of arrays, each array is 168 uint32 masks

# Map (u, v) to edge index
edge_index_map = {}
edge_list = []
for i in range(N):
    for j in range(i+1, N):
        edge_index_map[(i, j)] = len(edge_list)
        edge_list.append((i, j))
TOTAL_EDGES = len(edge_list) # 120

# Generate Q3 masks for a generic set of 8 vertices labeled 0..7
# Q3 vertices are 0..7. Edges are (i, i^1), (i, i^2), (i, i^4) for i in 0..7?
# Standard Q3: vertices 0..7. Edge if xor has 1 bit.
# We need all 168 automorphisms/isomorphisms to the specific vertex set.
# Actually, for a fixed set of 8 vertices, the 168 masks correspond to the 168 ways to map the abstract Q3 to these 8 vertices.
# The abstract Q3 has 8 vertices.
# We can generate the 168 masks by permuting the 8 vertices of the abstract Q3?
# No, the abstract Q3 is fixed. The "mask" is a set of edges in the induced subgraph.
# The induced subgraph has 28 edges.
# A Q3 subgraph is a set of 12 edges.
# We need to find all subsets of 12 edges from the 28 that form a Q3.
# There are 168 such subsets.

def get_q3_masks_for_subset(vertices):
    """
    vertices: list of 8 vertex ids from 0..15
    Returns a numpy array of 168 uint32 masks (28 bits)
    """
    # Map local index 0..7 to global vertex
    # Map local edge (i, j) to local edge index 0..27
    # Local edge index: 
    # We define a fixed ordering of the 28 edges for the 8 vertices.
    # Let's use a standard ordering: (0,1), (0,2), ..., (0,7), (1,2), ..., (6,7)
    local_edge_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_map[(i, j)] = idx
            idx += 1
            
    # Generate all 168 isomorphisms of Q3
    # Q3 vertices 0..7.
    # We can fix vertex 0 to local vertex 0 to reduce symmetry?
    # No, we need all 168 masks in the *induced* subgraph.
    # The induced subgraph is on the specific 8 vertices.
    # The 168 masks are the sets of 12 edges that form a Q3.
    
    # Abstract Q3 edges
    q3_edges_abs = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_abs.append((i, j))
                
    # We need to find all permutations of the 8 local vertices such that
    # the image of q3_edges_abs forms a valid set of edges in the local graph.
    # Since the local graph is complete K8, any permutation works.
    # The set of edges in the local graph is determined by the permutation.
    
    masks = set()
    # There are 8! = 40320 permutations.
    # But Q3 has symmetries. 40320 / 24 (automorphisms of Q3?) 
    # Actually, we just need the 168 distinct edge sets.
    # We can iterate all permutations of 0..7.
    for perm in itertools.permutations(range(8)):
        # perm[i] is the local vertex corresponding to abstract vertex i
        # Check if this permutation generates a unique edge set
        edge_set = set()
        for u, v in q3_edges_abs:
            lu = perm[u]
            lv = perm[v]
            if lu > lv:
                lu, lv = lv, lu
            edge_set.add((lu, lv))
        # Convert to mask
        mask = 0
        for u, v in edge_set:
            bit = local_edge_map[(u, v)]
            mask |= (1 << bit)
        masks.add(mask)
        
    # Should be 168
    return np.array(list(masks), dtype=np.uint32)

# Precompute for all subsets
print("Precomputing subsets and masks...")
start_pre = time.time()
for verts in itertools.combinations(range(N), 8):
    subset_vertices.append(verts)
    
    # Map global edge indices for this subset
    # We need to map the 28 local edges to global edge indices
    local_to_global_edge = []
    for i in range(8):
        for j in range(i+1, 8):
            gu = verts[i]
            gv = verts[j]
            if gu > gv:
                gu, gv = gv, gu
            local_to_global_edge.append(edge_index_map[(gu, gv)])
    subset_edge_indices.append(local_to_global_edge)
    
    # Get masks
    m = get_q3_masks_for_subset(verts)
    all_q3_masks.append(m)

# Convert all_q3_masks to a single 3D array for NumPy efficiency
# Shape: (12870, 168)
all_q3_masks_np = np.array(all_q3_masks, dtype=np.uint32)

# Precompute edge_to_subsets
# For each edge (0..119), which subsets contain it?
# And which bit in the local 28-bit mask does it correspond to?
edge_to_subsets = [[] for _ in range(TOTAL_EDGES)]
for s_idx, verts in enumerate(subset_vertices):
    # For each of the 28 local edges, find the global edge
    for l_idx, g_idx in enumerate(subset_edge_indices[s_idx]):
        # g_idx is the global edge index
        # l_idx is the bit position in the local mask
        edge_to_subsets[g_idx].append((s_idx, l_idx))

print(f"Precomputation done in {time.time() - start_pre:.2f}s")

def construct():
    start_time = time.time()
    time_limit = 115
    
    # State
    edge_bits = np.random.randint(0, 2, TOTAL_EDGES, dtype=np.uint8)
    # current_states: 12870 uint32
    current_states = np.zeros(NUM_SUBSETS, dtype=np.uint32)
    
    # Initialize current_states
    # For each subset, build the 28-bit mask from edge_bits
    # This is slow if done naively.
    # We can do it using the precomputed indices.
    for s_idx in range(NUM_SUBSETS):
        mask = 0
        for l_idx, g_idx in enumerate(subset_edge_indices[s_idx]):
            if edge_bits[g_idx]:
                mask |= (1 << l_idx)
        current_states[s_idx] = mask
        
    # Initialize valid flags and cost
    # Check all subsets
    # We can do this in batches or just loop
    subset_valid = np.zeros(NUM_SUBSETS, dtype=bool)
    for s_idx in range(NUM_SUBSETS):
        state = current_states[s_idx]
        masks = all_q3_masks_np[s_idx]
        # Check if any mask is subset of state
        # (state & mask) == mask
        # Vectorized:
        valid = np.any(np.all((state[:, None] & masks) == masks, axis=1))
        subset_valid[s_idx] = valid
        
    cost = np.sum(subset_valid)
    edges_count = np.sum(edge_bits)
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    alpha = 0.05 # Penalty for edges? No, reward.
    # Objective: Cost - alpha * Edges
    # We want to minimize this.
    # Delta O = Delta Cost - alpha * Delta Edges
    
    best_cost = cost
    best_edges = edges_count
    best_edge_bits = edge_bits.copy()
    
    # SA Loop
    num_moves = 0
    while time.time() - start_time < time_limit - 5:
        # Pick random edge
        k = random.randint(0, TOTAL_EDGES - 1)
        
        # Get affected subsets
        aff_subs = edge_to_subsets[k]
        # aff_subs is list of (s_idx, l_idx)
        
        # Update current_states for these subsets
        # We need to store old states to revert if needed
        old_states = current_states[[s for s, _ in aff_subs]].copy()
        
        if edge_bits[k] == 1:
            # Removing edge
            for s_idx, l_idx in aff_subs:
                current_states[s_idx] &= ~(1 << l_idx)
            delta_edges = -1
        else:
            # Adding edge
            for s_idx, l_idx in aff_subs:
                current_states[s_idx] |= (1 << l_idx)
            delta_edges = 1
            
        # Compute new valid flags for affected subsets
        # Extract states
        new_states = current_states[[s for s, _ in aff_subs]]
        # Extract masks
        new_masks = all_q3_masks_np[[s for s, _ in aff_subs]]
        
        # Check validity
        # Shape: (3003, 168)
        # (new_states[:, None] & new_masks) == new_masks
        # This creates a 3003x168 array.
        # np.all(..., axis=1) -> 3003 bools
        # np.any(..., axis=1) -> 3003 bools
        
        # To save memory/time, we can do it in chunks if needed, but 3003x168 is small.
        new_valid = np.any(np.all((new_states[:, None] & new_masks) == new_masks, axis=1), axis=1)
        
        old_valid = subset_valid[[s for s, _ in aff_subs]]
        
        # Calculate delta cost
        # Cost is sum of valid
        delta_cost = np.sum(new_valid) - np.sum(old_valid)
        
        # Calculate delta objective
        delta_O = delta_cost - alpha * delta_edges
        
        # Acceptance
        accept = False
        if delta_O <= 0:
            accept = True
        else:
            # Boltzmann factor
            # exp(-delta_O / T)
            # If delta_O is large, prob is 0
            if T > 0:
                prob = np.exp(-delta_O / T)
                if random.random() < prob:
                    accept = True
                    
        if accept:
            edge_bits[k] = 1 - edge_bits[k]
            subset_valid[[s for s, _ in aff_subs]] = new_valid
            cost += delta_cost
            edges_count += delta_edges
            
            if cost < best_cost or (cost == best_cost and edges_count > best_edges):
                best_cost = cost
                best_edges = edges_count
                best_edge_bits = edge_bits.copy()
                
        else:
            # Revert
            current_states[[s for s, _ in aff_subs]] = old_states
            
        # Cool down
        T *= 0.999
        num_moves += 1
        
        if num_moves % 1000 == 0:
            # Optional: restart if stuck?
            pass
            
    # Final check and cleanup
    # Ensure we return the best found
    final_edge_bits = best_edge_bits
    
    edges = []
    for i in range(TOTAL_EDGES):
        if final_edge_bits[i]:
            u, v = edge_list[i]
            edges.append([u, v])
            
    return edges
