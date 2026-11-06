import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 masks in K8
    q3_edges_local = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
                
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
            
    q3_masks = set()
    # Generate all unique Q3 masks
    # We can use permutations of 0..7
    # To save time, we can just generate a representative set? 
    # 40320 permutations is fine for startup.
    for p in range(40320): # Using range for speed? No, need permutations.
        pass
    
    # Better: generate permutations
    from itertools import permutations
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # Precompute Q3 masks grouped by local edge bit
    # q3_by_bit[b] = list of q3 masks that have bit b set
    q3_by_bit = [[] for _ in range(28)]
    for qm in q3_masks_list:
        for b in range(28):
            if (qm >> b) & 1:
                q3_by_bit[b].append(qm)

    # Precompute subset mappings
    subsets = list(combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # subset_edges[s] = list of (global_edge_idx, local_bit)
    subset_edges = [None] * num_subsets
    # edge_subsets[g] = list of (subset_idx, local_bit)
    edge_subsets = [[] for _ in range(num_edges)]
    
    for s_idx, subset in enumerate(subsets):
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                mapping.append((ge_idx, le_bit))
                edge_subsets[ge_idx].append((s_idx, le_bit))
        subset_edges[s_idx] = mapping

    # State
    # active_edges: set of global edge indices
    # subset_masks: list of ints, mask of active local edges for each subset
    # best_mask: int bitmask of global edges
    # best_count: int
    
    random.seed(42)
    
    best_count = 0
    best_global_mask = 0
    
    # Initialize with a random dense graph and repair
    # We'll run multiple trials
    
    for trial in range(100):
        if time.time() > deadline - 5:
            break
            
        # 1. Initialize
        current_global_mask = 0
        current_count = 0
        subset_masks = [0] * num_subsets
        
        # Start with random p=0.8
        for g in range(num_edges):
            if random.random() < 0.8:
                current_global_mask |= (1 << g)
                current_count += 1
                for s_idx, le_bit in edge_subsets[g]:
                    subset_masks[s_idx] |= (1 << le_bit)
                    
        # Repair: Remove edges that form Q3
        # We need to find any Q3 and remove an edge.
        # To do this efficiently, we can scan subsets.
        # If a subset mask contains a Q3 mask, we have a violation.
        # We remove a random edge from that Q3.
        
        repair_limit = 500
        for _ in range(repair_limit):
            violated = False
            # Find a violation
            # We can iterate subsets and check
            # To speed up, we can just pick a random subset that has high density?
            # Or just iterate all. 12870 checks is fast.
            viol_subset_idx = -1
            viol_qm = -1
            for s_idx in range(num_subsets):
                m = subset_masks[s_idx]
                # Check if m contains any Q3
                # We can check against q3_masks_list
                # Optimization: only check if m has at least 12 bits?
                if bin(m).count('1') < 12:
                    continue
                for qm in q3_masks_list:
                    if (m & qm) == qm:
                        viol_subset_idx = s_idx
                        viol_qm = qm
                        violated = True
                        break
                if violated:
                    break
            
            if not violated:
                break
                
            # Remove a random edge from the violating Q3
            # Get the global edges corresponding to the bits in viol_qm for this subset
            mapping = subset_edges[viol_subset_idx]
            # Find which global edges are set in viol_qm
            candidate_removes = []
            for ge_idx, le_bit in mapping:
                if (viol_qm >> le_bit) & 1:
                    # Check if this edge is currently active
                    if (current_global_mask >> ge_idx) & 1:
                        candidate_removes.append(ge_idx)
            
            if not candidate_removes:
                # Should not happen
                break
                
            rem_g = random.choice(candidate_removes)
            # Remove edge rem_g
            current_global_mask &= ~(1 << rem_g)
            current_count -= 1
            for s_idx, le_bit in edge_subsets[rem_g]:
                subset_masks[s_idx] &= ~(1 << le_bit)
                
        # Check if valid
        if any((subset_masks[s] & qm) == qm for s in range(num_subsets) for qm in q3_masks_list if bin(subset_masks[s]).count('1')>=12):
            continue # Still violated, skip this trial

        if current_count > best_count:
            best_count = current_count
            best_global_mask = current_global_mask

        # 2. Tabu Search
        # Tabu list: list of (global_edge, direction)
        # direction: 1 for add, -1 for remove
        # We forbid reversing the last T moves.
        T = 50
        tabu_list = [] # (g, dir, expiry_trial)
        current_trial = 0
        
        # We will perform a fixed number of steps or until time runs out
        max_steps = 1000
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            current_trial += 1
            
            # Generate candidate moves
            # Add moves: edges not in current_global_mask
            # Remove moves: edges in current_global_mask
            
            # To be efficient, we prefer Add moves.
            # If no Add move is feasible (all create Q3 or are tabu), we do a Remove move.
            
            # 1. Try Add moves
            # We can pick a random set of candidate adds
            # Or iterate through missing edges?
            # Number of missing edges is large.
            # Let's pick a random sample of missing edges to try.
            
            # Get list of missing edges
            # This is expensive to build every time.
            # We can maintain a list?
            # Or just pick random g and check if it's missing.
            
            found_add = False
            # Try 10 random add candidates
            for _ in range(10):
                g = random.randint(0, num_edges - 1)
                if (current_global_mask >> g) & 1:
                    continue # Already present
                
                # Check tabu
                is_tabu = False
                for tg, td, te in tabu_list:
                    if tg == g and td == -1: # If we recently removed g, adding it is tabu
                        if current_trial < te:
                            is_tabu = True
                            break
                if is_tabu:
                    continue
                
                # Check feasibility
                # Update subset masks virtually
                # We need to check if adding g creates a Q3
                # g affects subsets in edge_subsets[g]
                creates_q3 = False
                for s_idx, le_bit in edge_subsets[g]:
                    new_mask = subset_masks[s_idx] | (1 << le_bit)
                    # Check if new_mask contains a Q3
                    # Only check Q3s that have le_bit set
                    for qm in q3_by_bit[le_bit]:
                        if (new_mask & qm) == qm:
                            creates_q3 = True
                            break
                    if creates_q3:
                        break
                
                if not creates_q3:
                    # Perform add
                    current_global_mask |= (1 << g)
                    current_count += 1
                    for s_idx, le_bit in edge_subsets[g]:
                        subset_masks[s_idx] |= (1 << le_bit)
                    
                    # Update tabu
                    tabu_list.append((g, 1, current_trial + T))
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_global_mask = current_global_mask
                        
                    found_add = True
                    break
            
            if found_add:
                continue
                
            # 2. Try Remove moves
            # If no add found, we must remove an edge to escape local optimum
            # Pick a random active edge
            # To make it smarter, pick an edge involved in many "near" Q3s?
            # For now, random.
            
            found_remove = False
            # Try 10 random remove candidates
            for _ in range(10):
                g = random.randint(0, num_edges - 1)
                if not ((current_global_mask >> g) & 1):
                    continue # Not present
                
                # Check tabu
                is_tabu = False
                for tg, td, te in tabu_list:
                    if tg == g and td == 1: # If we recently added g, removing it is tabu
                        if current_trial < te:
                            is_tabu = True
                            break
                if is_tabu:
                    continue
                
                # Perform remove
                current_global_mask &= ~(1 << g)
                current_count -= 1
                for s_idx, le_bit in edge_subsets[g]:
                    subset_masks[s_idx] &= ~(1 << le_bit)
                
                # Update tabu
                tabu_list.append((g, -1, current_trial + T))
                
                found_remove = True
                break
                
            if not found_remove:
                # All moves tabu? Clear tabu list or reduce T
                # Just break or restart
                break
                
        # End Tabu Search for this trial

    # Convert best_global_mask to list of edges
    result = []
    for i in range(num_edges):
        if (best_global_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
