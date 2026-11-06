import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    
    # Precompute all 8-vertex subsets
    subsets_8 = []
    subset_masks = []
    for s in itertools.combinations(range(n), 8):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subsets_8.append(s)
        subset_masks.append(mask)
    
    # Precompute all partitions of 8 positions into two sets of 4
    partitions = []
    for x_combo in itertools.combinations(range(8), 4):
        x_indices = tuple(x_combo)
        y_indices = tuple(i for i in range(8) if i not in x_indices)
        partitions.append((x_indices, y_indices))
    
    num_subsets = len(subsets_8)
    num_partitions = len(partitions)
    
    # Precompute masks and vertex lists for each (subset, partition) pair
    precomputed = [[None for _ in range(num_partitions)] for _ in range(num_subsets)]
    
    for k, s in enumerate(subsets_8):
        for j, (x_indices, y_indices) in enumerate(partitions):
            x_mask = 0
            y_mask = 0
            x_verts = []
            y_verts = []
            for i in x_indices:
                v = s[i]
                x_mask |= (1 << v)
                x_verts.append(v)
            for i in y_indices:
                v = s[i]
                y_mask |= (1 << v)
                y_verts.append(v)
            precomputed[k][j] = (x_mask, y_mask, tuple(x_verts), tuple(y_verts))
    
    def has_q3(adj_masks):
        for k in range(num_subsets):
            mask = subset_masks[k]
            s = subsets_8[k]
            
            # Quick edge count check
            edge_count = 0
            for v in s:
                edge_count += (adj_masks[v] & mask).bit_count()
            if edge_count < 24:
                continue
            
            for j in range(num_partitions):
                x_mask, y_mask, x_verts, y_verts = precomputed[k][j]
                
                valid = True
                for v in x_verts:
                    if (adj_masks[v] & y_mask).bit_count() < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for v in y_verts:
                    if (adj_masks[v] & x_mask).bit_count() < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False
    
    # Build initial graph: K_{3,13} with NO A-A edges (valid Q3-free base)
    # A = {0,1,2}, B = {3..15}
    # This is bipartite, hence Q3-free
    adj_masks = [0] * n
    current_edges = set()
    
    a_vertices = [0, 1, 2]
    b_vertices = list(range(3, 16))
    
    for a in a_vertices:
        for b in b_vertices:
            adj_masks[a] |= (1 << b)
            adj_masks[b] |= (1 << a)
            current_edges.add((a, b))
    
    # All possible edges to consider adding
    all_candidate_edges = []
    # A-A edges
    for i in range(len(a_vertices)):
        for j in range(i+1, len(a_vertices)):
            all_candidate_edges.append((a_vertices[i], a_vertices[j]))
    # B-B edges
    for i in range(len(b_vertices)):
        for j in range(i+1, len(b_vertices)):
            all_candidate_edges.append((b_vertices[i], b_vertices[j]))
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search with greedy addition and edge swaps
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Reset to base state
        adj_masks = [0] * n
        current_edges = set()
        
        for a in a_vertices:
            for b in b_vertices:
                adj_masks[a] |= (1 << b)
                adj_masks[b] |= (1 << a)
                current_edges.add((a, b))
        
        # Greedy phase: try adding edges in random order
        random.shuffle(all_candidate_edges)
        for u, v in all_candidate_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            current_edges.add((u, v))
            
            if has_q3(adj_masks):
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                current_edges.remove((u, v))
        
        # Local search phase: try edge swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(all_candidate_edges)
            for u, v in all_candidate_edges:
                if time.time() > deadline - 5:
                    break
                if (u, v) in current_edges:
                    continue
                
                # Try adding (u,v)
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                if not has_q3(adj_masks):
                    current_edges.add((u, v))
                    improved = True
                    break
                else:
                    # Revert
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
