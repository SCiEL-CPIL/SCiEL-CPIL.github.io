def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_edges = None
    best_count = float('inf')
    best_n = 9
    
    for n in [9, 10, 11]:
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        
        bad_masks = [0] * (1 << n)
        
        # Precompute masks for edges to speed up bad_masks calculation?
        # Or just do it directly.
        edge_vertex_masks = []
        for e in all_edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_vertex_masks.append(m)
            
        for c in range(1 << n):
            bm = 0
            for i, vm in enumerate(edge_vertex_masks):
                inter = c & vm
                if inter == 0 or inter == vm:
                    bm |= (1 << i)
            bad_masks[c] = bm
            
        def is_non_colorable(active_mask):
            for c in range(1 << n):
                if (bad_masks[c] & active_mask) == 0:
                    return False
            return True
            
        current_mask = (1 << num_edges) - 1
        
        if not is_non_colorable(current_mask):
            continue
            
        best_local_mask = current_mask
        best_local_count = num_edges
        
        # Random restarts
        while time.time() < deadline - 5:
            curr = current_mask
            changed = True
            while changed:
                changed = False
                indices = list(range(num_edges))
                random.shuffle(indices)
                for i in indices:
                    if curr & (1 << i):
                        new_mask = curr & ~(1 << i)
                        if is_non_colorable(new_mask):
                            curr = new_mask
                            changed = True
                            
            count = bin(curr).count('1')
            if count < best_local_count:
                best_local_count = count
                best_local_mask = curr
            if best_local_count <= 51:
                break
                
        if best_local_count < best_count:
            best_count = best_local_count
            best_n = n
            best_edges = [all_edges[i] for i in range(num_edges) if best_local_mask & (1 << i)]
            
    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
