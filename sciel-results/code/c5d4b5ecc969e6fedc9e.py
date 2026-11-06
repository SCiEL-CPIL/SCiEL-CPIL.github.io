import heapq

def pebble(dag, fast_memory_budget):
    # Parse DAG
    vertices = []
    outputs = set()
    parents_map = {}
    
    if isinstance(dag, list):
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                vid = v.get('id', i)
                pars = v.get('parents', [])
                is_out = v.get('is_output', False)
                vertices.append(vid)
                parents_map[vid] = pars
                if is_out:
                    outputs.add(vid)
    elif isinstance(dag, dict):
        for vid, v in dag.items():
            if isinstance(v, dict):
                pars = v.get('parents', [])
                is_out = v.get('is_output', False)
                vertices.append(vid)
                parents_map[vid] = pars
                if is_out:
                    outputs.add(vid)
    
    if not vertices:
        return []
        
    id_to_idx = {vid: i for i, vid in enumerate(vertices)}
    n = len(vertices)
    
    # Identify inputs (no parents)
    input_indices = set()
    for i in range(n):
        vid = vertices[i]
        if not parents_map.get(vid, []):
            input_indices.add(i)
            
    # Identify outputs
    output_indices = set()
    for vid in outputs:
        if vid in id_to_idx:
            output_indices.add(id_to_idx[vid])
            
    # Compute ancestors
    ancestors = [set() for _ in range(n)]
    for i in range(n):
        vid = vertices[i]
        pars = parents_map.get(vid, [])
        for p in pars:
            if p in id_to_idx:
                p_idx = id_to_idx[p]
                ancestors[i].add(p_idx)
                ancestors[i].update(ancestors[p_idx])

    # Heuristic
    def h(mask):
        needed = set()
        for i in output_indices:
            if not (mask >> i) & 1:
                needed.add(i)
                needed.update(ancestors[i])
        needed -= input_indices
        # Subtract those already in mask
        for i in range(n):
            if (mask >> i) & 1:
                needed.discard(i)
        return len(needed)

    start_mask = 0
    goal_mask = 0
    for i in output_indices:
        goal_mask |= (1 << i)
        
    if start_mask == goal_mask:
        return []
        
    g_score = {start_mask: 0}
    came_from = {}
    open_set = []
    heapq.heappush(open_set, (h(start_mask), 0, start_mask))
    
    while open_set:
        f, g, mask = heapq.heappop(open_set)
        
        if mask == goal_mask:
            path = []
            cur = mask
            while cur in came_from:
                prev, op, vid = came_from[cur]
                path.append((op, vid))
                cur = prev
            path.reverse()
            return [[op, vertices[vid]] for op, vid in path]

        if g > g_score.get(mask, float('inf')):
            continue
            
        # Compute
        for i in range(n):
            if i in input_indices:
                continue
            if (mask >> i) & 1:
                continue
            
            vid = vertices[i]
            pars = parents_map.get(vid, [])
            available = True
            for p in pars:
                if p in id_to_idx:
                    p_idx = id_to_idx[p]
                    if not ((mask >> p_idx) & 1):
                        if p_idx not in input_indices:
                            available = False
                            break
            if available:
                new_mask = mask | (1 << i)
                if new_mask.bit_count() <= fast_memory_budget:
                    new_g = g + 1
                    if new_g < g_score.get(new_mask, float('inf')):
                        g_score[new_mask] = new_g
                        came_from[new_mask] = (mask, 'compute', i)
                        heapq.heappush(open_set, (new_g + h(new_mask), new_g, new_mask))

        # Delete
        for i in range(n):
            if i in input_indices:
                continue
            if not ((mask >> i) & 1):
                continue
            if i in output_indices:
                continue
            new_mask = mask & ~(1 << i)
            new_g = g + 1
            if new_g < g_score.get(new_mask, float('inf')):
                g_score[new_mask] = new_g
                came_from[new_mask] = (mask, 'delete', i)
                heapq.heappush(open_set, (new_g + h(new_mask), new_g, new_mask))

    return []
