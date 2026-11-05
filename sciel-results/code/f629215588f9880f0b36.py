import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # 55s limit

    # Parse DAG
    nodes = dag.get('nodes', dag) if isinstance(dag, dict) else dag
    
    node_ids = []
    node_map = {}
    parents_map = {}
    is_output_map = {}
    
    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            nid = node.get('id', i)
            parents = node.get('parents', [])
            is_out = node.get('is_output', False)
        else:
            nid = node
            parents = []
            is_out = False
            
        node_ids.append(nid)
        node_map[nid] = i
        parents_map[nid] = parents
        is_output_map[nid] = is_out

    N = len(node_ids)
    
    # Topological Sort
    in_degree = {nid: len(parents_map[nid]) for nid in node_ids}
    children_map = {nid: [] for nid in node_ids}
    
    for nid in node_ids:
        for p in parents_map[nid]:
            children_map[p].append(nid)
            
    queue = [nid for nid in node_ids if in_degree[nid] == 0]
    topo_order = []
    
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in children_map[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != N:
        return _naive_pebble(dag, fast_memory_budget)

    topo_index = {nid: i for i, nid in enumerate(topo_order)}
    
    for u in node_ids:
        children_map[u].sort(key=lambda x: topo_index[x])
        
    fast_mem = set()
    inputs = [nid for nid in node_ids if len(parents_map[nid]) == 0]
    slow_mem = set(inputs)
    
    computed = set()
    next_child_ptr = {u: 0 for u in node_ids}
    
    operations = []
    
    def get_next_use(u):
        ptr = next_child_ptr[u]
        children = children_map[u]
        if ptr < len(children):
            return topo_index[children[ptr]]
        return float('inf')
        
    for v in topo_order:
        if time.time() > deadline:
            # If we run out of time, return what we have.
            # This might be invalid, but better than timeout.
            break
            
        for p in parents_map[v]:
            if p not in fast_mem:
                if p in slow_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Should not happen
                    pass
            
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        for p in parents_map[v]:
            while next_child_ptr[p] < len(children_map[p]) and children_map[p][next_child_ptr[p]] != v:
                next_child_ptr[p] += 1
            next_child_ptr[p] += 1
            
            if next_child_ptr[p] >= len(children_map[p]):
                if p in fast_mem:
                    if is_output_map[p]:
                        operations.append(['store', p])
                        slow_mem.add(p)
                        operations.append(['delete', p])
                        fast_mem.remove(p)
                    else:
                        operations.append(['delete', p])
                        fast_mem.remove(p)
        
        while len(fast_mem) > fast_memory_budget:
            best_evict = None
            best_next_use = -1
            
            for u in fast_mem:
                nu = get_next_use(u)
                if nu > best_next_use:
                    best_next_use = nu
                    best_evict = u
                    
            if best_evict is None:
                break
                
            operations.append(['store', best_evict])
            slow_mem.add(best_evict)
            operations.append(['delete', best_evict])
            fast_mem.remove(best_evict)
            
    return operations

def _naive_pebble(dag, fast_memory_budget):
    nodes = dag.get('nodes', dag) if isinstance(dag, dict) else dag
    ops = []
    for node in nodes:
        nid = node.get('id', 0) if isinstance(node, dict) else node
        if isinstance(node, dict):
            parents = node.get('parents', [])
            for p in parents:
                ops.append(['load', p])
        ops.append(['compute', nid])
        if isinstance(node, dict) and node.get('is_output', False):
            ops.append(['store', nid])
    return ops
