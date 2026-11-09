def pebble(dag, fast_memory_budget):
    from collections import deque
    
    parent_map = dag.get('parents', {})
    vertices = list(parent_map.keys())
    if 'vertices' in dag:
        vertices = list(dag['vertices'])
    
    children_map = {}
    for v, parents in parent_map.items():
        for p in parents:
            if p not in children_map:
                children_map[p] = []
            children_map[p].append(v)
    
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    in_degree = {v: len(parent_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) < len(vertices):
        for v in vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    red = set()
    operations = []
    computed = set()
    
    def can_delete(r):
        children = children_map.get(r, [])
        return all(c in computed for c in children)
    
    for v in topo_order:
        if v in inputs:
            if v not in red:
                operations.append(['load', v])
                red.add(v)
        else:
            parents = parent_map.get(v, [])
            for p in parents:
                if p not in red:
                    operations.append(['load', p])
                    red.add(p)
            operations.append(['compute', v])
            red.add(v)
            computed.add(v)
        
        if v in outputs:
            operations.append(['store', v])
            red.discard(v)
        
        for r in list(red):
            if can_delete(r):
                operations.append(['delete', r])
                red.discard(r)
        
        while len(red) > fast_memory_budget:
            deleted = False
            for r in list(red):
                if can_delete(r):
                    operations.append(['delete', r])
                    red.discard(r)
                    deleted = True
                    break
            if not deleted:
                for r in list(red):
                    if r not in outputs:
                        operations.append(['delete', r])
                        red.discard(r)
                        break
                else:
                    break
        
        if len(operations) >= 20000:
            break
    
    return operations
