import inspect, sys, time
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Solves the DAG pebbling problem.
    """
    # Helper to get attribute or key
    def get_attr(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
        
    inputs = get_attr(dag, 'inputs', [])
    outputs = get_attr(dag, 'outputs', [])
    parents_dict = get_attr(dag, 'parents', {})
    vertices_list = get_attr(dag, 'vertices', [])
    
    if not isinstance(inputs, list):
        inputs = list(inputs)
    if not isinstance(outputs, list):
        outputs = list(outputs)
    if not isinstance(vertices_list, list):
        vertices_list = list(vertices_list)
        
    all_vertices = set()
    all_vertices.update(inputs)
    all_vertices.update(outputs)
    all_vertices.update(vertices_list)
    
    if isinstance(parents_dict, dict):
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            if isinstance(ps, list):
                all_vertices.update(ps)
            else:
                all_vertices.add(ps)
    else:
        # Fallback: assume parents_dict is a list of (v, [ps]) or similar?
        # If not dict, we might not have parent info. 
        # If no parent info, we can't compute topo order properly.
        # Assume vertices_list is topo sorted?
        pass
        
    # Build children graph
    children = {v: [] for v in all_vertices}
    in_degree = {v: 0 for v in all_vertices}
    
    if isinstance(parents_dict, dict):
        for v, ps in parents_dict.items():
            if not isinstance(ps, list):
                ps = [ps]
            in_degree[v] = len(ps)
            for p in ps:
                if p in children:
                    children[p].append(v)
                else:
                    # p might not be in all_vertices if not listed, but we added it above
                    if p not in children:
                        children[p] = []
                    children[p].append(v)
                        
    # Topological Sort
    queue = deque()
    for v in inputs:
        if v in in_degree:
            in_degree[v] = 0
        else:
            in_degree[v] = 0
        queue.append(v)
        
    for v in all_vertices:
        if v not in inputs and in_degree[v] == 0:
            queue.append(v)
            
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) < len(all_vertices):
        # Add missing
        missing = set(all_vertices) - set(topo_order)
        topo_order.extend(list(missing))
        
    # Last use calculation
    child_indices = {v: i for i, v in enumerate(topo_order)}
    last_use = {}
    for v in all_vertices:
        max_idx = child_indices.get(v, 0)
        for c in children.get(v, []):
            if c in child_indices:
                max_idx = max(max_idx, child_indices[c])
        last_use[v] = max_idx
        
    for o in outputs:
        if o in last_use:
            last_use[o] = len(topo_order) - 1
            
    operations = []
    blue_set = set(inputs)
    
    for i, v in enumerate(topo_order):
        if v in inputs:
            pass
        else:
            ps = parents_dict.get(v, []) if isinstance(parents_dict, dict) else []
            if not isinstance(ps, list):
                ps = [ps]
                
            ready = True
            for p in ps:
                if p not in blue_set:
                    ready = False
                    break
            
            if ready:
                operations.append(["compute", v])
                blue_set.add(v)
                
        # Cleanup
        for u in list(blue_set):
            if u in outputs:
                continue
            if last_use[u] < i:
                operations.append(["delete", u])
                blue_set.remove(u)
                
    return operations
