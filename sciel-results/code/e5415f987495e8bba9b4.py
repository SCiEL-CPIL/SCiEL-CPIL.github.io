import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    # Extract DAG info
    if isinstance(dag, dict):
        parents_dict = dict(dag.get('parents', {}))
        inputs = list(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        parents_dict = dict(getattr(dag, 'parents', {}))
        inputs = list(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))
        
    # Compute topological order
    all_vertices = set(inputs)
    for v, ps in parents_dict.items():
        all_vertices.add(v)
        all_vertices.update(ps)
        
    in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
    children = {v: [] for v in all_vertices}
    for v, ps in parents_dict.items():
        for p in ps:
            children[p].append(v)
            
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if not topo_order:
        return []
        
    # Compute next_use
    next_use = {}
    vertex_use_positions = {v: [] for v in all_vertices}
    for i, v in enumerate(topo_order):
        ps = parents_dict.get(v, [])
        for p in ps:
            vertex_use_positions[p].append(i)
            
    for v in all_vertices:
        if vertex_use_positions[v]:
            next_use[v] = vertex_use_positions[v][0]
        else:
            next_use[v] = float('inf')
            
    # Greedy Belady
    in_fast = set()
    ops = []
    
    for i, v in enumerate(topo_order):
        if v in inputs:
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    best_evict = None
                    max_nu = -1
                    for u in in_fast:
                        nu = next_use.get(u, float('inf'))
                        if nu > max_nu:
                            max_nu = nu
                            best_evict = u
                    if best_evict is not None:
                        in_fast.discard(best_evict)
                        if best_evict in outputs:
                            ops.append(['store', best_evict])
                        else:
                            ops.append(['delete', best_evict])
                in_fast.add(v)
                ops.append(['load', v])
        else:
            ps = parents_dict.get(v, [])
            for p in ps:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        best_evict = None
                        max_nu = -1
                        for u in in_fast:
                            if u != p:
                                nu = next_use.get(u, float('inf'))
                                if nu > max_nu:
                                    max_nu = nu
                                    best_evict = u
                        if best_evict is not None:
                            in_fast.discard(best_evict)
                            if best_evict in outputs:
                                ops.append(['store', best_evict])
                            else:
                                ops.append(['delete', best_evict])
                    in_fast.add(p)
                    ops.append(['load', p])
                    
            in_fast.add(v)
            ops.append(['compute', v])
            
        # Evict if no longer needed
        # last_use is the last index where v is used as a parent.
        # If i > last_use[v], v is no longer needed.
        lu = vertex_use_positions.get(v, [-1])[-1] if vertex_use_positions.get(v) else -1
        if i > lu and v not in outputs:
            if v in in_fast:
                in_fast.discard(v)
                ops.append(['delete', v])
        elif v in outputs:
            if v in in_fast:
                in_fast.discard(v)
                ops.append(['store', v])
                
    # Final cleanup
    for v in in_fast:
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops
