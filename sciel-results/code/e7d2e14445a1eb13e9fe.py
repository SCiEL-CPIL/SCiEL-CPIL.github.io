import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # Parse DAG structure
    # Assumed structure: 
    # dag['vertices']: list of vertex ids
    # dag['parents']: dict or list of lists, parents[v] = list of parent ids
    # dag['inputs']: list of input vertex ids
    # dag['outputs']: list of output vertex ids
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # If parents is a list, convert to dict
    if isinstance(parents, list):
        parents_dict = {v: parents[i] for i, v in enumerate(vertices)}
    else:
        parents_dict = parents

    # Identify relevant subgraph: vertices that can reach an output
    # Reverse graph
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_dict.get(v, []):
            children[p].append(v)
            
    # BFS/DFS from outputs to find ancestors
    relevant = set(outputs)
    stack = list(outputs)
    while stack:
        v = stack.pop()
        for p in parents_dict.get(v, []):
            if p not in relevant:
                relevant.add(p)
                stack.append(p)
                
    # Topological sort of relevant subgraph
    # Only consider vertices in 'relevant'
    in_degree = {v: 0 for v in relevant}
    for v in relevant:
        for p in parents_dict.get(v, []):
            if p in relevant:
                in_degree[v] += 1
                
    queue = [v for v in relevant if in_degree[v] == 0]
    topo_order = []
    while queue:
        # Sort queue to have a deterministic order (optional, but good for consistency)
        queue.sort()
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            if c in relevant:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    
    # If some vertices are not in topo_order, they might be in cycles or not reachable, but we filtered by relevant.
    # Ensure all relevant are in topo_order
    if len(topo_order) != len(relevant):
        # Fallback: add missing
        for v in relevant:
            if v not in topo_order:
                topo_order.append(v)

    # Calculate last_use for each vertex
    # last_use[v] is the index in topo_order of the last vertex that uses v as a parent or is v itself (if output)
    last_use = {v: -1 for v in relevant}
    for i, v in enumerate(topo_order):
        last_use[v] = i
        for p in parents_dict.get(v, []):
            if p in relevant:
                last_use[p] = i

    # Pebbling Algorithm
    in_mem = set()
    ops = []
    
    def evict_one():
        if not in_mem:
            return
        # Find vertex in in_mem with max last_use
        max_last = -1
        victim = None
        for v in in_mem:
            if last_use[v] > max_last:
                max_last = last_use[v]
                victim = v
        if victim is not None:
            if victim in inputs:
                ops.append(['delete', victim])
            else:
                ops.append(['store', victim])
            in_mem.remove(victim)

    def ensure_in_memory(v):
        if v in in_mem:
            return
        if v in inputs:
            if len(in_mem) >= fast_memory_budget:
                evict_one()
            ops.append(['load', v])
            in_mem.add(v)
        else:
            # Intermediate
            for p in parents_dict.get(v, []):
                if p in relevant:
                    ensure_in_memory(p)
            if len(in_mem) >= fast_memory_budget:
                evict_one()
            ops.append(['compute', v])
            in_mem.add(v)

    # Process in topological order
    for v in topo_order:
        ensure_in_memory(v)
        
        # Check deadline
        if time.time() > deadline:
            break

    # Ensure outputs are in memory at the end
    for v in outputs:
        ensure_in_memory(v)
        
        if time.time() > deadline:
            break

    return ops
