"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0

    # Robustly parse DAG
    parent_map = {}
    vertices = []
    
    if isinstance(dag, dict):
        for v, info in dag.items():
            if isinstance(info, (list, tuple)):
                parent_map[v] = list(info)
            elif isinstance(info, dict):
                parent_map[v] = list(info.get('parents', []))
            else:
                parent_map[v] = []
            vertices.append(v)
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                parents = item[1] if isinstance(item[1], (list, tuple)) else []
                parent_map[v] = list(parents)
                vertices.append(v)
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('id', None))
                if v is not None:
                    parents = item.get('parents', [])
                    parent_map[v] = list(parents)
                    vertices.append(v)
    
    if not vertices:
        return []
    
    # Build children map
    children_map = defaultdict(list)
    for v, parents in parent_map.items():
        for p in parents:
            children_map[p].append(v)
    
    # Identify inputs and outputs
    inputs = [v for v in vertices if not parent_map.get(v, [])]
    outputs = [v for v in vertices if not children_map.get(v, [])]
    
    # Topological sort
    in_degree = {v: len(parent_map.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for child in children_map.get(v, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    if len(topo_order) != len(vertices):
        topo_order = vertices
    
    # Compute levels
    level = {}
    for v in topo_order:
        parents = parent_map.get(v, [])
        level[v] = 0 if not parents else max(level[p] for p in parents) + 1
    
    max_level = max(level.values()) if level else 0
    levels = defaultdict(list)
    for v in topo_order:
        levels[level[v]].append(v)
    
    operations = []
    in_fast = set()
    
    # Load all inputs
    for v in inputs:
        if time.time() - start_time > DEADLINE:
            break
        operations.append(['load', v])
        in_fast.add(v)
    
    if time.time() - start_time > DEADLINE:
        return operations
    
    # Process layers
    for lvl in range(1, max_level + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        current = levels[lvl]
        
        # Check memory
        if len(in_fast) + len(current) > fast_memory_budget:
            # Need to free space
            uncomputed = set(v for l in range(lvl, max_level + 1) for v in levels[l])
            needed = set()
            for u in uncomputed:
                for p in parent_map.get(u, []):
                    needed.add(p)
            
            to_store = [v for v in in_fast if v not in needed and v not in uncomputed]
            to_store.sort(key=lambda x: level[x])
            for v in to_store:
                if len(in_fast) + len(current) <= fast_memory_budget:
                    break
                operations.append(['store', v])
                in_fast.discard(v)
        
        for v in current:
            if time.time() - start_time > DEADLINE:
                break
            for p in parent_map.get(v, []):
                if p not in in_fast:
                    operations.append(['load', p])
                    in_fast.add(p)
            operations.append(['compute', v])
            in_fast.add(v)
    
    if time.time() - start_time > DEADLINE:
        return operations
    
    # Store outputs
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if v in in_fast:
            operations.append(['store', v])
            in_fast.discard(v)
        else:
            operations.append(['store', v])
    
    return operations
# EVOLVE-BLOCK-END
