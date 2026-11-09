"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50
    
    # Extract DAG structure
    if isinstance(dag, dict):
        if 'parents' in dag:
            parents = dag['parents']
        elif 'edges' in dag:
            parent_map = defaultdict(list)
            for u, v in dag['edges']:
                parent_map[v].append(u)
            parents = parent_map
        else:
            parents = dag.get('parents', {})
        
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        vertices = dag.get('vertices', list(parents.keys()) if isinstance(parents, dict) else [])
    else:
        parents = getattr(dag, 'parents', None)
        outputs = getattr(dag, 'outputs', None)
        inputs = getattr(dag, 'inputs', None)
        vertices = getattr(dag, 'vertices', None)
        
        if parents is None:
            return []
    
    if not isinstance(parents, dict):
        if isinstance(parents, list):
            n = len(parents)
            parents = {i: parents[i] for i in range(n)}
        else:
            parents = {}
    
    if outputs is None:
        outputs = []
    if inputs is None:
        inputs = []
    if vertices is None:
        vertices = list(parents.keys())
    
    # Build children map
    children_map = defaultdict(list)
    for v, par_list in parents.items():
        for p in par_list:
            children_map[p].append(v)
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for child in children_map.get(u, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    if len(topo_order) != len(vertices):
        return []
    
    input_set = set(inputs) if inputs else set(v for v in vertices if len(parents.get(v, [])) == 0)
    output_set = set(outputs) if outputs else set()
    
    vertex_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Compute latest need for each vertex
    latest_need = {}
    for v in vertices:
        if v in output_set:
            latest_need[v] = len(topo_order)
        else:
            needs = [vertex_pos[c] for c in children_map.get(v, []) if c in vertex_pos]
            latest_need[v] = max(needs) if needs else vertex_pos.get(v, -1)
    
    # Simulate pebbling with greedy memory management
    fast_mem = set()
    ops = []
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # Load parents if needed
        for p in parents.get(v, []):
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Manage memory: free slots if over budget
        while len(fast_mem) > fast_memory_budget:
            # Find vertex to evict: prefer those with earliest latest_need
            candidates = []
            for x in fast_mem:
                if x in output_set:
                    continue
                candidates.append((latest_need.get(x, -1), x))
            
            if not candidates:
                break
            
            candidates.sort()
            _, to_remove = candidates[0]
            
            # Store if it will be needed again, else delete
            if latest_need.get(to_remove, -1) > vertex_pos.get(v, -1):
                ops.append(['store', to_remove])
            else:
                ops.append(['delete', to_remove])
            
            fast_mem.discard(to_remove)
    
    # Ensure all outputs are blue
    for o in output_set:
        if o in fast_mem:
            ops.append(['store', o])
            fast_mem.discard(o)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
