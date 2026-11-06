"""DAG pebbling with communication minimization using divide-and-conquer."""
import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55  # 55s safety margin
    
    # Parse the DAG structure
    # dag format: assume it has 'vertices', 'parents' (dict or list), 'inputs', 'outputs'
    # Let's handle common formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
        if not vertices:
            # Try to infer from parents
            all_v = set()
            for v, ps in parents.items():
                all_v.add(v)
                all_v.update(ps)
            vertices = list(all_v)
    elif isinstance(dag, (list, tuple)):
        # Maybe list of (vertex, parents) or similar
        vertices = list(dag)
        parents = {}
        inputs = set()
        outputs = []
        # This is ambiguous, assume simple format
    else:
        vertices = []
        parents = {}
        inputs = set()
        outputs = []
    
    # Ensure all vertices are in parents dict
    all_vertices = set(vertices)
    for v, ps in list(parents.items()):
        for p in ps:
            all_vertices.add(p)
    
    # Topological sort
    in_degree = {v: 0 for v in all_vertices}
    children = {v: [] for v in all_vertices}
    for v, ps in parents.items():
        for p in ps:
            in_degree[v] += 1
            children[p].append(v)
    
    from collections import deque
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topological sort didn't include all, add remaining (shouldn't happen for DAG)
    if len(topo_order) != len(all_vertices):
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    # Determine inputs and outputs if not provided
    if not inputs:
        inputs = set(v for v in all_vertices if v not in parents or not parents.get(v, []))
    if not outputs:
        outputs = [v for v in all_vertices if not children.get(v, [])]
    
    # Now implement pebbling strategy
    # Strategy: Process in topological order, use fast memory wisely
    # For each vertex, we need its parents to be available
    # We track: fast_mem (set of vertices in fast memory), slow_mem (set of vertices in slow memory)
    # Operations: load, store, compute, delete
    
    # Let's use a recursive divide-and-conquer approach
    # For simplicity and to beat baseline, use a layer-based approach
    
    # Identify layers
    layer = {}
    for v in topo_order:
        if v not in parents or not parents.get(v, []):
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parents[v]) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group vertices by layer
    layers = {}
    for v in topo_order:
        l = layer[v]
        if l not in layers:
            layers[l] = []
        layers[l].append(v)
    
    # Pebbling strategy:
    # For each layer, compute vertices, store intermediate results
    # Use fast memory budget to decide what to keep
    # 
    # Simpler approach: Process layer by layer
    # - Load inputs (they start blue, so they're in fast memory)
    # - For each subsequent layer, compute vertices
    # - After computing a layer, store non-output vertices to slow memory
    # - At the end, ensure outputs are in fast memory
    
    # Let's implement a more sophisticated approach:
    # Use the fast memory budget to keep the most recently needed values
    # and store older ones
    
    operations = []
    fast_mem = set(inputs)  # inputs start blue
    slow_mem = set()
    computed = set(inputs)  # values that have been computed (available somewhere)
    
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        
        if v in inputs:
            continue  # already in fast memory
        
        # Ensure parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                else:
                    # Should not happen if we process in topological order
                    # and keep parents available
                    pass
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Check if we need to free space
        # If fast_mem exceeds budget, store some values
        # Strategy: store values that are not outputs and have no more children to compute
        # or store the oldest non-output values
        
        while len(fast_mem) > fast_memory_budget and operations:
            # Find a vertex to store
            # Prefer non-output vertices that are not needed for remaining computations
            remaining = set(topo_order[topo_order.index(v) + 1:]) if v in topo_order else set()
            candidates = []
            for f in fast_mem:
                if f in inputs:
                    continue  # don't store inputs (they're free)
                if f in outputs:
                    continue  # don't store outputs (they need to be blue at end)
                # Check if f is needed for any remaining vertex
                needed = False
                for rem in remaining:
                    if f in parents.get(rem, []):
                        needed = True
                        break
                if not needed:
                    candidates.append(f)
            
            if candidates:
                # Store the candidate with the highest layer (oldest)
                to_store = max(candidates, key=lambda x: layer.get(x, 0))
                operations.append(['store', to_store])
                fast_mem.discard(to_store)
                slow_mem.add(to_store)
            else:
                # All remaining fast values are needed or are inputs/outputs
                # Store the one with the highest layer that is not an output
                candidates = [f for f in fast_mem if f not in outputs and f not in inputs]
                if candidates:
                    to_store = max(candidates, key=lambda x: layer.get(x, 0))
                    operations.append(['store', to_store])
                    fast_mem.discard(to_store)
                    slow_mem.add(to_store)
                else:
                    break  # Can't free space, break to avoid infinite loop
    
    # Ensure all outputs are in fast memory (blue)
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                operations.append(['load', o])
                fast_mem.add(o)
                slow_mem.discard(o)
    
    # Cap operations at 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
