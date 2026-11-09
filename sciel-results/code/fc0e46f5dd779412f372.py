import time

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # safety margin before 60s/90s kill
    ops = []
    
    # Parse DAG structure
    # Assume dag is a dict or object with 'vertices', 'parents', 'inputs', 'outputs'
    # Try common formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    else:
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        inputs = getattr(dag, 'inputs', [])
        outputs = getattr(dag, 'outputs', [])
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build set for quick lookup
    input_set = set(inputs)
    output_set = set(outputs)
    
    # If budget is large enough to hold all vertices, no transfers needed
    if fast_memory_budget >= n:
        # Compute all non-input vertices in topological order
        for v in vertices:
            if v not in input_set:
                ops.append(['compute', v])
        return ops
    
    # Otherwise, use a pebbling strategy
    # Process in topological order, maintaining a cache of blue vertices
    # Simple strategy: keep a sliding window of size fast_memory_budget
    
    # We'll use a dictionary to track which vertices are currently blue
    blue_set = set(inputs)  # inputs start blue
    # But we can only hold fast_memory_budget vertices total
    # If inputs exceed budget, we need to store some
    
    # Strategy: process layer by layer or in topo order
    # For each vertex, ensure parents are blue, compute, manage cache
    
    # Simple approach: use a queue-based eviction
    # Keep track of order of blue vertices for eviction
    
    blue_order = list(inputs)  # order of addition
    # If too many inputs, store extras
    while len(blue_set) > fast_memory_budget:
        # Evict oldest input (store it)
        evict = blue_order.pop(0)
        if evict in blue_set:
            blue_set.discard(evict)
            ops.append(['store', evict])
    
    for v in vertices:
        if time.time() > deadline:
            break
        if v in input_set:
            continue  # already handled
        
        # Ensure parents are blue
        for p in parents.get(v, []):
            if p not in blue_set:
                # Load parent
                ops.append(['load', p])
                blue_set.add(p)
                blue_order.append(p)
                # Evict if over budget
                while len(blue_set) > fast_memory_budget:
                    evict = blue_order.pop(0)
                    if evict in blue_set:
                        blue_set.discard(evict)
                        ops.append(['store', evict])
        
        # Compute v
        ops.append(['compute', v])
        blue_set.add(v)
        blue_order.append(v)
        
        # Evict if over budget, but don't evict outputs if possible
        while len(blue_set) > fast_memory_budget:
            # Find oldest non-output to evict
            evicted = False
            for i, evict in enumerate(blue_order):
                if evict in blue_set and evict not in output_set:
                    blue_order.pop(i)
                    blue_set.discard(evict)
                    ops.append(['store', evict])
                    evicted = True
                    break
            if not evicted:
                # Must evict an output (bad, but necessary)
                evict = blue_order.pop(0)
                if evict in blue_set:
                    blue_set.discard(evict)
                    ops.append(['store', evict])
    
    # Ensure all outputs are blue at the end
    for v in outputs:
        if v not in blue_set:
            ops.append(['load', v])
            blue_set.add(v)
            blue_order.append(v)
            while len(blue_set) > fast_memory_budget:
                evicted = False
                for i, evict in enumerate(blue_order):
                    if evict in blue_set and evict not in output_set:
                        blue_order.pop(i)
                        blue_set.discard(evict)
                        ops.append(['store', evict])
                        evicted = True
                        break
                if not evicted:
                    evict = blue_order.pop(0)
                    if evict in blue_set:
                        blue_set.discard(evict)
                        ops.append(['store', evict])
    
    return ops
