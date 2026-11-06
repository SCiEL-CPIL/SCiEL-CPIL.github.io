import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    MAX_OPS = 20000
    
    # Parse DAG
    if isinstance(dag, dict):
        parents = dag['parents']
        outputs = dag['outputs']
    else:
        parents = dag.parents
        outputs = dag.outputs
    n = len(parents)
    
    # Children list
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Inputs: vertices with no parents
    inputs = [i for i in range(n) if len(parents[i]) == 0]
    
    # remaining_use[u] = number of children of u not yet processed
    remaining_use = [len(children[u]) for u in range(n)]
    
    ops = []
    blue = set(inputs)
    computed = set(inputs)  # Inputs are already "computed"
    
    def evict_one(exclude=None, protect_outputs=False):
        """Evict the blue vertex with lowest remaining_use."""
        nonlocal blue
        if exclude is None:
            exclude = set()
        candidates = [u for u in blue if u not in exclude]
        if protect_outputs:
            candidates = [u for u in candidates if u not in outputs]
        if not candidates:
            # If all candidates are outputs, evict the one with lowest remaining_use
            candidates = [u for u in blue if u not in exclude]
        if not candidates:
            return False
        # Pick the one with lowest remaining_use, breaking ties by vertex id
        evict_v = min(candidates, key=lambda u: (remaining_use[u], u))
        ops.append(['delete', evict_v])
        blue.remove(evict_v)
        return True
    
    for v in range(n):
        if time.time() - start_time > DEADLINE or len(ops) >= MAX_OPS:
            break
        
        if v in blue:
            # Already blue, no action needed
            continue
        
        # Need to compute v
        # Ensure all parents are blue
        for p in parents[v]:
            if p not in blue:
                # Need to load p
                if len(blue) >= fast_memory_budget:
                    # Make room
                    # Don't evict parents of v that we still need to load
                    # But we can evict parents that are already blue (we'll use them, then they stay blue)
                    # Actually, we need all parents to be blue before computing v
                    # So we should not evict any parent of v
                    evict_one(exclude=set(parents[v]), protect_outputs=True)
                
                ops.append(['load', p])
                blue.add(p)
        
        # Compute v
        ops.append(['compute', v])
        blue.add(v)
        computed.add(v)
        
        # Decrement remaining_use for parents
        for p in parents[v]:
            remaining_use[p] -= 1
        
        # If over budget, evict
        while len(blue) > fast_memory_budget and len(ops) < MAX_OPS:
            evict_one(exclude={v}, protect_outputs=True)
    
    # Ensure all outputs are blue
    for out in outputs:
        if time.time() - start_time > DEADLINE or len(ops) >= MAX_OPS:
            break
        if out not in blue:
            if len(blue) >= fast_memory_budget:
                evict_one(exclude=set(outputs), protect_outputs=False)
            ops.append(['load', out])
            blue.add(out)
    
    return ops
