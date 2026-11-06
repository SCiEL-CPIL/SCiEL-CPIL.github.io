def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 80
    MAX_OPS = 20000
    
    # Parse the DAG
    if isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {v: list(dag[v]) for v in vertices}
    elif isinstance(dag, list):
        vertices = [item[0] for item in dag]
        parents = {item[0]: list(item[1]) for item in dag}
    else:
        raise ValueError("Unknown DAG format")
    
    # Compute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    # State: 'blue' (in fast memory), 'red' (on disk)
    state = {}
    for v in vertices:
        if not parents[v]:
            state[v] = 'blue'  # inputs start blue
        else:
            state[v] = 'red'  # not computed yet
    
    # Remaining use count
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    # Operations list
    operations = []
    
    # Fast memory usage
    def fast_memory_used():
        return sum(1 for v in vertices if state[v] == 'blue')
    
    # Process vertices in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE or len(operations) >= MAX_OPS:
            break
        
        # Ensure all parents are blue
        for p in parents[v]:
            if state[p] == 'red':
                # Load p
                operations.append(['load', p])
                state[p] = 'blue'
        
        # Compute v
        operations.append(['compute', v])
        state[v] = 'blue'
        
        # Update remaining uses
        for child in children[v]:
            remaining_uses[child] -= 1
        
        # If fast memory is full, store a value to disk
        while fast_memory_used() > fast_memory_budget:
            if len(operations) >= MAX_OPS:
                break
            # Find the value with the lowest remaining uses
            candidates = [u for u in vertices if state[u] == 'blue']
            if not candidates:
                break
            # Choose the value with the lowest remaining uses
            best = min(candidates, key=lambda u: (remaining_uses[u], len(children[u])))
            operations.append(['store', best])
            state[best] = 'red'
    
    return operations
