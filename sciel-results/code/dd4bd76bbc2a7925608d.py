# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        topo_order = dag.get('topo_order', None)
    else:
        raise ValueError("Unknown dag format")
    
    if topo_order is None:
        in_degree = {v: 0 for v in vertices}
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                in_degree[v] = in_degree.get(v, 0) + 1
                children[p].append(v)
        
        queue = [v for v in vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            v = queue.pop(0)
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        if len(topo_order) != len(vertices):
            topo_order = list(vertices)
    
    # Blue = slow memory, Red = fast memory
    # Inputs start blue (in slow memory)
    # Outputs must finish blue (in slow memory)
    
    fast_mem = set()  # red pebbles
    slow_mem = set(inputs)  # blue pebbles (initially just inputs)
    operations = []
    
    # Count remaining uses for each vertex to decide when to delete
    # A vertex is needed until all its children are computed
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # remaining_children[v] = number of children of v that haven't been computed yet
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Process in topological order
    computed_set = set()
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            # Load input into fast memory
            if v not in fast_mem:
                operations.append(['load', v])
                fast_mem.add(v)
            computed_set.add(v)
        else:
            # Ensure parents are in fast memory
            for p in parents.get(v, []):
                if p not in fast_mem:
                    if p in slow_mem:
                        operations.append(['load', p])
                        fast_mem.add(p)
                    else:
                        # Parent not available? This shouldn't happen with proper topological order
                        # If parent was deleted, we need to recompute it
                        # For now, skip (shouldn't occur)
                        pass
            
            # Compute v
            operations.append(['compute', v])
            fast_mem.add(v)
            computed_set.add(v)
        
        # After computing v, decrement remaining_children for its parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
        
        # Manage memory: if over budget, delete vertices that are no longer needed
        # A vertex can be deleted if:
        #   1. It's not an output (or output already stored)
        #   2. All its children have been computed (remaining_children == 0)
        #   3. It's not needed for any future computation
        
        while len(fast_mem) > fast_memory_budget:
            # Find a vertex to delete
            # Prefer vertices that are no longer needed
            candidates = []
            for u in fast_mem:
                # Can delete if: not output, or output already stored
                # And all children computed
                if u not in outputs and remaining_children[u] == 0:
                    candidates.append(u)
            
            if not candidates:
                # No safe candidates, delete something that's an output but we'll reload later
                # Or delete a non-output that still has children (will need to reload)
                # Prefer non-outputs first
                non_outputs = [u for u in fast_mem if u not in outputs]
                if non_outputs:
                    # Delete the one with most remaining children? Or least?
                    # Delete the one that's hardest to recompute (most parents)
                    u = min(non_outputs, key=lambda x: len(parents.get(x, [])))
                else:
                    u = next(iter(fast_mem))
                operations.append(['delete', u])
                fast_mem.discard(u)
            else:
                # Delete a safe candidate
                # Prefer the one with most remaining children (least useful)
                u = max(candidates, key=lambda x: remaining_children[x])
                operations.append(['delete', u])
                fast_mem.discard(u)
    
    # Store all outputs to slow memory (they must finish blue)
    for v in outputs:
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
        elif v in slow_mem:
            # Already in slow memory
            pass
        else:
            # Need to compute it? Shouldn't happen
            pass
    
    return operations
# EVOLVE-BLOCK-END
