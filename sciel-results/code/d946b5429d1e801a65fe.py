"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    inputs = set(dag['inputs'])
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    in_fast = set()
    in_slow = set(inputs)
    computed = set()
    operations = []
    
    for v in topo_order:
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                if p in in_slow:
                    operations.append(['load', p])
                    in_fast.add(p)
                    in_slow.discard(p)
        
        # Compute the vertex
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # Free up memory: delete values no longer needed if over budget
        # Prioritize deleting non-output, non-input values that have no future consumers
        if len(in_fast) > fast_memory_budget:
            # Find candidates for deletion
            candidates = []
            for x in list(in_fast):
                if x in outputs or x in inputs:
                    continue
                # Check if x is needed by any uncomputed vertex
                needed = False
                for c in children.get(x, []):
                    if c not in computed:
                        needed = True
                        break
                if not needed:
                    candidates.append(x)
            # Delete until within budget
            for x in candidates:
                if len(in_fast) <= fast_memory_budget:
                    break
                operations.append(['delete', x])
                in_fast.discard(x)
    
    # Store all outputs that are in fast memory
    for v in outputs:
        if v in in_fast:
            operations.append(['store', v])
            in_fast.discard(v)
            in_slow.add(v)
        elif v in in_slow:
            # Already stored, nothing to do
            pass
        else:
            # Output not in fast or slow - shouldn't happen
            pass
    
    return operations
# EVOLVE-BLOCK-END
