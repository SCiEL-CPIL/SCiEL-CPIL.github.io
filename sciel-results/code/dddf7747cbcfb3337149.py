# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Minimize transfers in DAG pebbling.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Extract DAG structure robustly
    try:
        if hasattr(dag, 'vertices'):
            vertices = list(dag.vertices)
        elif isinstance(dag, dict) and 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif isinstance(dag, dict):
            vertices = list(dag.keys())
        else:
            vertices = list(range(len(dag)))
    except:
        vertices = list(range(len(dag))) if isinstance(dag, (list, tuple)) else list(range(10))
    
    try:
        if hasattr(dag, 'parents'):
            parents = dict(dag.parents)
        elif isinstance(dag, dict) and 'parents' in dag:
            parents = dict(dag['parents'])
        else:
            parents = {}
    except:
        parents = {}
    
    try:
        if hasattr(dag, 'inputs'):
            inputs = set(dag.inputs)
        elif isinstance(dag, dict) and 'inputs' in dag:
            inputs = set(dag['inputs'])
        else:
            inputs = set()
    except:
        inputs = set()
    
    try:
        if hasattr(dag, 'outputs'):
            outputs = set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            outputs = set(dag['outputs'])
        else:
            outputs = set()
    except:
        outputs = set()
    
    # Build children map and position map
    children = {v: [] for v in vertices}
    pos_map = {v: i for i, v in enumerate(vertices)}
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in children:
                    children[p].append(v)
    
    # State
    fast_memory = set(inputs)
    slow_memory = set()  # Values that have been stored
    computed = set(inputs)
    operations = []
    
    # Process in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        if len(operations) >= 20000:
            break
        
        if v in computed:
            continue
        
        # Load missing parents
        for p in parents.get(v, []):
            if p not in fast_memory:
                operations.append(['load', p])
                fast_memory.add(p)
                slow_memory.discard(p)  # Now in fast, not slow
        
        # Compute
        operations.append(['compute', v])
        computed.add(v)
        fast_memory.add(v)
        
        # Free memory if over budget
        while len(fast_memory) > fast_memory_budget:
            best_candidate = None
            best_priority = float('inf')
            
            for fm_v in fast_memory:
                if fm_v in inputs or fm_v in outputs:
                    continue
                if fm_v not in computed:
                    continue
                
                # Find next use position
                next_use_pos = float('inf')
                for c in children.get(fm_v, []):
                    if c not in computed and c in pos_map:
                        next_use_pos = min(next_use_pos, pos_map[c])
                
                if next_use_pos == float('inf'):
                    priority = 0  # Can delete
                else:
                    priority = next_use_pos
            
            if priority < best_priority:
                best_priority = priority
                best_candidate = fm_v
            
            if best_candidate is None:
                break
            
            can_delete = all(c in computed for c in children.get(best_candidate, []))
            if can_delete:
                operations.append(['delete', best_candidate])
            else:
                operations.append(['store', best_candidate])
                slow_memory.add(best_candidate)
            fast_memory.discard(best_candidate)
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in fast_memory:
            operations.append(['load', o])
            fast_memory.add(o)
            slow_memory.discard(o)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
