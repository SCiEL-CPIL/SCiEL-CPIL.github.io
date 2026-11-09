"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

_DEADLINE = 55.0  # Safety margin before 60s limit

def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return parents.get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    else:
        return []

def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        # Try to infer from parents
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []

def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs

def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {}
    children = defaultdict(list)
    
    for v in all_v:
        ps = _get_parents(dag, v)
        parents[v] = ps
        for p in ps:
            children[p].append(v)
    
    # Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = [v for v in all_v if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        # Cycle or error, just return all vertices
        return all_v
    
    return topo

def _find_articulation_points(dag):
    """Find articulation points in the DAG (vertices whose removal increases connected components)."""
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    # Build adjacency
    adj = defaultdict(list)
    for v in all_v:
        for p in _get_parents(dag, v):
            adj[p].append(v)
    
    # For DAGs, find vertices that separate the graph
    # A simpler heuristic: find vertices with both parents and children that are "bottlenecks"
    articulation = []
    for v in all_v:
        parents = _get_parents(dag, v)
        children = adj[v]
        if parents and children:
            # Check if this vertex is critical
            articulation.append(v)
    
    return articulation

def _compute_use_counts(dag):
    """Compute how many times each vertex's value is used."""
    all_v = _get_all_vertices(dag)
    use_count = defaultdict(int)
    
    for v in all_v:
        for p in _get_parents(dag, v):
            use_count[p] += 1
    
    # Outputs are also "used"
    for o in _get_outputs(dag):
        use_count[o] += 1
    
    return use_count

def _pebble_divide_conquer(dag, fast_memory_budget, deadline):
    """
    Recursive divide-and-conquer pebbling strategy.
    Returns a list of operations.
    """
    start_time = time.time()
    elapsed = start_time - deadline[0]
    
    if elapsed > _DEADLINE:
        return []
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    use_counts = _compute_use_counts(dag)
    
    # Strategy: Process in topological order, but smartly manage fast memory
    # Keep in fast memory: vertices that are used multiple times and fit in budget
    # Compute and immediately use: vertices used only once
    
    operations = []
    in_fast_memory = set()
    
    # Phase 1: Load all inputs (they start blue, need to be loaded if we want to use them)
    # Actually, inputs start blue. We need to load them to compute their children.
    
    # Determine which vertices to keep in fast memory
    # Priority: high use count, then topological order
    candidates_for_fast = []
    for v in topo:
        if v not in inputs:  # Don't count inputs as candidates for "keeping"
            candidates_for_fast.append((use_counts[v], v))
    
    # Sort by use count descending, then by topological order
    candidates_for_fast.sort(key=lambda x: (-x[0], topo.index(x[1])))
    
    fast_memory_set = set()
    for count, v in candidates_for_fast:
        if len(fast_memory_set) < fast_memory_budget:
            fast_memory_set.add(v)
    
    # Now process in topological order
    for v in topo:
        if time.time() - deadline[0] > _DEADLINE:
            break
        
        parents = _get_parents(dag, v)
        
        if not parents:
            # Input vertex - it starts blue
            # We need to load it if any of its children will use it
            # For now, load it when needed
            pass
        else:
            # Need to compute v
            # Ensure all parents are in fast memory
            for p in parents:
                if p not in in_fast_memory:
                    # Load p from slow memory
                    operations.append(['load', p])
                    in_fast_memory.add(p)
            
            # Compute v
            operations.append(['compute', v])
            in_fast_memory.add(v)
            
            # If v is not in our fast memory set and is used only once, delete it
            if v not in fast_memory_set and use_counts[v] <= 1:
                operations.append(['delete', v])
                in_fast_memory.remove(v)
    
    # Phase 2: Ensure all outputs are in fast memory (blue)
    for o in outputs:
        if o not in in_fast_memory:
            operations.append(['load', o])
            in_fast_memory.add(o)
    
    # Store all outputs
    for o in outputs:
        if o in in_fast_memory:
            operations.append(['store', o])
            in_fast_memory.remove(o)
    
    return operations

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True

def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _naive_schedule_transfers(dag, fast_memory_budget):
    """Estimate naive schedule transfers."""
    all_v = _get_all_vertices(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    
    # Naive: load all inputs, compute all, store all outputs
    # This is an estimate
    transfers = len(inputs) + len(outputs)
    return transfers

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    # Try divide-and-conquer approach
    try:
        ops = _pebble_divide_conquer(dag, fast_memory_budget, deadline)
        
        if not ops:
            # Fallback to baseline
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        # Validate
        if _validate_operations(ops, dag, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # If invalid, try simpler approach
        ops_simple = []
        all_v = _get_all_vertices(dag)
        topo = _topological_order(dag)
        inputs = _get_inputs(dag)
        outputs = _get_outputs(dag)
        
        in_fast = set()
        
        for v in topo:
            if time.time() - deadline[0] > _DEADLINE:
                break
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    ops_simple.append(['load', p])
                    in_fast.add(p)
            ops_simple.append(['compute', v])
            in_fast.add(v)
        
        for o in outputs:
            if o not in in_fast:
                ops_simple.append(['load', o])
                in_fast.add(o)
            ops_simple.append(['store', o])
            in_fast.remove(o)
        
        if _validate_operations(ops_simple, dag, fast_memory_budget):
            if len(ops_simple) <= 20000:
                return ops_simple
        
    except Exception:
        pass
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
