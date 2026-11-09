"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    # Try to import baseline as fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except:
        pebble_baseline = None
    
    # Parse DAG structure robustly
    n = None
    parents = None
    outputs = None
    
    # Try various attribute names
    for attr in ['num_vertices', 'n', 'size']:
        if hasattr(dag, attr):
            n = getattr(dag, attr)
            break
    
    if n is None and isinstance(dag, dict):
        if 'n' in dag:
            n = dag['n']
        elif 'num_vertices' in dag:
            n = dag['num_vertices']
        elif 'vertices' in dag:
            n = len(dag['vertices'])
    
    if n is None and hasattr(dag, '__len__'):
        try:
            n = len(dag)
        except:
            pass
    
    # Get parents
    for attr in ['parents', 'parent_list', 'in_edges']:
        if hasattr(dag, attr):
            parents = getattr(dag, attr)
            break
    
    if parents is None and isinstance(dag, dict):
        for key in ['parents', 'parent_list', 'in_edges']:
            if key in dag:
                parents = dag[key]
                break
    
    # Get outputs
    for attr in ['outputs', 'output_vertices', 'sinks']:
        if hasattr(dag, attr):
            outputs = getattr(dag, attr)
            break
    
    if outputs is None and isinstance(dag, dict):
        for key in ['outputs', 'output_vertices', 'sinks']:
            if key in dag:
                outputs = dag[key]
                break
    
    # If we couldn't parse, use baseline
    if n is None or parents is None or outputs is None:
        if pebble_baseline is not None:
            return pebble_baseline(dag, fast_memory_budget)
        # Last resort: return empty
        return []
    
    # Normalize parents to list of lists
    if isinstance(parents, dict):
        parents_list = [parents.get(i, []) for i in range(n)]
    elif isinstance(parents, (list, tuple)):
        parents_list = [list(p) if not isinstance(p, int) else [p] for p in parents]
        # If it's a flat list of parent indices, we need to group them
        # But typically it's already grouped
    else:
        parents_list = [list(parents[i]) for i in range(n)]
    
    # Normalize outputs
    if isinstance(outputs, (set, frozenset)):
        outputs_list = list(outputs)
    elif isinstance(outputs, (list, tuple)):
        outputs_list = list(outputs)
    else:
        outputs_list = [outputs]
    
    output_set = set(outputs_list)
    
    # Build children map
    children = defaultdict(list)
    for i in range(n):
        for p in parents_list[i]:
            if p is not None:
                children[p].append(i)
    
    # Compute reuse count (out-degree)
    reuse_count = [len(children[i]) for i in range(n)]
    
    # Identify inputs (vertices with no parents)
    inputs = [i for i in range(n) if len(parents_list[i]) == 0]
    
    # State
    red = set()   # vertices in fast memory
    slow = set()  # vertices in slow memory
    operations = []
    
    # Inputs start blue (in slow)
    for i in inputs:
        slow.add(i)
    
    def can_evict(v, current_vertex):
        """Check if vertex v can be evicted (not needed for future computations)."""
        for child in children[v]:
            if child > current_vertex:
                return False
        return True
    
    def get_value(v):
        """Get value of keeping vertex v in fast memory."""
        # Higher is better
        # Use reuse count as proxy
        return reuse_count[v]
    
    # Process in topological order
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break
        
        # If already red, skip
        if v in red:
            continue
        
        # Check if all parents are red
        all_red = True
        for p in parents_list[v]:
            if p is not None and p not in red:
                all_red = False
                break
        
        if all_red:
            # Compute without transfers
            operations.append(['compute', v])
        else:
            # Load missing parents
            for p in parents_list[v]:
                if p is not None and p not in red:
                    operations.append(['load', p])
                    red.add(p)
                    if p in slow:
                        slow.remove(p)
            operations.append(['compute', v])
        
        # Decide whether to keep v red
        if len(red) < fast_memory_budget:
            red.add(v)
        else:
            # Need to evict
            best_evict = None
            best_val = float('inf')
            for r in red:
                if can_evict(r, v):
                    val = get_value(r)
                    if val < best_val:
                        best_val = val
                        best_evict = r
            
            if best_evict is not None:
                operations.append(['delete', best_evict])
                red.remove(best_evict)
                slow.add(best_evict)
                red.add(v)
            else:
                # Can't evict, store v
                operations.append(['store', v])
                slow.add(v)
    
    # Ensure outputs are blue
    for v in outputs_list:
        if v in red:
            operations.append(['store', v])
            red.remove(v)
            slow.add(v)
    
    # Enforce operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
