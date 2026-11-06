"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    try:
        return _pebble_improved(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_improved(dag, fast_memory_budget, deadline):
    """
    Improved pebbling strategy with better deletion heuristics:
    - Process vertices in topological order
    - Use reference counting to know when values are no longer needed
    - Delete values as soon as they are no longer needed (not just when memory is full)
    - Prioritize deleting values with 0 remaining references
    - Ensure outputs end up blue (in fast memory)
    """
    # Extract DAG structure
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    # Get parents for each vertex (topologically ordered)
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Build children map for reference counting
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
    
    # Compute reference count: how many times each value will be needed
    ref_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            ref_count[p] += 1
    
    # Initial state: inputs are blue (in fast memory)
    in_fast_memory = set(inputs)
    # Track which vertices have been computed
    computed = set(inputs)
    operations = []
    
    # Fast memory budget
    budget = fast_memory_budget
    
    def make_room(operations, in_fast_memory, ref_count, outputs, budget, v_current):
        """Make room in fast memory by deleting unnecessary values."""
        while len(in_fast_memory) > budget:
            # Find a value to delete: prioritize ref_count 0, non-output
            best_candidate = None
            for candidate in in_fast_memory:
                if candidate not in outputs and ref_count[candidate] == 0:
                    best_candidate = candidate
                    break
            
            if best_candidate is None:
                # Delete any non-output value
                for candidate in in_fast_memory:
                    if candidate not in outputs:
                        best_candidate = candidate
                        break
            
            if best_candidate is None:
                # Emergency: delete any value (shouldn't happen if budget >= num_outputs)
                for candidate in in_fast_memory:
                    best_candidate = candidate
                    break
            
            if best_candidate is not None:
                operations.append(['delete', best_candidate])
                in_fast_memory.discard(best_candidate)
            else:
                break
    
    def free_unneeded(operations, in_fast_memory, ref_count, outputs):
        """Free values that are no longer needed to keep memory usage low."""
        to_delete = []
        for v in in_fast_memory:
            if v not in outputs and ref_count[v] == 0:
                to_delete.append(v)
        for v in to_delete:
            operations.append(['delete', v])
            in_fast_memory.discard(v)
    
    # Process vertices in topological order (excluding inputs which are already computed)
    for v in vertices:
        if v in inputs:
            continue
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Need to load parent from slow memory
                operations.append(['load', p])
                in_fast_memory.add(p)
                # Check if we exceed budget - if so, we need to delete something
                make_room(operations, in_fast_memory, ref_count, outputs, budget, v)
        
        # Compute the vertex
        operations.append(['compute', v])
        in_fast_memory.add(v)
        computed.add(v)
        
        # Check if we exceed budget after compute
        make_room(operations, in_fast_memory, ref_count, outputs, budget, v)
        
        # Decrement reference count for parents
        for p in parents[v]:
            ref_count[p] -= 1
        
        # Free values that are no longer needed
        free_unneeded(operations, in_fast_memory, ref_count, outputs)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast_memory:
            operations.append(['load', v])
            in_fast_memory.add(v)
            # Make room if needed
            make_room(operations, in_fast_memory, ref_count, outputs, budget, v)
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate and try to ensure outputs are still blue
        operations = operations[:20000]
        # Re-ensure outputs are in fast memory
        in_fast_memory_final = set()
        # Reconstruct state from operations (simplified - just ensure outputs are loaded)
        for v in outputs:
            if v not in in_fast_memory_final:
                operations.append(['load', v])
                in_fast_memory_final.add(v)
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
