<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling to minimize communication.
    
    dag: dict with keys:
      - 'vertices': list of vertex ids
      - 'edges': list of (parent, child) tuples, or similar
      - 'inputs': list of input vertex ids (start blue)
      - 'outputs': list of output vertex ids (must end blue)
      - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
    
    fast_memory_budget: int, max number of red values in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Parse the DAG structure
    # Try to handle different possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        parents_dict = dag.get('parents', {})
        children_dict = dag.get('children', {})
    else:
        # Fallback: assume dag is a list of vertex dicts or similar
        raise ValueError(f"Unexpected dag format: {type(dag)}")
    
    # Build adjacency if needed
    if not parents_dict and edges:
        parents_dict = defaultdict(list)
        children_dict = defaultdict(list)
        for parent, child in edges:
            parents_dict[child].append(parent)
            children_dict[parent].append(child)
    elif not children_dict and edges:
        children_dict = defaultdict(list)
        for parent, child in edges:
            children_dict[parent].append(child)
    elif not children_dict:
        # Build from parents
        children_dict = defaultdict(list)
        for v, pars in parents_dict.items():
            for p in pars:
                children_dict[p].append(v)
    
    # Topological sort
    in_degree = {v: len(parents_dict.get(v, [])) for v in vertices}
    topo_queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while topo_queue:
        v = topo_queue.popleft()
        topo_order.append(v)
        for c in children_dict.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                topo_queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle or disconnected - use original order
        topo_order = list(vertices)
    
    # State tracking
    # fast_memory: set of vertices currently in fast memory (red or blue)
    # slow_memory: set of vertices stored in slow memory
    # computed: set of vertices that have been computed at least once
    fast_memory = set(inputs)  # Inputs start blue in fast memory
    slow_memory = set()
    computed = set()
    
    # Track what's in fast memory: value is available
    # For pebbling: 
    #   - Blue in fast memory = available, no cost to use
    #   - Red in fast memory = available, consumes slot
    #   - In slow memory = need to load (cost 1) to use
    #   - Not anywhere = need to compute (requires parents available)
    
    operations = []
    
    # Strategy: Process in topological order
    # For each vertex, ensure parents are available in fast memory
    # Then compute the vertex
    # After computing, decide whether to keep in fast memory or store to slow
    
    # We need to track which vertices' values are currently in fast memory
    # and which are in slow memory
    
    # Let's use a different approach:
    # - available_fast: set of vertices whose values are in fast memory
    # - available_slow: set of vertices whose values are in slow memory
    # - When we compute a vertex, it goes to fast memory (if budget allows)
    # - We can delete from fast memory (frees slot) or store to slow memory
    
    available_fast = set(inputs)
    available_slow = set()
    
    # Process in topological order
    for v in topo_order:
        # Check if we're close to deadline
        if time.time() > deadline:
            break
        
        # Ensure all parents are available in fast memory
        for p in parents_dict.get(v, []):
            if p not in available_fast:
                if p in available_slow:
                    # Load from slow to fast
                    # Check budget
                    if len(available_fast) >= fast_memory_budget:
                        # Need to free a slot - delete something not needed
                        # Find a vertex in fast memory that's not an input, not v, 
                        # and whose children have all been processed or will be recomputed
                        # Simple: delete a non-input vertex
                        to_delete = None
                        for x in list(available_fast):
                            if x not in inputs and x != v:
                                to_delete = x
                                break
                        if to_delete is not None:
                            operations.append(['delete', to_delete])
                            available_fast.discard(to_delete)
                            if to_delete in available_slow:
                                pass  # still in slow
                        else:
                            # Can't free space - this shouldn't happen if budget is sane
                            break
                    
                    operations.append(['load', p])
                    available_fast.add(p)
                else:
                    # Parent not available anywhere - need to recompute
                    # This is complex - for now, skip or handle simply
                    # In a proper implementation, we'd recompute the parent
                    pass
        
        # Compute v if all parents are in fast memory
        all_parents_available = all(p in available_fast for p in parents_dict.get(v, []))
        if all_parents_available or not parents_dict.get(v, []):
            # For inputs, they're already available
            if v not in available_fast:
                # Need to compute
                if len(available_fast) >= fast_memory_budget:
                    # Free a slot
                    to_delete = None
                    for x in list(available_fast):
                        if x not in inputs and x != v:
                            to_delete = x
                            break
                    if to_delete is not None:
                        operations.append(['delete', to_delete])
                        available_fast.discard(to_delete)
                
                operations.append(['compute', v])
                available_fast.add(v)
        
        # After computing, decide what to do with v
        # If v is an output, it must end up in fast memory (blue)
        # Otherwise, we can store to slow memory to free fast memory
        
        if v in outputs:
            # Keep in fast memory - must end blue
            pass
        else:
            # Store to slow memory to free fast memory slot
            # But only if we might need it again
            children_of_v = children_dict.get(v, [])
            # Check if any unprocessed child needs v
            unprocessed_children = [c for c in children_of_v if c not in computed]
            if unprocessed_children and v in available_fast:
                # Store to slow memory
                operations.append(['store', v])
                available_slow.add(v)
                available_fast.discard(v)
            # If no unprocessed children, we can just leave it or delete
            # Actually, if no unprocessed children, we don't need it anymore
    
        # Mark as computed
        computed.add(v)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in available_fast:
            if v in available_slow:
                # Load it
                if len(available_fast) >= fast_memory_budget:
                    to_delete = None
                    for x in list(available_fast):
                        if x not in inputs and x not in outputs:
                            to_delete = x
                            break
                    if to_delete is not None:
                        operations.append(['delete', to_delete])
                        available_fast.discard(to_delete)
                operations.append(['load', v])
                available_fast.add(v)
            else:
                # Need to recompute - this is a problem
                # For now, just mark as load (shouldn't happen)
                operations.append(['load', v])
                available_fast.add(v)
    
    # Limit operations to 20000
    operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to reconsider. The task says "Inputs start blue, all outputs must finish blue". In pebbling terminology:
- Blue pebble = value is in fast memory (available for free)
- Red pebble = value is in fast memory but was just computed (consumes a slot)

Actually, re-reading: "every red value consumes one fast-memory slot, including newly computed results". So:
- Blue values don't consume slots? Or do they?

Let me re-read: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"

I think:
- Blue = in fast memory, doesn't count against budget (or is free)
- Red = in fast memory, counts against budget

Actually, that doesn't make sense for pebbling. Let me think differently:

In standard DAG pebbling:
- A pebble (red or blue) on a vertex means its value is in fast memory
- Red pebbles are those placed by computation
- Blue pebbles are those that are "permanent" (like inputs)

The constraint is typically on the total number of pebbles (red + blue) in fast memory.

But the task says "every red value consumes one fast-memory slot". This suggests blue values might not consume slots, or the budget only applies to red values.

Let me just implement a reasonable strategy and see. The key is to minimize transfers (load + store operations).

Let me reconsider the approach with a cleaner implementation:

