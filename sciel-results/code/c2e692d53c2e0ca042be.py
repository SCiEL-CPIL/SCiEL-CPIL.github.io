"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """
    Returns a list of vertices in topological order.
    vertices: list of vertex IDs
    parents: dict mapping vertex -> list of parent IDs
    """
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return vertices
    return order

def _validate_schedule(dag, ops, fast_memory_budget):
    """
    Validates a schedule by simulating the state machine.
    Returns True if valid, False otherwise.
    """
    vertices = set(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    blue = set(inputs)
    red = set()
    computed = set()
    
    for op, v in ops:
        if v not in vertices:
            return False
        
        if op == 'load':
            if v not in blue or v in red:
                return False
            if len(red) >= fast_memory_budget:
                return False
            red.add(v)
            blue.remove(v)
            
        elif op == 'store':
            if v not in red or v in blue:
                return False
            red.remove(v)
            blue.add(v)
            
        elif op == 'compute':
            if v in red:
                pass # Recompute allowed
            
            for p in parents.get(v, []):
                if p not in red:
                    return False
            
            if v not in red:
                if len(red) >= fast_memory_budget:
                    return False
                red.add(v)
            computed.add(v)
            
        elif op == 'delete':
            if v not in red:
                return False
            red.remove(v)
            
        else:
            return False
            
    for v in outputs:
        if v not in blue:
            return False
            
    return True

def _count_transfers(ops):
    """Count the number of load and store operations (transfers)."""
    count = 0
    for op, v in ops:
        if op == 'load' or op == 'store':
            count += 1
    return count

def _layered_schedule(dag, fast_memory_budget):
    """
    A layered pebbling strategy suitable for layered DAGs.
    This approach processes vertices layer by layer (based on distance from inputs).
    It keeps the current layer and next layer in memory, discarding the previous layer.
    This is often optimal for layered structures with small width.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Compute layers (distance from source)
    # Since it's a DAG, we can use BFS from sources.
    # However, "layer" in pebbling often refers to topological depth.
    # Let's compute the longest path distance from any source.
    
    # First, topological sort
    order = _topological_sort(vertices, parents)
    
    # Compute depth
    depth = {v: 0 for v in vertices}
    for v in order:
        for p in parents.get(v, []):
            if depth[p] + 1 > depth[v]:
                depth[v] = depth[p] + 1
    
    # Group vertices by depth
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    
    max_depth = max(layers.keys()) if layers else 0
    
    ops = []
    in_memory = set()
    
    # We need to be careful: a vertex might be needed by a vertex in a much later layer?
    # In a "layered" DAG, edges usually go from layer i to i+1.
    # If edges can skip layers, this simple layering might fail if we discard too early.
    # Let's check if we can safely discard a vertex after its children in the NEXT layer are computed.
    # Actually, if we process layer by layer, we need to ensure all parents are in memory.
    # If a parent is in layer i-1 and we are computing layer i, the parent must be in memory.
    # After computing all of layer i, can we discard layer i-1?
    # Only if no vertex in layer > i depends on layer i-1.
    # In a standard layered DAG, this is true.
    
    # Let's try a robust version:
    # Process layers 0 to max_depth.
    # Keep track of which vertices are "active" (needed for future computations).
    
    # Pre-compute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # remaining_consumers[v] = number of children of v not yet computed
    # We need to know when a vertex is no longer needed.
    
    # Strategy:
    # 1. Load inputs (layer 0).
    # 2. For each layer L from 0 to max_depth:
    #    a. For each vertex v in layer L:
    #       - If v is an input (L=0), it's already loaded.
    #       - Else, ensure parents are in memory.
    #         - If a parent is not in memory, it must have been discarded.
    #           - If it was an input or output (stored), reload it.
    #           - If it was intermediate (deleted), recompute it? This is expensive.
    #           - To avoid recomputation, we must NOT discard a vertex until all its descendants are computed.
    #           - This is the "safe" discard rule.
    
    # Let's implement the "safe" discard rule with a priority on discarding vertices from older layers first.
    
    # Reset state
    in_memory = set()
    ops = []
    
    # We need a simulation of the schedule to be sure.
    # Let's use a greedy approach:
    # 1. Iterate through vertices in topological order.
    # 2. For each vertex v:
    #    - If v is an input:
    #      - If not in memory, load it.
    #    - Else:
    #      - For each parent p:
    #        - If p not in memory:
    #          - If p was stored (output or input that was stored?), load it.
    #          - Else, we need to recompute p? Or we made a mistake in discarding.
    #          - To simplify, let's assume we only discard vertices that have NO remaining children.
    #          - This guarantees we never need to recompute.
    
    # This is essentially the _bottom_up_schedule logic but applied in forward topological order?
    # No, bottom-up was reverse topological.
    # Forward topological with "discard when no remaining children" is also valid and often better for forward flow.
    
    # Let's try a forward schedule:
    # 1. in_memory = set()
    # 2. remaining_children = {v: len(children[v]) for v in vertices}
    # 3. For v in topological_order:
    #    - If v is input:
    #      - If len(in_memory) >= budget:
    #          - Find a vertex u in in_memory with remaining_children[u] == 0.
    #          - If found, delete/store u.
    #      - Load v.
    #    - Else:
    #      - For each parent p of v:
    #          - If p not in in_memory:
    #              - This should not happen if we only discard when remaining_children == 0.
    #              - Because if p is a parent of v, and v is not computed, remaining_children[p] > 0.
    #              - So p should still be in memory.
    #              - UNLESS p was an input and we discarded it? No, we only discard if remaining_children == 0.
    #      - Compute v.
    #      - Update remaining_children for parents.
    #      - If a parent's remaining_children becomes 0, we CAN discard it.
    #        - But do we HAVE to? Maybe we need space for v.
    #        - If we have space, keep it? Or discard immediately to minimize memory?
    #        - Discarding immediately is safer for memory but might increase transfers if we need to reload?
    #        - No, if remaining_children == 0, we will NEVER need it again. So discard is always good.
    #        - Exception: if it's an output, we must 'store' it, not 'delete'.
    
    # Let's implement this forward greedy schedule.
    
    # 1. Initialize
    in_memory = set()
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # 2. Iterate
    for v in order:
        # Check if we need to make space for v (if not input) or its parents (if not input)
        # Actually, parents must be in memory.
        
        # If v is not an input, its parents must be in memory.
        # They should be, based on the invariant.
        
        # If v is an input, we need to load it.
        if not parents.get(v, []):
            if v not in in_memory:
                # Need space?
                if len(in_memory) >= fast_memory_budget:
                    # Find a discardable vertex
                    discarded = False
                    for u in list(in_memory):
                        if remaining_children[u] == 0:
                            if u in outputs:
                                ops.append(['store', u])
                            else:
                                ops.append(['delete', u])
                            in_memory.remove(u)
                            discarded = True
                            break
                    if not discarded:
                        return None # Should not happen if logic is correct
                ops.append(['load', v])
                in_memory.add(v)
        else:
            # v is not an input
            # Ensure parents are in memory (they should be)
            for p in parents[v]:
                if p not in in_memory:
                    # This is a logic error in our assumption
                    return None
            
            # Check space for v
            if v not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    discarded = False
                    for u in list(in_memory):
                        if remaining_children[u] == 0:
                            if u in outputs:
                                ops.append(['store', u])
                            else:
                                ops.append(['delete', u])
                            in_memory.remove(u)
                            discarded = True
                            break
                    if not discarded:
                        return None
            
            ops.append(['compute', v])
            in_memory.add(v)
            
            # Update remaining_children for parents
            for p in parents[v]:
                remaining_children[p] -= 1
    
    # 3. Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    try:
        best_schedule = None
        best_transfers = float('inf')
        
        # Strategy 1: Bottom-up (Reverse Topological)
        # This is good for "keeping the tail" in memory.
        schedule_bottom = _bottom_up_schedule(dag, fast_memory_budget)
        if schedule_bottom is not None and _validate_schedule(dag, schedule_bottom, fast_memory_budget):
            t = _count_transfers(schedule_bottom)
            if t < best_transfers:
                best_transfers = t
                best_schedule = schedule_bottom
        
        # Strategy 2: Forward Greedy (Topological with immediate discard)
        # This is good for "streaming" through the DAG.
        schedule_forward = _layered_schedule(dag, fast_memory_budget)
        if schedule_forward is not None and _validate_schedule(dag, schedule_forward, fast_memory_budget):
            t = _count_transfers(schedule_forward)
            if t < best_transfers:
                best_transfers = t
                best_schedule = schedule_forward
        
        # If we found a valid schedule, return it
        if best_schedule is not None:
            return best_schedule
        
        # Fallback to baseline if both failed
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
            
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

def _bottom_up_schedule(dag, fast_memory_budget):
    """
    Bottom-up pebbling strategy:
    - Process vertices in reverse topological order
    - Store final outputs in fast memory
    - Discard intermediate results immediately after their last consumer is processed
    - This minimizes peak memory usage
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # remaining_consumers[v] = number of children of v that have NOT been processed yet
    remaining_consumers = {v: len(children[v]) for v in vertices}
    
    ops = []
    in_memory = set()
    
    # Identify inputs
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    # Process in reverse topological order (bottom-up)
    for v in reversed(order):
        if v in inputs:
            # Inputs are loaded on demand when needed by their consumers
            continue
        
        # 1. Ensure parents are in memory for computation
        for p in parents[v]:
            if p not in in_memory:
                if p in inputs:
                    # Load input on demand
                    if len(in_memory) >= fast_memory_budget:
                        # Need to free space before loading
                        freed = False
                        for u in list(in_memory):
                            if remaining_consumers[u] == 0:
                                if u in outputs:
                                    ops.append(['store', u])
                                else:
                                    ops.append(['delete', u])
                                in_memory.remove(u)
                                freed = True
                                break
                        if not freed:
                            return None # Cannot fit even one more
                    ops.append(['load', p])
                    in_memory.add(p)
                else:
                    # Parent should be in memory if it's not an input
                    # If it's not, something went wrong
                    return None
        
        # 2. Check if we have space for v
        if len(in_memory) >= fast_memory_budget:
            # Find a victim to evict - a vertex with 0 remaining consumers
            freed = False
            for u in list(in_memory):
                if remaining_consumers[u] == 0:
                    if u in outputs:
                        ops.append(['store', u])
                    else:
                        ops.append(['delete', u])
                    in_memory.remove(u)
                    freed = True
                    break
            
            if not freed:
                # Cannot free space, schedule fails
                return None

        # 3. Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        
        # 4. Update remaining consumers for parents and evict them if done
        for p in parents[v]:
            remaining_consumers[p] -= 1
            if remaining_consumers[p] == 0:
                # p is no longer needed by any consumer
                if p in in_memory:
                    if p in outputs:
                        ops.append(['store', p])
                    else:
                        ops.append(['delete', p])
                    in_memory.remove(p)
    
    # 5. Final cleanup - store any remaining outputs
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

# EVOLVE-BLOCK-END
