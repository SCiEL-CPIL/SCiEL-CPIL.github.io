"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the communication-minimizing DAG pebbling problem.
    
    Args:
        dag: A dictionary with keys:
            - 'n': number of vertices
            - 'parents': list of lists, parents[i] is list of parent indices for vertex i
            - 'inputs': list of input vertex indices (start blue)
            - 'outputs': list of output vertex indices (must finish blue)
        fast_memory_budget: integer, number of red values that can be stored in fast memory
    
    Returns:
        list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s limit
    
    n = dag['n']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build children map
    children = defaultdict(list)
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Topological order (vertices are already topologically ordered by index)
    # Compute depth/level for each vertex
    depth = [0] * n
    for i in range(n):
        if parents[i]:
            depth[i] = max(depth[p] for p in parents[i]) + 1
        else:
            depth[i] = 0
    
    # Find layers
    max_depth = max(depth) if depth else 0
    layers = defaultdict(list)
    for i in range(n):
        layers[depth[i]].append(i)
    
    # For each vertex, compute its "fan-in" (number of parents)
    fan_in = [len(parents[i]) for i in range(n)]
    
    # Strategy: Process layers in topological order.
    # For each layer, compute vertices in an order that minimizes memory pressure.
    # Use a greedy approach: compute vertices whose parents are all in fast memory,
    # and when memory is full, delete vertices that won't be needed soon.
    
    # First, let's try a simple layer-by-layer approach with smart ordering.
    # Within each layer, sort vertices by:
    # 1. Number of children that are outputs (prioritize output paths)
    # 2. Number of children (prioritize high-fanout vertices)
    
    # Precompute: for each vertex, how many of its children are outputs
    child_output_count = [0] * n
    for i in range(n):
        for c in children[i]:
            if c in outputs:
                child_output_count[i] += 1
    
    # Also count total children
    child_count = [len(children[i]) for i in range(n)]
    
    # Build schedule
    schedule = []
    
    # Track state:
    # blue_set: vertices that are currently blue (in slow memory, free)
    # red_set: vertices that are currently red (in fast memory, consume slots)
    # computed: vertices that have been computed at least once
    
    blue_set = set(inputs)
    red_set = set()
    computed = set(inputs)
    
    # We need to process vertices in topological order.
    # A vertex can be computed when all its parents are available (blue or red).
    # After computing, the result is red.
    # To free memory, we can store (make blue) or delete (discard) red values.
    
    # Greedy strategy:
    # Process vertices layer by layer. Within each layer, process vertices
    # in order of decreasing "urgency" (outputs first, then high-fanout).
    
    # For each vertex in topological order:
    # 1. If not yet computed, compute it (requires all parents to be blue or red)
    # 2. If memory is full, delete/store some red values to make room
    
    # Let's use a more sophisticated approach:
    # Process in topological order. Maintain a set of "ready" vertices (all parents available).
    # When we need to compute a vertex and memory is full, we evict the least useful red value.
    
    # Utility of keeping a red value:
    # - If it's an output, very high utility
    # - If it has many children not yet computed, high utility
    # - Otherwise, low utility
    
    # Let's implement this:
    
    # Ready set: vertices whose parents are all in (blue_set | red_set)
    # We process in topological order, but use a priority queue for ready vertices.
    
    import heapq
    
    # Priority: higher is better (we want to process high-priority vertices first)
    # Use negative priority for min-heap
    
    # Actually, let's just process in topological order (by index) and handle memory management.
    
    # Count how many children each vertex has that haven't been computed yet
    remaining_children = [child_count[i] for i in range(n)]
    
    # Process vertices in topological order
    for v in range(n):
        if time.time() > DEADLINE:
            break
        
        # If v is already computed and in memory, skip
        if v in red_set or v in blue_set:
            # If it's blue and we need it as a parent, it's fine
            # If it's red, it's in fast memory, fine
            continue
        
        # Check if all parents are available
        parents_available = all(p in red_set or p in blue_set for p in parents[v])
        if not parents_available:
            # Can't compute yet, skip (shouldn't happen in topological order)
            continue
        
        # Before computing, check if we have room in fast memory
        # The new result will be red, consuming 1 slot
        # Parents that are red will still be red (unless we delete them)
        
        # If memory is full, we need to evict something
        # The new vertex v will be red, so we need len(red_set) < fast_memory_budget
        # Actually, we need len(red_set) + 1 <= fast_memory_budget (for the new result)
        # But parents in red_set are still needed, so we can't delete them yet
        
        # Wait - after computing v, the parents of v might no longer be needed
        # (if all their other children have been computed)
        
        # Let's first try to free up memory by storing/deleting red values
        # that are no longer needed
        
        # A red value x is "no longer needed" if:
        # - All children of x have been computed (i.e., x has no remaining children)
        # - OR x is not an ancestor of any uncomputed vertex
        
        # Simpler: a red value x can be stored (made blue) if all its children
        # that need it have been computed. Actually, once all children are computed,
        # x is never needed again, so we can delete it.
        
        # If x is still needed by some uncomputed child, we should keep it red
        # or store it (blue). Storing frees a fast-memory slot.
        
        # Strategy: if memory is full, store red values that are not outputs
        # and have no remaining uncomputed children. If none exist, store the
        # red value with the fewest remaining children.
        
        while len(red_set) >= fast_memory_budget and red_set:
            # Find a red value to evict
            # Prefer values with 0 remaining children (can delete)
            # Then values with few remaining children (can store)
            
            best_evict = None
            best_score = float('inf')
            
            for r in red_set:
                rem = remaining_children[r]
                if r in outputs:
                    # Don't evict outputs if possible
                    score = 1000000 + rem
                else:
                    score = rem
                
                if score < best_score:
                    best_score = score
                    best_evict = r
            
            if best_evict is None:
                break
            
            # Evict best_evict
            if remaining_children[best_evict] == 0:
                # Can delete
                schedule.append(['delete', best_evict])
                red_set.remove(best_evict)
            else:
                # Store (make blue)
                schedule.append(['store', best_evict])
                red_set.remove(best_evict)
                blue_set.add(best_evict)
        
        # Now compute v
        # First, load any blue parents into fast memory if needed
        # Actually, parents can be blue or red; compute works either way
        # The operation 'load' brings a blue value into fast memory (red)
        
        # Wait, let me re-read the problem:
        # "every red value consumes one fast-memory slot, including newly computed results"
        # "Inputs start blue"
        # "all outputs must finish blue"
        
        # So:
        # - Blue values are in slow memory (free)
        # - Red values are in fast memory (cost 1 slot each)
        # - 'load': move blue -> red (consumes 1 slot)
        # - 'store': move red -> blue (frees 1 slot)
        # - 'compute': compute a vertex (result is red, consumes 1 slot)
        # - 'delete': remove a red value (frees 1 slot, value is lost)
        
        # For compute, I think the parents just need to be available (blue or red).
        # The result is red.
        
        # So before computing v, we don't need to load parents.
        # We just need room for the new red result.
        
        # After computing, we might want to store/delete parents that are no longer needed.
        
        # Compute v
        schedule.append(['compute', v])
        red_set.add(v)
        computed.add(v)
        
        # Update remaining_children for parents of v
        for p in parents[v]:
            remaining_children[p] -= 1
        
        # After computing v, check if any red parents can be evicted
        # (if they have no remaining children)
        for p in parents[v]:
            if p in red_set and remaining_children[p] == 0 and p not in outputs:
                # Can delete p
                schedule.append(['delete', p])
                red_set.remove(p)
        
        # If memory is still full, evict more
        while len(red_set) >= fast_memory_budget and red_set:
            best_evict = None
            best_score = float('inf')
            
            for r in red_set:
                rem = remaining_children[r]
                if r in outputs:
                    score = 1000000 + rem
                else:
                    score = rem
                
                if score < best_score:
                    best_score = score
                    best_evict = r
            
            if best_evict is None:
                break
            
            if remaining_children[best_evict] == 0:
                schedule.append(['delete', best_evict])
                red_set.remove(best_evict)
            else:
                schedule.append(['store', best_evict])
                red_set.remove(best_evict)
                blue_set.add(best_evict)
    
    # At the end, all outputs must be blue
    # Store any red outputs
    for v in outputs:
        if v in red_set:
            schedule.append(['store', v])
            red_set.remove(v)
            blue_set.add(v)
    
    # Check if we exceeded operation limit
    if len(schedule) > 20000:
        # Truncate (this would be invalid, but better than timeout)
        schedule = schedule[:20000]
    
    return schedule
# EVOLVE-BLOCK-END
