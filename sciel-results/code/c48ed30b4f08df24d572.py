"""Communication-minimizing DAG pebbling with memory-aware scheduling."""
# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize total transfers (load + store operations).
    
    Args:
        dag: dict with 'parents' (list of lists, topologically ordered), 'outputs' (list of vertex indices)
        fast_memory_budget: int, max number of red (in-memory) values allowed
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Parse DAG structure
    parents = dag['parents']
    outputs = set(dag['outputs'])
    n = len(parents)
    
    # Build child lists for reverse topological info
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Identify input vertices (no parents) and compute in-degree for scheduling
    is_input = [len(parents[v]) == 0 for v in range(n)]
    is_output = [v in outputs for v in range(n)]
    
    # Compute topological order (already given, but verify)
    # parents are topologically ordered, so parents[v] < v for all v
    
    # State: color of each vertex
    # 0 = white (not computed), 1 = blue (in slow memory), 2 = red (in fast memory)
    # Inputs start blue
    color = [0] * n
    for v in range(n):
        if is_input[v]:
            color[v] = 1  # blue (in slow memory)
    
    # Fast memory tracking: set of vertices currently red
    fast_mem = set()
    ops = []
    
    # Count transfers (load + store)
    transfer_count = 0
    
    # For each vertex, track how many times we need to have it available
    # and when it's last needed
    # Compute "last needed" time in topological order
    # A vertex is needed by all its children
    # last_needed[v] = max of children indices, or v itself if it's an output
    last_needed = list(range(n))
    for v in range(n - 1, -1, -1):
        if children[v]:
            last_needed[v] = max(last_needed[c] for c in children[v])
        if is_output[v]:
            last_needed[v] = max(last_needed[v], v)  # must persist until end
    
    # Strategy: Process vertices in topological order.
    # For each vertex, if it's not red, we need to load it (if parent) or compute it.
    # After computing, keep it red if memory allows, else store it.
    # Before computing a vertex, ensure all parents are available (red or load them).
    
    # Priority: vertices with more remaining uses should be kept in memory
    # Compute use_count: number of children that still need this vertex
    remaining_uses = [len(children[v]) for v in range(n)]
    for v in range(n):
        if is_output[v]:
            remaining_uses[v] += 1  # output must be materialized
    
    # Greedy: process in topological order, manage memory
    # For each vertex v:
    #   1. Ensure all parents are available (red, or load from blue)
    #   2. Compute v (if not already computed)
    #   3. Decide whether to keep v red or store it
    #   4. For parents no longer needed, delete from fast memory
    
    # We'll use a more sophisticated approach:
    # Process in topological order. Maintain a priority queue of "available" vertices.
    # Actually, let's do a simpler greedy:
    
    # For each vertex in topological order:
    # - If it's an input, it starts blue. Load it when needed.
    # - If it has parents, all parents must be red (or loaded) before computing.
    # - After computing, add to fast memory if budget allows.
    # - After a parent is no longer needed by any unprocessed vertex, evict it if memory is tight.
    
    # Let's track for each vertex whether it's been computed
    computed = [False] * n
    
    # For inputs, they're already "computed" (blue in slow memory)
    for v in range(n):
        if is_input[v]:
            computed[v] = True
    
    # Process in topological order
    for v in range(n):
        if time.time() > deadline:
            break
        
        if computed[v] and color[v] == 2:
            # Already in fast memory, nothing to do
            pass
        elif computed[v] and color[v] == 1:
            # In slow memory, need to load if we want it red
            # We'll load it when actually needed by a child
            pass
        else:
            # Need to compute this vertex
            # First, ensure all parents are red
            for p in parents[v]:
                if color[p] == 2:
                    continue  # already red
                elif color[p] == 1:
                    # Load from slow memory
                    # Check if we have memory
                    if len(fast_mem) < fast_memory_budget:
                        ops.append(['load', p])
                        color[p] = 2
                        fast_mem.add(p)
                        transfer_count += 1
                    else:
                        # Need to evict something
                        # Find a parent to evict: one with lowest remaining_uses
                        # or one that's not needed by current vertex's computation
                        evict_v = None
                        min_uses = float('inf')
                        for pm in fast_mem:
                            if pm not in parents[v]:  # don't evict a parent we need
                                if remaining_uses[pm] < min_uses:
                                    min_uses = remaining_uses[pm]
                                    evict_v = pm
                        if evict_v is None:
                            # All fast mem are parents we need; evict the one with lowest uses
                            for pm in fast_mem:
                                if remaining_uses[pm] < min_uses:
                                    min_uses = remaining_uses[pm]
                                    evict_v = pm
                        if evict_v is not None:
                            ops.append(['store', evict_v])
                            color[evict_v] = 1
                            fast_mem.remove(evict_v)
                            transfer_count += 1
                        # Now load
                        ops.append(['load', p])
                        color[p] = 2
                        fast_mem.add(p)
                        transfer_count += 1
                else:
                    # Parent is white, shouldn't happen in topological order
                    pass
            
            # Now compute v
            ops.append(['compute', v])
            computed[v] = True
            color[v] = 2
            fast_mem.add(v)
        
        # After processing v, update remaining_uses for its parents
        for p in parents[v]:
            remaining_uses[p] -= 1
        
        # Evict vertices that are no longer needed and memory is over budget
        while len(fast_mem) > fast_memory_budget:
            # Find vertex with lowest remaining_uses to evict
            evict_v = None
            min_uses = float('inf')
            for vm in fast_mem:
                if remaining_uses[vm] < min_uses:
                    min_uses = remaining_uses[vm]
                    evict_v = vm
            if evict_v is not None and remaining_uses[evict_v] <= 0:
                ops.append(['store', evict_v])
                color[evict_v] = 1
                fast_mem.remove(evict_v)
                transfer_count += 1
            elif evict_v is not None:
                # Must evict even if still needed
                ops.append(['store', evict_v])
                color[evict_v] = 1
                fast_mem.remove(evict_v)
                transfer_count += 1
            else:
                break
    
    # Ensure all outputs are blue at the end
    # Outputs should be red or blue; if red, store them
    for v in outputs:
        if color[v] == 2:
            ops.append(['store', v])
            color[v] = 1
            if v in fast_mem:
                fast_mem.remove(v)
            transfer_count += 1
    
    # Verify: all outputs must be blue
    # Count total operations
    if len(ops) > 20000:
        # Truncate if too many (shouldn't happen)
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
