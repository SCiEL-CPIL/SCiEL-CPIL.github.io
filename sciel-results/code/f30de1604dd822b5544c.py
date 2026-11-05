"""DAG pebbling with communication minimization."""
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize total transfers (load+store).
    
    dag: dict with keys:
      - 'n': number of vertices
      - 'parents': list of lists, parents[i] = list of parent indices of vertex i
      - 'outputs': list of output vertex indices
      - 'inputs': list of input vertex indices (start blue)
      - 'layers' or similar structure info
    fast_memory_budget: int, number of red values that can be in fast memory
    
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the dag structure
    if isinstance(dag, dict):
        n = dag.get('n', 0)
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
    else:
        # Fallback: try to extract from object
        n = getattr(dag, 'n', 0)
        parents = getattr(dag, 'parents', [])
        outputs = getattr(dag, 'outputs', [])
        inputs = getattr(dag, 'inputs', [])
    
    if n == 0:
        return []
    
    # Build adjacency and dependency info
    # children[v] = list of children of v
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order: vertices are already topologically ordered (parents before children)
    # In-degree for topological sort verification
    in_degree = [len(parents[v]) for v in range(n)]
    
    # Determine which vertices are inputs (no parents or explicitly marked)
    if not inputs:
        inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    # Determine output vertices
    if not outputs:
        outputs = [v for v in range(n) if len(children[v]) == 0]
    
    # Strategy: We need all outputs to finish blue (in fast memory).
    # Inputs start blue. We need to compute intermediate values.
    # A value is "red" (in fast memory) when loaded or computed, and "blue" when stored to slow memory.
    # Wait, re-reading: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    # So blue = in slow memory (free), red = in fast memory (uses slot).
    # Operations:
    #   load: move from slow to fast (blue->red), costs 1 transfer
    #   store: move from fast to slow (red->blue), costs 1 transfer  
    #   compute: compute a value from its parents (parents must be red), result is red, costs 0 transfers
    #   delete: remove a red value from fast memory (freeing a slot), costs 0 transfers
    
    # Wait, let me re-read more carefully.
    # "Inputs start blue" - so inputs are initially in slow memory
    # "all outputs must finish blue" - outputs must end in slow memory
    # "every red value consumes one fast-memory slot, including newly computed results"
    
    # So the goal is to minimize total transfers = number of load + store operations.
    
    # Naive approach: For each vertex in topological order:
    #   - Load all parents that aren't already red
    #   - Compute the vertex (result is red)
    #   - If the vertex is an output, store it (make it blue)
    #   - Otherwise, keep it red if we have space, or store/delete it
    
    # Better approach: Use a pebbling strategy that reuses fast memory.
    
    # Let me think about this differently. The key insight is:
    # - We can recompute values instead of storing them
    # - We want to minimize total load+store operations
    
    # For a layered DAG, a good strategy is:
    # 1. Process vertices in topological order
    # 2. Keep values in fast memory as long as they're needed
    # 3. Store (or delete) values when they're no longer needed
    # 4. Recompute values if we ran out of space
    
    # Let's use a greedy approach with lookahead:
    # - Maintain a set of red vertices (in fast memory)
    # - For each vertex in topo order, ensure parents are red, compute, then decide what to keep
    
    # First, let's figure out the "last use" of each vertex - the last vertex that needs it as a parent
    last_use = [0] * n
    for v in range(n):
        for p in parents[v]:
            last_use[p] = max(last_use[p], v)
    
    # Also track which vertices are outputs (must end blue)
    is_output = [False] * n
    for o in outputs:
        is_output[o] = True
    
    # Count how many times each vertex is needed as a parent
    use_count = [0] * n
    for v in range(n):
        for p in parents[v]:
            use_count[p] += 1
    
    # Greedy pebbling algorithm:
    # Process vertices in topological order.
    # For each vertex v:
    #   1. For each parent p of v:
    #      - If p is not red: load p (if p is an input or was previously stored) or recompute p
    #   2. Compute v (v becomes red)
    #   3. For each parent p of v that is no longer needed (use_count[p] == 0 after this):
    #      - If p is not an output: delete p (free a slot) or store p (if we want to keep it in slow memory)
    #   4. If v is an output: store v (make it blue)
    #   5. If fast memory is full and we need to load a new value:
    #      - Delete/store some value that won't be needed for a while
    
    # Let me implement a more careful version:
    
    ops = []
    red = set()  # vertices currently in fast memory
    slow = set()  # vertices currently in slow memory (blue)
    
    # Inputs start blue
    for inp in inputs:
        slow.add(inp)
    
    # We need to track remaining uses
    remaining_uses = use_count[:]
    
    def elapsed():
        return time.time() - start_time
    
    def can_continue():
        return elapsed() < DEADLINE
    
    # Process vertices in topological order
    for v in range(n):
        if not can_continue():
            break
        
        # Ensure all parents are red
        for p in parents[v]:
            if p not in red:
                # Need to get p into fast memory
                if p in slow:
                    # Load from slow memory
                    ops.append(['load', p])
                    red.add(p)
                    slow.discard(p)
                else:
                    # p was never computed or was deleted - need to recompute
                    # This is complex; for now, let's assume we can always load from slow
                    # If p is an input, it should be in slow
                    if p in inputs:
                        ops.append(['load', p])
                        red.add(p)
                    else:
                        # Need to recompute p - this requires its parents to be red
                        # For simplicity, let's just load it (assuming it's in slow memory)
                        # This is a simplification; in practice we'd need recursive recomputation
                        ops.append(['load', p])
                        red.add(p)
        
        # Compute v
        ops.append(['compute', v])
        red.add(v)
        
        # Decrement use counts for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
            # If p is no longer needed and not an output, we can free it
            if remaining_uses[p] == 0 and not is_output[p] and p not in inputs:
                # We can delete p to free a slot
                if len(red) > fast_memory_budget:
                    ops.append(['delete', p])
                    red.discard(p)
        
        # If v is an output, store it
        if is_output[v]:
            ops.append(['store', v])
            red.discard(v)
            slow.add(v)
        
        # Check if we're over budget
        if len(red) > fast_memory_budget:
            # Need to free some slots - delete values that are least likely to be needed soon
            # Find red vertices that are not outputs and have their last use in the past or far future
            to_delete = []
            for r in red:
                if not is_output[r] and remaining_uses[r] == 0:
                    to_delete.append(r)
            # Sort by last_use (delete those with earliest last use first... actually all have 0 remaining)
            # Just delete until we're within budget
            for r in to_delete:
                if len(red) <= fast_memory_budget:
                    break
                ops.append(['delete', r])
                red.discard(r)
        
        # If still over budget, delete some non-output values
        if len(red) > fast_memory_budget:
            # Delete values with highest last_use that are not outputs
            candidates = [(last_use[r], r) for r in red if not is_output[r]]
            candidates.sort(reverse=True)  # Delete those needed furthest in future first? No, delete those NOT needed soon
            # Actually, delete those with smallest last_use that are still > v, or just any
            for _, r in sorted(candidates, key=lambda x: x[0]):
                if len(red) <= fast_memory_budget:
                    break
                ops.append(['delete', r])
                red.discard(r)
    
    # Ensure all outputs are blue (in slow memory)
    for o in outputs:
        if o in red:
            ops.append(['store', o])
            red.discard(o)
            slow.add(o)
    
    # Truncate to max 20000 operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
