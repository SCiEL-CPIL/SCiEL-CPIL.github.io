"""Pebbling solver for communication-minimizing DAG pebbling."""
import time
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (transfers).
    
    Args:
        dag: Dictionary with keys:
            - 'parents': dict mapping vertex -> list of parent vertices
            - 'outputs': set or list of output vertices
            - 'num_vertices': total number of vertices (optional, can infer)
        fast_memory_budget: int, maximum number of red (fast memory) vertices
        
    Returns:
        list of [operation, vertex] where operation is one of
        'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Parse DAG structure
    parents = dict(dag.get('parents', {}))
    outputs = set(dag.get('outputs', set()))
    if isinstance(outputs, list):
        outputs = set(outputs)
    
    # Infer all vertices
    all_vertices = set(parents.keys())
    for v in parents:
        for p in parents[v]:
            all_vertices.add(p)
    for v in outputs:
        all_vertices.add(v)
    
    # Determine inputs: vertices with no parents
    inputs = {v for v in all_vertices if v not in parents or len(parents[v]) == 0}
    
    # Build children map
    children = defaultdict(list)
    for v, par in parents.items():
        for p in par:
            children[p].append(v)
    
    # Compute topological order
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        if v in parents:
            in_degree[v] = len(parents[v])
    
    topo_order = []
    queue = [v for v in all_vertices if in_degree[v] == 0]
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Compute remaining uses: how many times each vertex is needed as input
    # A vertex v is needed for each child c of v
    # Count total children (uses)
    total_uses = {v: len(children[v]) for v in all_vertices}
    
    # Determine which vertices are "blue" initially (inputs)
    # Inputs start blue; all others start in slow memory (need to be loaded)
    
    # Greedy strategy:
    # 1. We maintain a set of vertices in fast memory (red)
    # 2. Initially, inputs are blue (in fast memory? or in slow?)
    #    According to the problem: "Inputs start blue" - blue means in fast memory
    # 3. All outputs must finish blue (in fast memory)
    # 4. Every red value consumes one fast-memory slot
    
    # Strategy: Process vertices in topological order, but use a priority queue
    # based on "urgency" - how soon a vertex will be needed again
    
    # Let's use a different approach: 
    # - We need to compute each vertex once (or recompute if needed)
    # - We want to minimize loads/stores
    # - Key insight: keep vertices in fast memory as long as they'll be needed
    
    # Simpler greedy: 
    # 1. Start with all inputs in fast memory (blue)
    # 2. Process in topological order
    # 3. When we need to compute a vertex, ensure all parents are in fast memory
    # 4. If fast memory is full, evict the vertex with the least future use
    
    # State:
    # - in_fast: set of vertices currently in fast memory (red)
    # - computed: set of vertices that have been computed and are in fast memory
    # - remaining_uses: how many more times each vertex will be needed
    
    # Let me think about this more carefully.
    # 
    # Operations:
    # - load(v): move v from slow to fast memory (counts as transfer)
    # - store(v): move v from fast to slow memory (counts as transfer)
    # - compute(v): compute v from its parents (parents must be in fast memory)
    # - delete(v): remove v from fast memory (frees a slot, no transfer cost)
    #
    # Wait, I need to re-read the problem. The operations are load/store/compute/delete.
    # - load: bring a value into fast memory from slow memory
    # - store: put a value from fast memory back to slow memory
    # - compute: compute a new value (its result goes into fast memory)
    # - delete: free a fast memory slot
    #
    # "Inputs start blue" - blue means in fast memory
    # "all outputs must finish blue" - in fast memory
    # "every red value consumes one fast-memory slot"
    #
    # So the flow is:
    # 1. Inputs are already in fast memory (blue/red)
    # 2. We compute internal vertices, which also go into fast memory
    # 3. We may need to store values back to slow memory to free slots
    # 4. We may need to load values from slow memory if they were stored
    # 5. Outputs must remain in fast memory at the end
    
    # Let me implement a straightforward greedy:
    
    # First, let's count how many times each vertex is used as a parent
    use_count = {v: len(children[v]) for v in all_vertices}
    
    # Greedy pebbling:
    # - Keep track of which vertices are in fast memory
    # - Process in topological order
    # - When computing a vertex, its parents must be in fast memory
    # - If we don't have enough space, evict the vertex with the lowest "value"
    #   (lowest future use count, or highest topological index)
    
    ops = []
    in_fast = set(inputs)  # inputs start in fast memory
    computed = set()  # vertices that have been computed
    stored = set()  # vertices that have been stored to slow memory
    
    # For each vertex, track how many times it will still be needed
    remaining_uses = {v: use_count[v] for v in all_vertices}
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        if v in inputs:
            # Input is already in fast memory
            continue
        
        # Check if all parents are in fast memory
        par = parents.get(v, [])
        missing_parents = [p for p in par if p not in in_fast]
        
        if missing_parents:
            # We need to bring missing parents into fast memory
            # First, free up space if needed
            # Each missing parent needs to be loaded
            for mp in missing_parents:
                if time.time() > deadline:
                    break
                # Free space if needed
                while len(in_fast) >= fast_memory_budget:
                    # Find vertex to evict
                    evict_v = None
                    min_urgency = float('inf')
                    for fv in in_fast:
                        if fv in outputs and fv not in computed:
                            # Don't evict outputs that haven't been stored yet? 
                            # Actually, outputs must finish in fast memory
                            # So we should avoid evicting outputs
                            continue
                        # Don't evict vertices that are parents of uncomputed vertices
                        # Actually, we can evict and reload later
                        urgency = remaining_uses.get(fv, 0)
                        # Prefer to evict vertices with lower remaining uses
                        if urgency < min_urgency:
                            min_urgency = urgency
                            evict_v = fv
                    if evict_v is None:
                        # All remaining are outputs or have high urgency
                        # Just pick any non-output
                        for fv in in_fast:
                            if fv not in outputs:
                                evict_v = fv
                                break
                        if evict_v is None:
                            evict_v = next(iter(in_fast))
                    # Evict
                    ops.append(['store', evict_v])
                    in_fast.discard(evict_v)
                    stored.add(evict_v)
                    
                # Load the missing parent
                ops.append(['load', mp])
                in_fast.add(mp)
        
        # Now all parents should be in fast memory
        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # Decrement remaining uses of parents
        for p in par:
            if p in in_fast:
                remaining_uses[p] = remaining_uses.get(p, 0) - 1
                # If a parent is no longer needed, we could evict it
                if remaining_uses[p] <= 0 and p not in outputs:
                    # Evict to free space
                    ops.append(['store', p])
                    in_fast.discard(p)
                    stored.add(p)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            # Need to load it
            while len(in_fast) >= fast_memory_budget:
                evict_v = None
                min_urgency = float('inf')
                for fv in in_fast:
                    if fv in outputs:
                        continue
                    urgency = remaining_uses.get(fv, 0)
                    if urgency < min_urgency:
                        min_urgency = urgency
                        evict_v = fv
                if evict_v is None:
                    evict_v = next(iter(in_fast))
                ops.append(['store', evict_v])
                in_fast.discard(evict_v)
                stored.add(evict_v)
            ops.append(['load', v])
            in_fast.add(v)
    
    # Limit operations to 20000
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
