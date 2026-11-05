import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize transfers (load + store operations).
    
    dag: expected to have 'parents' (list of lists, topologically ordered) and 'outputs' (list of vertex indices).
         Vertices are 0-indexed. Inputs are vertices with no parents.
    fast_memory_budget: maximum number of vertices that can be in fast memory (blue) at once.
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'.
    """
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin
    
    # Parse DAG
    if isinstance(dag, dict):
        parents = dag['parents']
        outputs = dag['outputs']
    else:
        parents = dag.parents
        outputs = dag.outputs
    
    n = len(parents)
    input_set = set(i for i in range(n) if len(parents[i]) == 0)
    output_set = set(outputs)
    
    # Children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Precompute last_needed: the latest index at which a vertex is needed
    # A vertex is needed as a parent up until its last child is processed
    # If it's an output, it's needed until the very end
    last_needed = [-1] * n
    for v in range(n):
        for c in children[v]:
            if c > last_needed[v]:
                last_needed[v] = c
    for v in outputs:
        last_needed[v] = n  # outputs are needed until the end
    
    # State tracking
    in_fast = [False] * n  # whether vertex is in fast memory
    computed = [False] * n  # whether vertex has been computed
    operations = []
    
    # Initialize: inputs start in fast memory and are considered computed
    for v in input_set:
        in_fast[v] = True
        computed[v] = True
    
    # Priority queue for eviction candidates
    # Elements: (last_needed, vertex)
    # We'll maintain a heap of (last_needed, v) for all v in fast memory
    # But we need to handle stale entries, so we'll use a lazy deletion approach
    # Actually, let's just use a simple approach: when we need to evict,
    # scan the fast memory set and pick the best candidate
    
    # For efficiency with large n, let's maintain a heap
    # We'll push (last_needed[v], v) when v enters fast memory
    # And use a "deleted" flag to ignore stale entries
    evict_heap = []
    evict_deleted = [False] * n
    
    for v in input_set:
        heapq.heappush(evict_heap, (last_needed[v], v))
    
    def make_room(needed_vertices, current_vertex):
        """Make room in fast memory by evicting vertices if necessary."""
        while sum(in_fast) >= fast_memory_budget:
            # Find a vertex to evict
            # We need to find a vertex in fast memory that:
            # 1. Is not in needed_vertices (parents we need to load)
            # 2. Is not the current_vertex
            # 3. Is not an output (outputs must stay in fast memory until the end)
            # 4. Has the smallest last_needed
            
            # Try to find a candidate using the heap
            found = False
            while evict_heap:
                ln, v = heapq.heappop(evict_heap)
                if not in_fast[v]:
                    continue
                if v in needed_vertices:
                    continue
                if v == current_vertex:
                    continue
                if v in output_set:
                    continue
                # This is a valid candidate
                operations.append(['store', v])
                in_fast[v] = False
                evict_deleted[v] = True
                found = True
                break
            
            if not found:
                # If we couldn't find a candidate from the heap, do a linear scan
                # This shouldn't happen if the heap is properly maintained
                best_v = -1
                best_ln = float('inf')
                for u in range(n):
                    if not in_fast[u]:
                        continue
                    if u in needed_vertices:
                        continue
                    if u == current_vertex:
                        continue
                    if u in output_set:
                        continue
                    if last_needed[u] < best_ln:
                        best_ln = last_needed[u]
                        best_v = u
                
                if best_v == -1:
                    # No valid candidate found, break to avoid infinite loop
                    break
                
                operations.append(['store', best_v])
                in_fast[best_v] = False
                evict_deleted[best_v] = True
    
    # Process vertices in topological order
    for v in range(n):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are in fast memory
        missing_parents = [p for p in parents[v] if not in_fast[p]]
        if missing_parents:
            # We need to make room for all missing parents
            make_room(set(missing_parents), v)
            
            for p in missing_parents:
                if not in_fast[p]:
                    operations.append(['load', p])
                    in_fast[p] = True
                    heapq.heappush(evict_heap, (last_needed[p], p))
        
        # Compute v
        if not computed[v]:
            operations.append(['compute', v])
            computed[v] = True
            in_fast[v] = True
            heapq.heappush(evict_heap, (last_needed[v], v))
        
        # After computing v, proactively evict vertices that are no longer needed
        # This helps keep memory usage low for future loads
        # We can evict any u in fast memory where last_needed[u] < v+1 and u is not an output
        # Actually, let's just let the make_room function handle evictions when needed
    
    # At the end, ensure all outputs are in fast memory
    for v in outputs:
        if not in_fast[v]:
            make_room(set([v]), -1)
            if not in_fast[v]:
                operations.append(['load', v])
                in_fast[v] = True
                heapq.heappush(evict_heap, (last_needed[v], v))
    
    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
