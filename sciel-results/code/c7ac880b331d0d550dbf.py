"""DAG pebbling: communication-minimizing schedule with bounded fast memory."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule that minimizes total transfers (load+store).
    
    dag: object with:
        - vertices: list of vertex ids (topologically ordered, parents before children)
        - parents: dict mapping vertex -> list of parent vertices
        - inputs: list of input vertex ids (start blue)
        - outputs: list of output vertex ids (must finish blue)
    
    fast_memory_budget: int, max number of red values that can be in fast memory at once
    
    Returns: list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # --- Parse DAG ---
    vertices = list(dag.vertices) if hasattr(dag, 'vertices') else list(getattr(dag, 'get_vertices', lambda: list(range(1))())())
    parents = dict(dag.parents) if hasattr(dag, 'parents') else {}
    inputs = list(dag.inputs) if hasattr(dag, 'inputs') else []
    outputs = set(dag.outputs) if hasattr(dag, 'outputs') else set()
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order is given by vertices list order
    # Input vertices start blue (in fast memory)
    # Output vertices must finish blue (in fast memory)
    
    # --- Determine which vertices are inputs and outputs ---
    input_set = set(inputs)
    output_set = set(outputs)
    
    # --- Compute reference counts and in-degrees ---
    # For each vertex, count how many children need it
    child_count = {v: len(children.get(v, [])) for v in vertices}
    
    # --- Algorithm: Layered pebbling with recomputation ---
    # Strategy: Process vertices in topological order.
    # Each vertex must be computed (or loaded if input) and its result stored.
    # We keep red values in fast memory, bounded by fast_memory_budget.
    # When budget is exceeded, we must store (delete) some red values.
    # If a value is needed later and was deleted, we recompute it (recurring cost).
    
    # Better approach: Use a priority-based eviction.
    # When we need to free a slot, evict the red value whose next use is furthest away.
    
    # First pass: compute next-use for each vertex at each point in time.
    # Since DAG is topologically ordered, we can precompute for each vertex
    # the positions of all its children (where it's needed).
    
    # For each vertex, the set of positions where it's needed = positions of its children
    # (since a vertex's value is consumed by its children).
    
    # Let's use a simple strategy:
    # - Process in topological order
    # - Maintain a dict: red_values[v] = True if v is currently in fast memory
    # - When computing v: if v is an input, it's already blue (loaded). Otherwise, 
    #   compute it (needs all parents to be red or blue... actually parents must be available).
    #   Wait - "compute" operation: the vertex's parents must be in fast memory (red or blue).
    #   Actually, let me re-read: "every red value consumes one fast-memory slot, including newly computed results"
    #   "Inputs start blue, all outputs must finish blue"
    #   Blue = in fast memory, Red = also in fast memory? Or blue=permanent, red=temporary?
    
    # Let me reconsider the model:
    # - Fast memory has capacity = fast_memory_budget slots
    # - A vertex is either "blue" (in fast memory) or not
    # - Actually, I think: blue = in fast memory (counts toward budget), and we can also have values in slow memory (free, but accessing costs a transfer)
    # - load: bring from slow to fast (cost 1 transfer)
    # - store: send from fast to slow (cost 1 transfer)  
    # - compute: produces a new value in fast memory (cost 1 operation, uses 1 slot)
    # - delete: remove from fast memory (cost 1 operation, frees 1 slot)
    # - "red value consumes one fast-memory slot" - so all values in fast memory are "red"
    # - "Inputs start blue" - inputs are initially in fast memory
    # - "outputs must finish blue" - outputs must be in fast memory at the end
    
    # So the model is:
    # - Fast memory: up to fast_memory_budget values
    # - Slow memory: unlimited, but load/store cost transfers
    # - Initially: input vertices are in fast memory (blue), all others are in slow memory (or not computed)
    # - A vertex can only be computed if all its parents are currently in fast memory
    # - After computing, the result goes into fast memory
    # - At the end, all output vertices must be in fast memory
    # - Score = total number of load + store operations (transfers)
    # - compute and delete don't count as transfers? Or do they?
    # "Score mean transfers / naive schedule" - transfers = load + store
    
    # Let me assume: score = number of load + store operations.
    
    # Strategy: Minimize loads and stores.
    # - Don't delete unless necessary (budget exceeded)
    # - When deleting, evict the value whose next needed use is furthest in the future
    # - If a value is deleted and needed again, we must recompute it (which requires its parents)
    
    # Simpler initial approach: 
    # 1. Process in topological order
    # 2. Keep all computed values in fast memory if possible
    # 3. When budget exceeded, evict LRU or furthest-next-use
    # 4. Recompute if needed
    
    # Let me implement a straightforward topological pass with smart eviction.
    
    # Precompute: for each vertex v, the sorted list of child positions (where v is needed)
    pos = {v: i for i, v in enumerate(vertices)}
    vertex_children_sorted = {}
    for v in vertices:
        child_positions = sorted(pos[c] for c in children.get(v, []))
        vertex_children_sorted[v] = child_positions
    
    # Track state
    # in_fast: set of vertices currently in fast memory
    # computed: set of vertices that have been computed at least once
    # For eviction: use a heap of (next_use_position, vertex)
    
    # Next use: for vertex v at current position i, the next position where v is needed
    # (i.e., next child of v after position i)
    
    def get_next_use(v, current_pos):
        """Get the next position where v is needed (as a parent of some child)."""
        uses = vertex_children_sorted.get(v, [])
        for p in uses:
            if p > current_pos:
                return p
        # Also check if v is an output - it needs to be in fast memory at the end
        if v in output_set:
            return len(vertices)  # sentinel: needs to stay until end
        return len(vertices) + 1  # never needed again
    
    # Initial state
    in_fast = set(input_set)
    # Also, any vertex that is both input and output starts in fast memory
    # Non-input, non-output vertices are not in fast memory initially
    # They need to be computed
    
    # Budget check: inputs occupy fast_memory_budget slots
    if len(in_fast) > fast_memory_budget:
        # This shouldn't happen normally, but handle it
        # Evict some inputs that are furthest from being needed
        heap = []
        for v in in_fast:
            nu = get_next_use(v, -1)
            heapq.heappush(heap, (nu, v))
        while len(in_fast) > fast_memory_budget:
            nu, v = heapq.heappop(heap)
            in_fast.discard(v)
    
    # Priority queue for eviction: (next_use, vertex)
    # We'll maintain this as we go
    
    operations = []
    transfers = 0
    
    # Process vertices in topological order
    for i, v in enumerate(vertices):
        if time.time() - start_time > DEADLINE:
            break
        
        # If v is an input, it should already be in fast memory
        if v in input_set:
            if v not in in_fast:
                # Need to load it
                # Check budget
                if len(in_fast) >= fast_memory_budget:
                    # Evict one
                    # Find the vertex in fast memory with furthest next use
                    best_evict = None
                    best_nu = -1
                    for u in in_fast:
                        nu = get_next_use(u, i)
                        if nu > best_nu:
                            best_nu = nu
                            best_evict = u
                    if best_evict is not None:
                        in_fast.discard(best_evict)
                        operations.append(['store', best_evict])
                        transfers += 1
                in_fast.add(v)
                operations.append(['load', v])
                transfers += 1
            continue
        
        # v is not an input, needs to be computed
        # First, ensure all parents are in fast memory
        parent_list = parents.get(v, [])
        for p in parent_list:
            if p not in in_fast:
                # Need to bring p into fast memory
                # Check budget
                if len(in_fast) >= fast_memory_budget:
                    # Evict the vertex with furthest next use (not including v's needs)
                    best_evict = None
                    best_nu = -1
                    for u in in_fast:
                        nu = get_next_use(u, i)
                        if nu > best_nu:
                            best_nu = nu
                            best_evict = u
                    if best_evict is not None:
                        in_fast.discard(best_evict)
                        operations.append(['store', best_evict])
                        transfers += 1
                # Load p from slow memory
                in_fast.add(p)
                operations.append(['load', p])
                transfers += 1
        
        # Now compute v
        # Check budget after computation
        if len(in_fast) >= fast_memory_budget:
            # Need to make room
            # Evict the vertex with furthest next use (excluding v itself if possible)
            best_evict = None
            best_nu = -1
            for u in in_fast:
                if u == v:
                    continue
                nu = get_next_use(u, i)
                if nu > best_nu:
                    best_nu = nu
                    best_evict = u
            if best_evict is not None:
                in_fast.discard(best_evict)
                operations.append(['store', best_evict])
                transfers += 1
        
        in_fast.add(v)
        operations.append(['compute', v])
    
    # Ensure all outputs are in fast memory at the end
    for v in output_set:
        if v not in in_fast:
            if time.time() - start_time > DEADLINE:
                break
            # Need to load v
            if len(in_fast) >= fast_memory_budget:
                # Evict
                best_evict = None
                best_nu = -1
                for u in in_fast:
                    if u not in output_set:
                        nu = len(vertices) + 1  # outputs are most important
                        if nu > best_nu:
                            best_nu = nu
                            best_evict = u
                if best_evict is None:
                    # All are outputs, just evict any non-output... but they're all outputs
                    # This is a problem; just evict the first one
                    for u in list(in_fast):
                        if u != v:
                            best_evict = u
                            break
                if best_evict is not None:
                    in_fast.discard(best_evict)
                    operations.append(['store', best_evict])
                    transfers += 1
            in_fast.add(v)
            operations.append(['load', v])
            transfers += 1
    
    # Final: delete any non-output values that are in fast memory to save... 
    # Actually, delete doesn't count as transfer, so no benefit unless it frees budget
    # But we don't need to free budget at the end. Skip deletes.
    
    # Verify we're within operation limit
    if len(operations) > 20000:
        # Truncate (shouldn't happen for reasonable DAGs)
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
