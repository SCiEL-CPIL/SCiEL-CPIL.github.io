"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # --- 1. Parse DAG Structure ---
    def get_vertices(d):
        if hasattr(d, 'vertices'):
            return list(d.vertices)
        elif isinstance(d, dict) and 'vertices' in d:
            return list(d['vertices'])
        elif hasattr(d, 'nodes'):
            return list(d.nodes)
        elif isinstance(d, dict) and 'nodes' in d:
            return list(d['nodes'])
        else:
            if hasattr(d, 'parents'):
                parents = d.parents
            elif isinstance(d, dict) and 'parents' in d:
                parents = d['parents']
            else:
                raise ValueError("Cannot determine vertices from dag")
            verts = set()
            for v, ps in parents.items():
                verts.add(v)
                verts.update(ps)
            return list(verts)
    
    def get_parents(d):
        if hasattr(d, 'parents'):
            return dict(d.parents)
        elif isinstance(d, dict) and 'parents' in d:
            return dict(d['parents'])
        elif hasattr(d, 'edges'):
            parents = {}
            for e in d.edges:
                if len(e) == 2:
                    child, parent = e
                    parents.setdefault(child, []).append(parent)
            return parents
        else:
            raise ValueError("Cannot determine parents from dag")
    
    def get_inputs(d):
        if hasattr(d, 'inputs'):
            return list(d.inputs)
        elif isinstance(d, dict) and 'inputs' in d:
            return list(d['inputs'])
        else:
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]
    
    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            parents = get_parents(d)
            vertices = get_vertices(d)
            children = {}
            for v, ps in parents.items():
                for p in ps:
                    children.setdefault(p, []).append(v)
            return [v for v in vertices if not children.get(v, [])]
    
    try:
        vertices = get_vertices(dag)
        parents_map = get_parents(dag)
        inputs = get_inputs(dag)
        outputs = get_outputs(dag)
    except Exception:
        # Fallback to empty or baseline if parsing fails
        return []
    
    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # --- 2. Determine Traversal Order ---
    # We use a DFS-based topological order to maximize locality.
    # Roots (inputs) are the starting points.
    # We visit children in a way that keeps the "stack" of active nodes small.
    # Standard DFS: visit node, then visit all children recursively.
    # To ensure topological validity, we only proceed to a child if all its parents are already visited.
    # However, in a DAG, a node might have multiple parents. 
    # A safer approach for pebbling: 
    # If the graph is a tree, DFS is optimal. 
    # For general DAGs, we can use a "depth-first" heuristic on the topological sort.
    
    # Let's construct a traversal order that prioritizes finishing a branch before starting a new one.
    # We'll do a DFS from inputs.
    
    visited = set()
    order = []
    
    def dfs(v):
        if v in visited:
            return
        visited.add(v)
        order.append(v)
        # Sort children to ensure determinism, maybe by index or name
        # We want to go deep.
        for c in children_map.get(v, []):
            # We cannot simply recurse if c has other parents not yet visited.
            # So we check if all parents of c are visited? No, that would be a standard topo sort.
            # For pebbling, we want to keep the "current path" in memory.
            # If we just do a standard topo sort, we might lose locality.
            # Let's try a modified DFS that only recurses if it's "safe" to keep the current stack.
            # Actually, a simple DFS on the graph structure (ignoring multi-parents for traversal choice, 
            # but respecting them for computation) is a good heuristic.
            # We will visit children in the order they appear.
            dfs(c)
            
    # Start DFS from all inputs
    for inp in inputs:
        dfs(inp)
        
    # If there are nodes not visited (disconnected or complex DAG structure not reachable by simple DFS from inputs due to multi-parents?)
    # Ensure all nodes are in the order. If not, append them in topo order.
    if len(order) < len(vertices):
        # Fallback to topological sort for remaining
        in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            if v not in visited:
                topo_order.append(v)
                visited.add(v)
            for c in children_map.get(v, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        order.extend(topo_order)
        
    # If still not all, just add remaining
    if len(order) < len(vertices):
        for v in vertices:
            if v not in visited:
                order.append(v)
                visited.add(v)
                
    # --- 3. Precompute Last Use ---
    # For each vertex, find the last index in `order` where it is needed as a parent.
    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(order):
        for p in parents_map.get(v, []):
            last_use[p] = i
            
    # Also, outputs must be "used" (stored) at the end. 
    # Actually, the requirement is "all outputs must finish blue". 
    # Blue means in fast memory? Or stored? 
    # "Inputs start blue, all outputs must finish blue". 
    # "every red value consumes one fast-memory slot". 
    # Usually, Blue = in fast memory (or input tape), Red = computed/stored? 
    # The prompt says: "Inputs start blue... every red value consumes one fast-memory slot". 
    # This implies Blue is the "good" state for inputs/outputs? 
    # Wait, "Inputs start blue". If inputs are blue, they are available. 
    # "Outputs must finish blue". 
    # If Blue = in fast memory, then we must keep outputs in fast memory until the end? 
    # Or does "blue" mean "valid/available"? 
    # Let's re-read carefully: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot".
    # This phrasing is slightly ambiguous. Standard pebbling:
    # - Pebbles are placed on vertices.
    # - If a vertex has a pebble, it's "pebbled".
    # - Here, "blue" likely means "in fast memory" (pebbled).
    # - "Red" likely means "in slow memory" or "needs to be loaded".
    # - "Inputs start blue": Inputs are already in fast memory? Or just available?
    #   If inputs are in fast memory, they consume slots.
    # - "Outputs must finish blue": Outputs must be in fast memory at the end.
    # - "Red value consumes one fast-memory slot": This contradicts if Red is slow memory.
    #   Maybe "Red" means "Computed result in fast memory"? 
    #   Let's assume:
    #   - Blue: Input/Output value that is *currently in fast memory*.
    #   - Red: Intermediate computed value *currently in fast memory*.
    #   - Both consume slots.
    #   - Operations:
    #       - load: Move from slow to fast (becomes Blue/Red).
    #       - store: Move from fast to slow (becomes un-pebbled).
    #       - compute: If parents are in fast memory, result is in fast memory.
    #       - delete: Remove from fast memory (if no longer needed? or just free slot?).
    #       
    #   The prompt says: "Inputs start blue". This implies they are in fast memory initially.
    #   "All outputs must finish blue". This implies they must be in fast memory at the end.
    #   "Every red value consumes one fast-memory slot". This implies computed values (Red) also take slots.
    #   So, Fast Memory holds both Blue (inputs/outputs) and Red (intermediates).
    #   
    #   Goal: Minimize total operations (load/store/compute/delete).
    #   Note: `compute` is mandatory for each non-input vertex.
    #   `load` is mandatory for inputs if they were evicted, or for intermediates if evicted and needed again.
    #   `store` is used to free space.
    #   `delete` might be used to free space without going to slow memory? 
    #   Usually in pebbling, you just remove the pebble. Here `store` implies writing back.
    #   If we store an intermediate, we can reload it later. If we delete it, we must recompute.
    #   "Recomputations allowed".
    #   
    #   Strategy:
    #   1. Inputs start in fast memory.
    #   2. We process vertices in `order`.
    #   3. For each vertex `v`:
    #      - If `v` is an input: It's already in fast memory (unless evicted). If evicted, `load` it.
    #      - If `v` is not an input: 
    #         - Ensure all parents are in fast memory. If not, `load` them (from slow memory, assuming we stored them).
    #         - `compute` `v`. `v` is now in fast memory (Red).
    #      - After processing `v`, check if `v` is still needed.
    #         - If `last_use[v] < current_index` (i.e., we just used it for the last time):
    #            - If `v` is an output: It must finish blue. So we CANNOT store/delete it? 
    #              Wait, "finish blue" means in fast memory. So we must keep outputs in fast memory until the very end.
    #            - If `v` is not an output: We can `store` it (move to slow) or `delete` it.
    #              Storing allows reload. Deleting requires recompute.
    #              Reloading is 1 op (load). Recomputing is 1 op (compute) + loads for parents.
    #              Usually, reloading is cheaper than recomputing if parents are also evicted.
    #              But if we store, we use a slot in slow memory (unlimited). 
    #              So `store` is generally preferred over `delete` unless we need to free fast memory immediately and don't want to keep the value around?
    #              Actually, `store` moves it to slow. `delete` removes it.
    #              If we `store`, we can `load` later.
    #              If we `delete`, we must `compute` later.
    #              Since `compute` is 1 op and `load` is 1 op, but `compute` requires parents to be loaded...
    #              So `store` is better if we expect to reuse `v`.
    #              If `last_use` is passed, we won't reuse `v`. So we should free the slot.
    #              `store` vs `delete`: 
    #              If we `store`, we write to slow. If we `delete`, we do nothing to slow.
    #              Does `store` count as a transfer? Yes.
    #              Does `delete` count? Yes.
    #              If we don't need `v` again, `delete` is "cheaper" in terms of future work, but costs 1 op now.
    #              `store` costs 1 op now, and allows `load` (1 op) later.
    #              If we don't need it again, `delete` is better (no future load).
    #              However, if `v` is an output, we MUST keep it blue. So no store/delete for outputs until the end?
    #              Actually, if we store an output, it's no longer blue. We must load it back to make it blue.
    #              So for outputs, we should keep them in fast memory if possible.
    
    # Let's refine the "Last Use" for the purpose of eviction.
    # We want to evict values that are not needed soon.
    
    # --- 4. Simulation ---
    fast_mem = set(inputs) # Inputs start blue (in fast memory)
    # Note: If fast_memory_budget < len(inputs), we have a problem. 
    # But usually budget is sufficient for inputs or we must load/store dynamically.
    # If budget < len(inputs), we must load one, store others? 
    # "Inputs start blue". This implies they are in fast memory. 
    # If budget is too small, the instance is invalid or we must handle it.
    # Let's assume budget >= len(inputs) or we handle overflow by storing inputs immediately?
    # If we store an input, it's no longer blue. We must load it before use.
    
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    # Map vertex to its next use index in `order`
    # We need to know for each vertex, when is the next time it's needed.
    # Since we have `order`, we can precompute `next_use` for each position.
    # Actually, for eviction, we need to know: for each v in fast_mem, when is its next use?
    
    # Precompute uses: list of indices in `order` where v is a parent.
    uses = defaultdict(list)
    for i, v in enumerate(order):
        for p in parents_map.get(v, []):
            uses[p].append(i)
    
    # For each vertex, we can quickly find the next use after index i.
    # Since `order` is a single pass, we can just iterate.
    # But for efficiency, let's use pointers.
    
    # We will iterate through `order`.
    # State: fast_mem (set of vertices currently in fast memory)
    # For each v in fast_mem, we need to know its next use index.
    
    # Let's maintain a heap of (next_use_index, vertex) for fast_mem vertices.
    # If next_use_index is -1 (no more uses), it's a candidate for eviction.
    
    # Initialize heap with inputs.
    # Next use for inputs: first index in `order` where they are a parent.
    # If an input is never a parent, its next use is inf (but it might be an output?).
    
    # Let's define a helper to get next use.
    # Since `uses` is a list of indices, and we process `order` sequentially,
    # we can maintain a pointer `use_ptr[v]` into `uses[v]`.
    
    use_ptr = {v: 0 for v in vertices}
    
    def get_next_use(v):
        ptr = use_ptr[v]
        if ptr < len(uses[v]):
            return uses[v][ptr]
        return float('inf')
    
    # Initialize heap
    heap = []
    for v in fast_mem:
        nu = get_next_use(v)
        heapq.heappush(heap, (nu, v))
        
    # We also need to handle the case where a vertex is in fast_mem but its next use has passed.
    # We will update the heap as we go.
    
    # To avoid stale entries in the heap, we can use a "lazy deletion" approach or just pop until valid.
    # Valid if: v is in fast_mem AND heap_entry_matches_current_next_use.
    
    # Let's simplify: 
    # At each step i (processing vertex v = order[i]):
    # 1. If v is an input:
    #    - If v not in fast_mem:
    #        - Ensure space (evict if needed).
    #        - load v.
    #        - Add to fast_mem.
    #        - Push (next_use, v) to heap.
    #    - Else:
    #        - It's already there. Do nothing.
    # 2. If v is not an input:
    #    - For each parent p of v:
    #        - If p not in fast_mem:
    #            - Ensure space.
    #            - load p.
    #            - Add to fast_mem.
    #            - Push (next_use, p) to heap.
    #    - compute v.
    #    - Add v to fast_mem.
    #    - Push (next_use, v) to heap.
    #    
    # 3. After processing v (or at any point if space is low):
    #    - If len(fast_mem) > fast_memory_budget:
    #        - Evict.
    #        - Pop from heap until we find a valid entry (v in fast_mem, next_use matches).
    #        - Evict the one with largest next_use? 
    #        - Wait, heap is min-heap. We want to evict the one with LARGEST next_use (furthest away).
    #        - So we should use a max-heap or store negative next_use.
    #        - Or, we can just scan fast_mem if it's small. 
    #        - Given budget is likely small (10-100), scanning is O(B). 
    #        - Let's stick to scanning for simplicity and correctness, as B is small.
    
    def ensure_space():
        nonlocal op_count
        while len(fast_mem) >= fast_memory_budget and op_count < MAX_OPS:
            # Find best candidate to evict
            # Candidates: all v in fast_mem
            # Score: 
            # 1. If v is an output, we should NOT evict it if possible, because it must finish blue.
            #    But if we MUST evict, we evict the one with largest next use? 
            #    Actually, if we evict an output, we must load it back later.
            #    If we evict a non-output, we must load or recompute.
            #    Let's prioritize evicting non-outputs.
            #    Among non-outputs, evict the one with the furthest next use.
            
            best_v = None
            best_score = -1
            for v in fast_mem:
                # If v is an output, penalize heavily (don't evict unless necessary)
                penalty = 10000 if v in output_set else 0
                nu = get_next_use(v)
                # If no more uses, score is very high (evict now)
                if nu == float('inf'):
                    score = 100000 - penalty
                else:
                    # Larger nu = higher score (evict later, so prefer to evict if it's the largest? 
                    # No, we want to evict the one we need LEAST soon.
                    # So we want to evict the one with MAX next_use.
                    score = nu - penalty
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v is not None:
                fast_mem.remove(best_v)
                operations.append(['store', best_v])
                op_count += 1
                # If we stored it, it's in slow memory. 
                # If we need it again, we will load it.
                # If we don't need it again, it stays in slow memory (harmless).
            else:
                break # Should not happen if budget > 0
        
    # Main Loop
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
            
        # 1. Handle Input
        if v in input_set:
            if v not in fast_mem:
                ensure_space()
                fast_mem.add(v)
                operations.append(['load', v])
                op_count += 1
            # Advance use pointer? 
            # Wait, `uses` contains indices where v is a PARENT.
            # If v is an input, it is a parent for some children.
            # We don't "use" the input v when we process v itself.
            # We use v when we process its children.
            # So we should NOT advance use_ptr[v] here.
            continue
            
        # 2. Handle Computation
        # Ensure parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                ensure_space()
                fast_mem.add(p)
                operations.append(['load', p])
                op_count += 1
                # Update next use for p? 
                # p is being used now (at index i).
                # So its current use index is i.
                # We need to advance use_ptr[p] so that get_next_use(p) returns the NEXT use.
                # The current use is i. 
                # In our `uses[p]` list, the current use should be at use_ptr[p].
                # Let's verify: 
                # `uses[p]` is a sorted list of indices where p is a parent.
                # When we process v at index i, and p is a parent of v, then i is in `uses[p]`.
                # We assume use_ptr[p] points to the current use.
                # After using it, we advance the pointer.
                
                # Check if use_ptr[p] is valid
                if use_ptr[p] < len(uses[p]) and uses[p][use_ptr[p]] == i:
                    use_ptr[p] += 1
                else:
                    # Fallback: find i in uses[p] and advance
                    # This should be rare if logic is correct
                    pass
            else:
                # p is already in fast memory.
                # It is being used now.
                if use_ptr[p] < len(uses[p]) and uses[p][use_ptr[p]] == i:
                    use_ptr[p] += 1
                    pass # Already in memory, just advance pointer.
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        
        # v is now in fast memory
        fast_mem.add(v)
        
        # If v is an output, it must stay in fast memory.
        # If v is not an output, we might evict it later.
        
        # If v is not an output and has no more uses, we can evict it immediately to save space?
        # Yes, if we don't need it again, keeping it in fast memory wastes a slot.
        # Let's check if it's the last use.
        if v not in output_set:
            if get_next_use(v) == float('inf'):
                # No more uses. Evict immediately.
                fast_mem.remove(v)
                operations.append(['store', v])
                op_count += 1
            else:
                # Keep it, but ensure space if budget exceeded
                ensure_space()
        else:
            # Output: Must stay. Ensure space if budget exceeded.
            # If budget is too small to hold all outputs, we might have to evict some?
            # But they must finish blue. 
            # If we evict, we must reload.
            # Let's just ensure space.
            ensure_space()
            
    # Final Step: Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
        if v not in fast_mem:
            ensure_space()
            fast_mem.add(v)
            operations.append(['load', v])
            op_count += 1
    
    # Any remaining values in fast memory that are not outputs?
    # They don't need to be blue. We can leave them or store them.
    # Storing them costs ops. Leaving them costs nothing (unless verification checks final state strictly?).
    # "All outputs must finish blue". It doesn't say "only outputs".
    # So we can leave intermediates in fast memory.
    
    if op_count > MAX_OPS:
        operations = operations[:MAX_OPS]
        
    return operations

# EVOLVE-BLOCK-END
