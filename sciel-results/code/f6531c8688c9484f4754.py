"""Rewritten program for communication-minimizing DAG pebbling using A* search."""
# EVOLVE-BLOCK-START
import heapq
import functools
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Find a minimal-communication schedule for DAG pebbling.
    
    Args:
        dag: A dictionary with keys 'vertices' (list of vertex ids), 'parents' (dict mapping vertex to list of parents),
             'inputs' (list of input vertex ids), 'outputs' (list of output vertex ids).
             Vertices are topologically ordered.
        fast_memory_budget: Maximum number of red (loaded) values in fast memory at any time.
    
    Returns:
        List of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'.
    """
    # Parse the DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order is given; let's verify and create index mapping
    n = len(vertices)
    if n == 0:
        return []
    
    v_index = {v: i for i, v in enumerate(vertices)}
    
    # Convert to bitmask representation
    # State: (mask_loaded, mask_computed) where:
    #   mask_loaded: vertices currently in fast memory (red)
    #   mask_computed: vertices that have been computed and are either loaded or stored
    # Actually, let's think more carefully.
    #
    # A vertex can be in one of three states:
    #   0: Not computed (black, not in memory)
    #   1: Computed and in fast memory (red)
    #   2: Computed and stored on disk (blue, but not in fast memory)
    #
    # For inputs, they start as "blue" meaning available on disk. So they can be loaded.
    # For non-input vertices, they start as not computed.
    #
    # Let's use two bitmasks:
    #   in_memory: bitmask of vertices in fast memory
    #   computed: bitmask of vertices that have been computed at least once
    #
    # A vertex v is:
    #   - In fast memory if bit v is set in in_memory
    #   - Computed if bit v is set in computed
    #   - On disk (available to load) if it's an input OR if it's computed and not in memory
    #
    # Operations:
    #   load v: v must be on disk (input or computed & not in memory), not in memory. Adds to in_memory.
    #   store v: v must be in memory. Removes from in_memory.
    #   compute v: all parents must be in memory. v must not be in memory. Adds to in_memory and computed.
    #   delete v: v must be in memory. Removes from in_memory. (Same as store? Or is delete different?)
    #
    # Wait, the problem says operations are load/store/compute/delete.
    # Let me re-read: "operations load/store/compute/delete"
    # 
    # Typically in pebbling:
    #   - load: bring from disk to memory
    #   - store: write from memory to disk
    #   - compute: perform computation, result goes to memory
    #   - delete: free memory slot without writing to disk
    #
    # For our purposes, store and delete both remove from memory, but store writes to disk.
    # For a vertex to be re-loadable, it must have been stored (not just deleted).
    #
    # So we need to track whether a computed vertex is on disk.
    # Let's use:
    #   in_memory: bitmask of vertices in fast memory
    #   on_disk: bitmask of computed vertices that are on disk (can be loaded)
    #
    # Actually, once a vertex is computed, it can be stored to disk. Once on disk, it stays on disk
    # (we don't delete from disk in this model, I believe). Let me assume that once stored, it's available.
    #
    # State: (in_memory, on_disk)
    # - in_memory & on_disk should be disjoint? No, a vertex can be both in memory and on disk.
    # - Actually, let's say: a vertex is "available" if it's an input OR (computed AND (in_memory OR on_disk))
    #
    # Let me simplify: 
    #   - Inputs are always available on disk.
    #   - A non-input vertex is available if it has been computed and is either in memory or on disk.
    #   - When we compute a vertex, it goes into memory.
    #   - When we store a vertex, it goes to disk (and can be removed from memory via store or delete).
    #   - When we delete a vertex, it's removed from memory but NOT written to disk.
    #
    # So the state needs to track:
    #   in_memory: which vertices are in fast memory
    #   stored: which non-input vertices have been stored to disk (so they can be reloaded)
    #
    # A vertex v is loadable if:
    #   - v is an input, OR
    #   - v is in stored (non-input vertices that have been stored)
    #
    # And v is not currently in memory.
    #
    # A vertex v is computable if:
    #   - All parents of v are in memory
    #   - v is not in memory
    #
    # Goal: all outputs are in memory (blue in the final state? Or just computed?)
    # The problem says "all outputs must finish blue". In pebbling terminology, blue usually means
    # in memory. So all outputs must be in memory at the end.
    
    # Let's define the state as (in_memory_mask, stored_mask)
    # in_memory_mask: bitmask of vertices in fast memory
    # stored_mask: bitmask of vertices that have been stored to disk (only for non-inputs)
    
    # Initial state:
    #   in_memory = 0
    #   stored = 0
    #   All inputs are implicitly available (we don't need to track them in stored)
    
    # Goal state:
    #   All outputs are in memory
    #   We don't care about stored or other in_memory vertices
    
    # Cost: number of operations (load, store, compute, delete all count as 1)
    # We want to minimize total operations.
    
    # Heuristic for A*: 
    #   - For each output not yet in memory, we need at least 1 operation to get it there.
    #   - More precisely, we need to compute or load it.
    #   - A simple admissible heuristic: count of outputs not in memory.
    
    # Let's implement A* search.
    
    # First, let's check if the problem is small enough for exhaustive search.
    # The state space is 2^n * 2^n which is 4^n. For n up to ~20, this might be feasible.
    # For larger n, we may need a different approach.
    
    # Let's try A* with a good heuristic.
    
    # Precompute: for each vertex, its parents and whether it's an input/output
    is_input = [v in inputs for v in vertices]
    is_output = [v in outputs for v in vertices]
    
    # Bitmasks
    all_inputs_mask = 0
    all_outputs_mask = 0
    for i, v in enumerate(vertices):
        if is_input[i]:
            all_inputs_mask |= (1 << i)
        if is_output[i]:
            all_outputs_mask |= (1 << i)
    
    # Parent masks: for each vertex, bitmask of its parents
    parent_masks = [0] * n
    for i, v in enumerate(vertices):
        mask = 0
        for p in parents.get(v, []):
            if p in v_index:
                mask |= (1 << v_index[p])
        parent_masks[i] = mask
    
    # A* search
    # State: (in_memory, stored)
    # Start: (0, 0)
    # Goal: (in_memory & all_outputs_mask) == all_outputs_mask
    
    # We'll use a priority queue: (f, g, in_memory, stored, path)
    # To save memory, we'll store the path as a list of operations.
    
    # For large state spaces, this might not work. Let's add a fallback.
    
    # First, let's try a greedy baseline to get an upper bound, then use A* if the space is small.
    
    # Greedy approach: topological order, load parents, compute, store children when done.
    def greedy_schedule():
        """Generate a simple topological-order schedule."""
        ops = []
        in_mem = set()
        stored = set()
        
        for v in vertices:
            vi = v_index[v]
            # Load parents that aren't in memory
            for p in parents.get(v, []):
                pi = v_index[p]
                if p not in in_mem:
                    if p in inputs or p in stored:
                        ops.append(['load', p])
                        in_mem.add(p)
                    else:
                        # This shouldn't happen in topological order
                        raise ValueError(f"Parent {p} of {v} not available")
            
            # Compute v
            if v not in in_mem:
                ops.append(['compute', v])
                in_mem.add(v)
            
            # After computing, we can store parents that are no longer needed
            # A parent p is no longer needed if all its children are computed
            for p in parents.get(v, []):
                # Check if p is needed by any other uncomputed child
                needed = False
                for c in children[p]:
                    if c not in in_mem and c not in stored:
                        # c is not computed yet, so p might be needed
                        # But we're processing in topological order, so if c comes later, p is needed
                        needed = True
                        break
                if not needed and p in in_mem and p not in outputs:
                    ops.append(['store', p])
                    in_mem.discard(p)
                    stored.add(p)
        
        # Ensure all outputs are in memory
        for v in outputs:
            if v not in in_mem:
                if v in stored or v in inputs:
                    ops.append(['load', v])
                    in_mem.add(v)
        
        return ops
    
    # Let's first try the greedy schedule to see the baseline
    try:
        greedy_ops = greedy_schedule()
        greedy_cost = len(greedy_ops)
    except:
        greedy_cost = float('inf')
    
    # Now try A* if the state space is manageable
    # State space size is at most 4^n. For n > 25, this is too large.
    # Let's try A* for n <= 25, otherwise fall back to greedy.
    
    if n > 25:
        # Fall back to greedy
        return greedy_ops
    
    # A* search
    # State: (in_memory, stored)
    # in_memory and stored are bitmasks
    
    # Heuristic: number of outputs not in memory
    def heuristic(in_memory, stored):
        missing_outputs = all_outputs_mask & ~in_memory
        return bin(missing_outputs).count('1')
    
    # Priority queue: (f, g, in_memory, stored, ops)
    # We'll use a visited set to avoid revisiting states with higher cost
    # visited: dict mapping (in_memory, stored) -> min_g_seen
    
    start_state = (0, 0)
    start_ops = []
    
    pq = [(heuristic(0, 0), 0, 0, 0, start_ops)]
    visited = {}
    visited[(0, 0)] = 0
    
    best_ops = greedy_ops if greedy_ops else None
    best_cost = greedy_cost
    
    max_iterations = 50000  # Limit to avoid timeout
    
    while pq and len(visited) < max_iterations:
        f, g, in_memory, stored, ops = heapq.heappop(pq)
        
        # Check if goal
        if (in_memory & all_outputs_mask) == all_outputs_mask:
            if g < best_cost:
                best_cost = g
                best_ops = list(ops)
                # Since A* with admissible heuristic, first goal is optimal
                break
        
        # Generate successors
        # Possible operations:
        # 1. Load a vertex: v must be (input or stored) and not in memory
        # 2. Store a vertex: v must be in memory and not an input (inputs can't be stored? or can they?)
        #    Actually, storing an input doesn't make sense since inputs are always available.
        #    Let's only store non-input vertices.
        # 3. Compute a vertex: all parents in memory, v not in memory
        # 4. Delete a vertex: v in memory, not an output (we need outputs in memory at end)
        #    Actually, we can delete outputs too, but then we'd need to reload them.
        #    Let's allow deleting any vertex in memory, but it's usually not optimal to delete outputs.
        
        # Load operations
        for i, v in enumerate(vertices):
            if in_memory & (1 << i):
                continue  # Already in memory
            if is_input[i] or (stored & (1 << i)):
                # Can load v
                new_in_memory = in_memory | (1 << i)
                new_stored = stored
                new_g = g + 1
                state_key = (new_in_memory, new_stored)
                if state_key not in visited or new_g < visited[state_key]:
                    visited[state_key] = new_g
                    new_f = new_g + heuristic(new_in_memory, new_stored)
                    new_ops = ops + [['load', v]]
                    heapq.heappush(pq, (new_f, new_g, new_in_memory, new_stored, new_ops))
        
        # Store operations (only for non-inputs in memory)
        for i, v in enumerate(vertices):
            if not (in_memory & (1 << i)):
                continue  # Not in memory
            if is_input[i]:
                continue  # Don't store inputs
            new_in_memory = in_memory & ~(1 << i)
            new_stored = stored | (1 << i)
            new_g = g + 1
            state_key = (new_in_memory, new_stored)
            if state_key not in visited or new_g < visited[state_key]:
                visited[state_key] = new_g
                new_f = new_g + heuristic(new_in_memory, new_stored)
                new_ops = ops + [['store', v]]
                heapq.heappush(pq, (new_f, new_g, new_in_memory, new_stored, new_ops))
        
        # Compute operations
        for i, v in enumerate(vertices):
            if in_memory & (1 << i):
                continue  # Already in memory
            # Check if all parents are in memory
            if (in_memory & parent_masks[i]) == parent_masks[i]:
                new_in_memory = in_memory | (1 << i)
                new_stored = stored
                new_g = g + 1
                state_key = (new_in_memory, new_stored)
                if state_key not in visited or new_g < visited[state_key]:
                    visited[state_key] = new_g
                    new_f = new_g + heuristic(new_in_memory, new_stored)
                    new_ops = ops + [['compute', v]]
                    heapq.heappush(pq, (new_f, new_g, new_in_memory, new_stored, new_ops))
        
        # Delete operations
        for i, v in enumerate(vertices):
            if not (in_memory & (1 << i)):
                continue  # Not in memory
            # We can delete any vertex in memory
            # But deleting an output is probably not useful, so let's skip outputs
            if is_output[i]:
                continue
            new_in_memory = in_memory & ~(1 << i)
            new_stored = stored
            new_g = g + 1
            state_key = (new_in_memory, new_stored)
            if state_key not in visited or new_g < visited[state_key]:
                visited[state_key] = new_g
                new_f = new_g + heuristic(new_in_memory, new_stored)
                new_ops = ops + [['delete', v]]
                heapq.heappush(pq, (new_f, new_g, new_in_memory, new_stored, new_ops))
    
    if best_ops is not None:
        return best_ops
    else:
        # Fallback to greedy
        return greedy_ops

# EVOLVE-BLOCK-END
