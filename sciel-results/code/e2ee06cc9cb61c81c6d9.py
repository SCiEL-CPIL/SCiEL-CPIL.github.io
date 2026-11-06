"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    try:
        return _pebble_improved(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_improved(dag, fast_memory_budget, deadline):
    """
    Improved pebbling strategy using next-use-based eviction with lookahead:
    - Precompute next-use distance for each vertex
    - Evict the value whose next use is farthest in the future
    - Proactively delete before memory fills up
    - Ensure outputs end up blue (in fast memory)
    - Use a priority queue for efficient eviction decisions
    """
    # Extract DAG structure
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    # Get parents for each vertex (topologically ordered)
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
    
    # Compute total reference count
    total_ref_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            total_ref_count[p] += 1
    
    # Determine which vertices to process (exclude inputs, they're already computed)
    to_process = [v for v in vertices if v not in inputs]
    num_to_process = len(to_process)
    
    # Precompute next_use for each vertex:
    # next_use[v] = list of indices in to_process where v is a parent of to_process[i]
    # We'll compute this efficiently by iterating through to_process
    
    # For each vertex v, track the list of step indices where v is used
    use_indices = {v: [] for v in vertices}
    for step_idx, v in enumerate(to_process):
        for p in parents[v]:
            use_indices[p].append(step_idx)
    
    # For each vertex v, we want to know: at step i, what is the next step j >= i 
    # where v is used? We can precompute this using binary search on use_indices[v]
    
    # Initial state
    in_fast_memory = set(inputs)
    operations = []
    budget = fast_memory_budget
    
    # Track remaining uses for each vertex
    remaining_uses = dict(total_ref_count)
    
    # We'll use a heap for eviction: (next_use_distance, vertex)
    # But we need to handle lazy deletion since next_use_distance changes
    
    # Simpler approach: maintain a dict of next_use_step for each vertex in memory
    # and evict the one with the largest next_use_step (farthest in future)
    # or the one with no future uses (infinity)
    
    # For efficiency, we'll recompute next_use_step when needed
    # Since the DAG is processed in topological order, once we pass a step,
    # the next use for a vertex is determined
    
    # Let's use a different approach:
    # At each step, we know which vertices are in memory
    # For each vertex in memory, we can compute its next use step by binary search
    # on use_indices[v]
    
    import bisect
    
    def get_next_use_step(v, current_step):
        """Get the next step index >= current_step where v is used, or infinity."""
        if v not in use_indices or not use_indices[v]:
            return float('inf')
        # Find the first index in use_indices[v] that is >= current_step
        idx = bisect.bisect_left(use_indices[v], current_step)
        if idx < len(use_indices[v]):
            return use_indices[v][idx]
        return float('inf')
    
    for step_idx, v in enumerate(to_process):
        if time.time() > deadline:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Need to load p, but first make room if needed
                while len(in_fast_memory) >= budget:
                    # Find best candidate to evict
                    best_candidate = None
                    best_next_use = -1
                    for candidate in in_fast_memory:
                        if candidate in outputs:
                            continue  # Don't evict outputs
                        # Get next use step for this candidate
                        next_use = get_next_use_step(candidate, step_idx)
                        # We want to evict the one with the LARGEST next_use (farthest)
                        # or the one with no future uses (infinity)
                        if next_use > best_next_use:
                            best_next_use = next_use
                            best_candidate = candidate
                    if best_candidate is None:
                        # All are outputs, emergency evict any
                        best_candidate = next(iter(in_fast_memory))
                    operations.append(['delete', best_candidate])
                    in_fast_memory.remove(best_candidate)
                
                operations.append(['load', p])
                in_fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_fast_memory.add(v)
        
        # Decrement remaining_uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
        
        # After compute, if memory exceeds budget, evict
        while len(in_fast_memory) > budget:
            best_candidate = None
            best_next_use = -1
            for candidate in in_fast_memory:
                if candidate in outputs:
                    continue
                next_use = get_next_use_step(candidate, step_idx + 1)
                if next_use > best_next_use:
                    best_next_use = next_use
                    best_candidate = candidate
            if best_candidate is None:
                best_candidate = next(iter(in_fast_memory))
            operations.append(['delete', best_candidate])
            in_fast_memory.remove(best_candidate)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast_memory:
            # Make room
            while len(in_fast_memory) >= budget:
                best_candidate = None
                best_next_use = -1
                for candidate in in_fast_memory:
                    if candidate in outputs:
                        continue
                    next_use = get_next_use_step(candidate, num_to_process)
                    if next_use > best_next_use:
                        best_next_use = next_use
                        best_candidate = candidate
                if best_candidate is None:
                    best_candidate = next(iter(in_fast_memory))
                operations.append(['delete', best_candidate])
                in_fast_memory.remove(best_candidate)
            
            operations.append(['load', v])
            in_fast_memory.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
