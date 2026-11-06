"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s
    
    try:
        return _pebble_improved(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_improved(dag, fast_memory_budget, deadline):
    """
    Improved pebbling strategy with future-cost-aware memory management:
    - Process vertices in topological order
    - Compute future cost for each value (number of future reloads if evicted now)
    - Delete values with lowest future cost when memory is full
    - Proactively evict low-value items to keep memory tight
    - Ensure outputs end up blue (in fast memory)
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
    
    # Compute total reference count for each vertex
    total_refs = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            total_refs[p] += 1
    
    # Initial state: inputs are blue (in fast memory)
    in_fast_memory = set(inputs)
    computed = set(inputs)
    operations = []
    
    budget = fast_memory_budget
    
    # Track remaining references (how many more times each parent will be needed)
    remaining_refs = {v: total_refs[v] for v in vertices}
    
    # For each vertex, precompute the list of children that need it
    # and their topological positions
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    def compute_future_cost(v, current_pos):
        """
        Estimate the future cost of evicting value v at current position.
        This is the number of times v will need to be reloaded if evicted now.
        If v is still needed by future children, each use after eviction costs a load.
        """
        if v in outputs:
            return float('inf')  # Never evict outputs
        
        # Count how many future children (with index > current_pos) need v
        # and group by whether they're contiguous or not
        future_needs = []
        for child in children[v]:
            if vertex_index[child] > current_pos:
                future_needs.append(vertex_index[child])
        
        if not future_needs:
            return 0
        
        # The cost is: if we evict now, we need to load before each future use
        # But if consecutive children need it, we only load once
        # Simple approximation: number of "gaps" in the sequence of future needs
        if len(future_needs) == 1:
            return 1
        
        # Sort future needs
        future_needs.sort()
        
        # Count contiguous groups
        groups = 1
        for i in range(1, len(future_needs)):
            if future_needs[i] != future_needs[i-1] + 1:
                groups += 1
        
        # Each group requires one load if evicted
        return groups
    
    def get_eviction_priority(v, current_pos):
        """
        Get the priority for evicting value v (lower = better to evict).
        We want to evict values with low future cost.
        """
        if v in outputs:
            return float('inf')
        
        # Future cost if evicted now
        fc = compute_future_cost(v, current_pos)
        
        # Also consider: if remaining_refs is 0, it's definitely safe to evict
        if remaining_refs[v] == 0:
            return 0
        
        return fc
    
    def make_room(operations, in_fast_memory, current_pos):
        """Make room in fast memory by evicting lowest-priority values."""
        while len(in_fast_memory) > budget:
            # Find the value with lowest eviction priority
            best_candidate = None
            best_priority = float('inf')
            
            for candidate in in_fast_memory:
                priority = get_eviction_priority(candidate, current_pos)
                if priority < best_priority:
                    best_priority = priority
                    best_candidate = candidate
            
            if best_candidate is not None:
                operations.append(['delete', best_candidate])
                in_fast_memory.discard(best_candidate)
            else:
                break
    
    def proactive_evict(operations, in_fast_memory, current_pos, target_size):
        """Proactively evict values to keep memory at target_size."""
        while len(in_fast_memory) > target_size:
            best_candidate = None
            best_priority = float('inf')
            
            for candidate in in_fast_memory:
                if candidate in outputs:
                    continue
                priority = get_eviction_priority(candidate, current_pos)
                if priority < best_priority:
                    best_priority = priority
                    best_candidate = candidate
            
            if best_candidate is not None:
                operations.append(['delete', best_candidate])
                in_fast_memory.discard(best_candidate)
            else:
                break
    
    # Process vertices in topological order
    for idx, v in enumerate(vertices):
        if v in inputs:
            continue
        
        # Check deadline
        if time.time() > deadline:
            break
        
        current_pos = idx
        
        # Load parents that are not in fast memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Need to make room before loading
                if len(in_fast_memory) >= budget:
                    make_room(operations, in_fast_memory, current_pos)
                
                # Load parent from slow memory
                operations.append(['load', p])
                in_fast_memory.add(p)
        
        # Compute the vertex
        operations.append(['compute', v])
        in_fast_memory.add(v)
        computed.add(v)
        
        # Make room if over budget
        make_room(operations, in_fast_memory, current_pos)
        
        # Decrement reference count for parents
        for p in parents[v]:
            remaining_refs[p] -= 1
        
        # Proactively evict to keep memory tight (target slightly below budget)
        # This helps avoid last-minute expensive evictions
        target = max(0, budget - 1)
        proactive_evict(operations, in_fast_memory, current_pos, target)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast_memory:
            if len(in_fast_memory) >= budget:
                make_room(operations, in_fast_memory, len(vertices))
            operations.append(['load', v])
            in_fast_memory.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
