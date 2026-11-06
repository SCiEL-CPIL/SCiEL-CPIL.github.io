"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    try:
        return _pebble_optimized(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized pebbling strategy:
    - Correctly uses 'store' for values needed later, 'delete' for values not needed.
    - Evicts based on 'next use' distance to minimize loads.
    """
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Map vertex to index for quick lookup
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build dependents list: for each vertex p, list of indices of vertices that depend on p
    dependents = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            dependents[p].append(v_to_idx[v])
            
    # Sort dependents lists (they should be in topological order already if we iterate vertices in order)
    for p in vertices:
        dependents[p].sort()
        
    # Pointers to next dependent
    ptr = {v: 0 for v in vertices}
    
    # Initial state
    in_fast = set(inputs)
    in_slow = set() # Values stored to slow memory
    operations = []
    
    budget = fast_memory_budget
    
    # Helper to get next use index for a vertex
    def get_next_use(v):
        if ptr[v] < len(dependents[v]):
            return dependents[v][ptr[v]]
        return None

    # Process vertices in topological order
    for i, v in enumerate(vertices):
        if time.time() > deadline:
            break
            
        if v in inputs:
            # Inputs are already in fast memory.
            # We need to update pointers for inputs?
            # Inputs don't have parents, but they are dependents of nothing.
            # However, they are PARENTS of other vertices.
            # The pointer update happens when the DEPENDENT is processed.
            # So we just skip computation.
            continue
        
        # Load parents if not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if p in in_slow:
                    operations.append(['load', p])
                    in_slow.remove(p)
                    in_fast.add(p)
                else:
                    # This should not happen if we correctly store/delete
                    # If it happens, it's a bug. We might need to recompute, 
                    # but for now, assume it's in slow (error handling)
                    # For safety, if not in slow, we can't load. 
                    # But with correct store/delete, it must be in slow.
                    pass
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        
        # Update pointers for parents of v
        # v is a dependent of its parents
        for p in parents[v]:
            # Advance ptr[p] past the current index i
            while ptr[p] < len(dependents[p]) and dependents[p][ptr[p]] <= i:
                ptr[p] += 1
        
        # Eviction phase
        while len(in_fast) > budget:
            # Find best candidate to evict
            best_candidate = None
            best_cost = float('inf') # Lower cost is better to evict
            best_action = None
            
            for c in in_fast:
                next_use = get_next_use(c)
                
                # Calculate the cost of evicting this vertex
                # If we evict and later need it, what's the cost?
                
                if next_use is None:
                    # Not needed again
                    if c in outputs:
                        # Must keep in fast until end, but if we must evict, store it
                        # Cost: 1 store now + 1 load at end = 2 transfers
                        # But we avoid a load if we keep it
                        cost = 2.0
                        action = 'store'
                    else:
                        # Can delete, no future cost
                        cost = 0.0
                        action = 'delete'
                else:
                    # Needed again at index next_use
                    distance = next_use - i
                    
                    # Cost depends on whether we can recompute or must load
                    if c in inputs:
                        # Input: must load from slow, cost = 1 load
                        # But if we store it, we pay 1 store now + 1 load later
                        cost = 1.0 + 1.0  # store + future load
                        action = 'store'
                    else:
                        # Computed value: can recompute if parents available
                        # Cost of recomputation: need all parents in fast
                        # If parents are still in fast, recompute cost = 1 compute
                        # If parents not in fast, need to load them too
                        parent_cost = 0
                        for p in parents[c]:
                            if p not in in_fast:
                                parent_cost += 1  # need to load parent
                        # Total recompute cost: load parents + 1 compute
                        recompute_cost = parent_cost + 1
                        # Store cost: 1 store now + 1 load later
                        store_cost = 1.0 + 1.0
                        
                        # Choose cheaper option
                        if recompute_cost < store_cost:
                            cost = recompute_cost
                            action = 'delete'  # will recompute later
                        else:
                            cost = store_cost
                            action = 'store'
                        
                        # Add a small penalty for values used very soon
                        if distance <= 2:
                            cost += 0.5  # discourage evicting soon-to-be-used values
                    
                    # Also consider: if this is an output, prefer to keep it
                    if c in outputs:
                        cost += 1.0  # penalty for evicting outputs
            
            if best_candidate is None:
                # Should not happen if budget >= 1
                break
            
            # Find the candidate with minimum cost
            for c in in_fast:
                next_use = get_next_use(c)
                
                if next_use is None:
                    if c in outputs:
                        cost = 2.0
                        action = 'store'
                    else:
                        cost = 0.0
                        action = 'delete'
                else:
                    distance = next_use - i
                    if c in inputs:
                        cost = 2.0
                        action = 'store'
                    else:
                        parent_cost = 0
                        for p in parents[c]:
                            if p not in in_fast:
                                parent_cost += 1
                        recompute_cost = parent_cost + 1
                        store_cost = 2.0
                        if recompute_cost < store_cost:
                            cost = recompute_cost
                            action = 'delete'
                        else:
                            cost = store_cost
                            action = 'store'
                        if distance <= 2:
                            cost += 0.5
                    if c in outputs:
                        cost += 1.0
                
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = c
                    best_action = action
            
            # Execute eviction
            if best_action == 'delete':
                operations.append(['delete', best_candidate])
                in_fast.remove(best_candidate)
                if best_candidate in in_slow:
                    in_slow.remove(best_candidate)
            else: # store
                operations.append(['store', best_candidate])
                in_fast.remove(best_candidate)
                in_slow.add(best_candidate)
                
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                operations.append(['load', v])
                in_slow.remove(v)
                in_fast.add(v)
            else:
                # Should not happen
                pass
            # Make room if needed
            while len(in_fast) > budget:
                deleted = False
                for candidate in list(in_fast):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast.remove(candidate)
                        if candidate in in_slow:
                            in_slow.remove(candidate)
                        deleted = True
                        break
                if not deleted:
                    break
    
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
# EVOLVE-BLOCK-END
