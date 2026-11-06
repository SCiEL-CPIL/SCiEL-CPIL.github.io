import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0
    
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        num_vertices = dag.get('num_vertices', len(parents))
    else:
        parents = list(dag)
        outputs = []
        num_vertices = len(parents)
        
    if not outputs:
        children = [[] for _ in range(num_vertices)]
        for v in range(num_vertices):
            for p in parents[v]:
                children[p].append(v)
        outputs = [v for v in range(num_vertices) if len(children[v]) == 0]
    else:
        children = [[] for _ in range(num_vertices)]
        for v in range(num_vertices):
            for p in parents[v]:
                children[p].append(v)
                
    inputs = [v for v in range(num_vertices) if len(parents[v]) == 0]
    input_set = set(inputs)
    output_set = set(outputs)
    
    in_fast_mem = set(inputs)
    in_slow_mem = set(inputs)
    
    operations = []
    
    def is_needed_later(v, current_v):
        if v in output_set:
            return False # Outputs don't have children, so "later" in topo order is irrelevant for compute deps, but they must be blue at end.
        for c in children[v]:
            if c > current_v:
                return True
        return False

    def ensure_in_fast(v):
        if v in in_fast_mem:
            return
        if v in in_slow_mem:
            operations.append(['load', v])
            in_fast_mem.add(v)
        else:
            # Need to recompute
            for p in parents[v]:
                ensure_in_fast(p)
            operations.append(['compute', v])
            in_fast_mem.add(v)
            # Note: v is now in fast mem. Is it in slow mem? No, unless we store it.

    def evict_vertex(v):
        in_fast_mem.remove(v)
        if v in input_set:
            # Inputs are always in slow mem
            operations.append(['delete', v])
        else:
            # Check if it will be needed later
            # We need to know the current vertex index to check children.
            # This function is called inside the loop, so we can pass current_v or use a nonlocal var.
            # Let's pass current_v.
            pass # Placeholder

    # We need to refactor evict to take current_v
    def evict_vertex_impl(v, current_v):
        in_fast_mem.remove(v)
        if v in input_set:
            operations.append(['delete', v])
        else:
            # If it's in slow mem, we can just delete from fast.
            # If it's NOT in slow mem, we must store it first IF it will be needed later.
            # If it will NOT be needed later, we can just delete it (and it's gone).
            if is_needed_later(v, current_v):
                # Must preserve value
                if v not in in_slow_mem:
                    operations.append(['store', v])
                    in_slow_mem.add(v)
                # Now it's in slow mem, safe to delete from fast
                operations.append(['delete', v])
            else:
                # Not needed later, just delete
                operations.append(['delete', v])
                # Optional: remove from slow mem if it was there?
                # If we remove from slow mem, we save space? No, slow mem is unbounded.
                # But if we delete it, it's not in fast.
                # If it was in slow mem, it stays there. That's fine.

    for v in range(num_vertices):
        if time.time() - start_time > DEADLINE:
            break
            
        # Ensure parents are in fast mem
        for p in parents[v]:
            ensure_in_fast(p)
            
        # Compute v
        operations.append(['compute', v])
        in_fast_mem.add(v)
        
        # Manage memory
        while len(in_fast_mem) > fast_memory_budget:
            # Find candidate to evict
            # Prefer:
            # 1. Not an output (outputs must stay blue? No, they just must finish blue. We can evict and reload? 
            #    "all outputs must finish blue". So at the very end, they must be in fast mem.
            #    During the process, can we evict an output? Yes, as long as we load it back at the end.
            #    But if we evict an output that is not an input, and it's not in slow mem, we lose it.
            #    So we should store it if we evict it.
            
            # Let's score candidates.
            # Lower score = evict first.
            best_candidate = None
            best_score = float('inf')
            
            for u in in_fast_mem:
                # Score calculation
                # 1. If u is an output, high score (keep it)
                if u in output_set:
                    score = 10000
                else:
                    score = 0
                    
                # 2. If u will be needed later, higher score (keep it)
                if is_needed_later(u, v):
                    score += 100
                else:
                    score -= 100 # Prefer evicting unneeded ones
                
                # 3. If u is in slow mem, lower score (safer to evict)
                if u in in_slow_mem:
                    score -= 50
                else:
                    score += 50 # Costly to evict if not in slow mem (requires store)
                
                if score < best_score:
                    best_score = score
                    best_candidate = u
                    
            if best_candidate is None:
                break # Should not happen
            
            evict_vertex_impl(best_candidate, v)
            
    # Ensure outputs are in fast mem at the end
    for v in outputs:
        if v not in in_fast_mem:
            ensure_in_fast(v)
            
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
