import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Extract DAG structure
    nodes = dag.get('nodes', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    if not nodes:
        return []
        
    # Identify inputs (nodes with no parents)
    inputs = [n for n in nodes if not parents.get(n, [])]
    
    # Build children map
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    queue = [n for n in nodes if in_degree[n] == 0]
    topo_order = []
    while queue:
        n = queue.pop(0)
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(nodes):
        # Cycle detected, return empty or partial? Task implies DAG.
        return []

    # Reference counting for "lazy" eviction
    # total_children[n] is the number of children n has.
    # computed_children[n] tracks how many of n's children have been processed.
    total_children = {n: len(children[n]) for n in nodes}
    computed_children = {n: 0 for n in nodes}
    
    fast_mem = set()
    ops = []
    K = fast_memory_budget
    
    for n in topo_order:
        # Ensure parents are in fast memory
        for p in parents.get(n, []):
            if p not in fast_mem:
                # Need to make space if full
                while len(fast_mem) >= K:
                    # Find eviction candidate
                    # Priority 1: Non-output with 0 remaining children (can delete, cost 0)
                    # Priority 2: Non-output with >0 remaining children (must store, cost 1)
                    # Priority 3: Output (cannot evict? or must store? Outputs must be Blue.
                    #             If in Fast, they are Red. We must Store them to make them Blue.
                    #             But we stored them immediately when computed, so they shouldn't be here.
                    #             If they are, it's a bug in previous logic.
                    
                    evict_candidate = None
                    # Check for "dead" non-outputs first
                    for m in fast_mem:
                        if m not in outputs and computed_children[m] == total_children[m]:
                            evict_candidate = m
                            break
                            
                    if evict_candidate is None:
                        # Check for any non-output
                        for m in fast_mem:
                            if m not in outputs:
                                evict_candidate = m
                                break
                                
                    if evict_candidate is None:
                        # Only outputs in fast mem?
                        # This should not happen if we store outputs immediately.
                        # If it does, we are stuck.
                        break
                    
                    if evict_candidate in outputs:
                        # Should not happen, but if it does, store it
                        ops.append(['store', evict_candidate])
                        ops.append(['delete', evict_candidate])
                        fast_mem.remove(evict_candidate)
                    else:
                        if computed_children[evict_candidate] == total_children[evict_candidate]:
                            # Can just delete
                            ops.append(['delete', evict_candidate])
                        else:
                            # Must store to preserve value for future children
                            ops.append(['store', evict_candidate])
                            ops.append(['delete', evict_candidate])
                        fast_mem.remove(evict_candidate)
                
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
            
        # Compute n
        ops.append(['compute', n])
        fast_mem.add(n)
        
        # Update reference counts for parents
        for p in parents.get(n, []):
            computed_children[p] += 1
            
        # If n is an output, it must finish Blue.
        # Store it now to move to Blue, then delete from Fast to free space.
        if n in outputs:
            ops.append(['store', n])
            ops.append(['delete', n])
            fast_mem.remove(n)
            
    # End of schedule.
    # Any remaining nodes in fast_mem that are outputs?
    # We stored them immediately, so they shouldn't be here.
    # Any remaining non-outputs?
    # They don't need to be Blue. We can delete them to be clean, or leave them.
    # Leaving them is fine as long as outputs are Blue.
    # But if we leave them, they are Red.
    # The requirement is only "all outputs must finish blue".
    # So we can stop here.
    
    return ops[:20000]
