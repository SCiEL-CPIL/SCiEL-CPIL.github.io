"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication.
    
    Args:
        dag: dict with keys 'parents' (list of lists, parents for each vertex),
             'outputs' (list of output vertex indices)
        fast_memory_budget: int, max number of red (in-memory) values
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Parse DAG structure
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    n = len(parents)
    
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Determine relevant vertices (on path to an output)
    relevant = set(outputs)
    changed = True
    while changed:
        changed = False
        for v in list(relevant):
            for p in parents[v]:
                if p not in relevant:
                    relevant.add(p)
                    changed = True
    
    # Count how many times each vertex is needed (number of children that are relevant)
    needed_count = [0] * n
    for v in relevant:
        for p in parents[v]:
            needed_count[p] += 1
    
    # For each vertex, compute the next time it will be needed (in topological order)
    # This helps with eviction decisions
    # next_use[v] = the smallest index of a child of v that is relevant and not yet computed
    # We'll update this as we go
    
    red = set()  # vertices currently in fast memory
    operations = []
    computed = [False] * n
    
    # Track the next child that needs each vertex
    # For each vertex, maintain a queue of children that need it
    pending_children = defaultdict(deque)
    for v in relevant:
        for p in parents[v]:
            if p in relevant:
                pending_children[p].append(v)
    
    # Process in topological order
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break
        
        if v not in relevant:
            continue
        
        # Check if v is already red
        if v in red:
            computed[v] = True
            # Mark this child as done for all parents
            for p in parents[v]:
                if p in relevant and pending_children[p]:
                    if pending_children[p][0] == v:
                        pending_children[p].popleft()
            continue
        
        # Need to compute v
        # Ensure all relevant parents are red
        for p in parents[v]:
            if p not in relevant:
                continue
            if p not in red:
                # Need to load p
                if len(red) >= fast_memory_budget:
                    # Need to evict something
                    evicted = False
                    # Prefer evicting vertices with no more pending uses
                    for r in list(red):
                        if r != v and not pending_children[r]:
                            operations.append(['store', r])
                            red.discard(r)
                            evicted = True
                            break
                    if not evicted:
                        # Evict a vertex whose next use is furthest away
                        best_r = None
                        best_dist = -1
                        for r in list(red):
                            if r != v and pending_children[r]:
                                next_child = pending_children[r][0]
                                if next_child > best_dist:
                                    best_dist = next_child
                                    best_r = r
                        if best_r is not None:
                            operations.append(['store', best_r])
                            red.discard(best_r)
                            evicted = True
                    if not evicted and red:
                        r = next(iter(red))
                        if r != v:
                            operations.append(['store', r])
                            red.discard(r)
                
                operations.append(['load', p])
                red.add(p)
        
        # Compute v
        operations.append(['compute', v])
        red.add(v)
        computed[v] = True
        
        # Mark this child as done for all parents
        for p in parents[v]:
            if p in relevant and pending_children[p]:
                if pending_children[p][0] == v:
                    pending_children[p].popleft()
        
        # Evict if over budget
        while len(red) > fast_memory_budget:
            evicted = False
            for r in list(red):
                if r != v and not pending_children[r]:
                    operations.append(['store', r])
                    red.discard(r)
                    evicted = True
                    break
            if not evicted:
                best_r = None
                best_dist = -1
                for r in list(red):
                    if r != v and pending_children[r]:
                        next_child = pending_children[r][0]
                        if next_child > best_dist:
                            best_dist = next_child
                            best_r = r
                if best_r is not None:
                    operations.append(['store', best_r])
                    red.discard(best_r)
                    evicted = True
            if not evicted and red:
                for r in list(red):
                    if r != v:
                        operations.append(['store', r])
                        red.discard(r)
                        evicted = True
                        break
            if not evicted:
                break
    
    # At the end, all outputs must be blue
    for v in sorted(outputs):
        if v in red:
            operations.append(['store', v])
            red.discard(v)
    
    # Ensure we don't exceed operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
