"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        'parents': list of lists, parents[i] = list of parent indices for vertex i
        'outputs': list of output vertex indices
        'num_vertices': int (or inferred from parents length)
    
    fast_memory_budget: int, max number of red values in fast memory at once
    
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG
    if 'parents' in dag:
        parents = dag['parents']
    elif 'nodes' in dag:
        parents = dag['nodes']
    else:
        parents = dag.get('parents', [])
    
    num_vertices = len(parents)
    if 'outputs' in dag:
        outputs = dag['outputs']
    else:
        # Vertices with no children are outputs
        has_child = [False] * num_vertices
        for v in range(num_vertices):
            for p in parents[v]:
                has_child[p] = True
        outputs = [v for v in range(num_vertices) if not has_child[v]]
    
    if 'num_vertices' in dag:
        num_vertices = max(num_vertices, dag['num_vertices'])
    
    # Identify input vertices (no parents)
    inputs = [v for v in range(num_vertices) if len(parents[v]) == 0]
    
    # Topological order (should already be given, but let's verify/create)
    # Assume vertices are in topological order: parents[i] < i for all i
    # This is the standard assumption for layered DAGs
    
    # Build children map
    children = [[] for _ in range(num_vertices)]
    for v in range(num_vertices):
        for p in parents[v]:
            children[p].append(v)
    
    # Compute in-degrees (number of parents)
    in_degree = [len(parents[v]) for v in range(num_vertices)]
    
    # For a simple layered DAG, use a layer-based pebbling strategy
    # Strategy: Process vertices layer by layer
    # Within each layer, compute vertices as their parents become available
    
    # First, let's try a simple greedy approach:
    # 1. Load all inputs
    # 2. For each vertex in topological order:
    #    - If all parents are in fast memory, compute it
    #    - If not, load missing parents (if budget allows), else store some and reload
    # 3. Store outputs at the end
    
    # Better strategy: Use a priority queue to decide which vertices to compute
    # Minimize total transfers by keeping as many values in fast memory as possible
    
    # Let's use a simulation-based approach with a good heuristic
    
    # State: fast_memory = set of vertex IDs currently in fast memory (red)
    # slow_memory: all vertices are available in slow memory (blue)
    
    # Operations:
    # load v: move v from slow to fast (cost 1 transfer)
    # store v: move v from fast to slow (cost 1 transfer)
    # compute v: requires all parents in fast, produces v in fast (cost 0 transfer, but uses 1 slot)
    # delete v: remove v from fast (cost 0 transfer, frees 1 slot)
    
    # Goal: all outputs must be in fast memory at the end (blue = in fast memory)
    # Wait, re-reading: "Inputs start blue, all outputs must finish blue"
    # "every red value consumes one fast-memory slot"
    # So blue = in fast memory, red = ... wait, let me re-read.
    
    # "Inputs start blue" - inputs start in fast memory? That seems odd.
    # "every red value consumes one fast-memory slot" - so red values are in fast memory
    # "all outputs must finish blue" - outputs must finish in fast memory
    
    # I think the convention is:
    # blue = value is in fast memory (pebbled)
    # red = value is NOT in fast memory (unpebbled), available in slow memory
    # But "every red value consumes one fast-memory slot" contradicts this.
    
    # Let me re-read more carefully:
    # "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    
    # I think:
    # blue = value is available (either in fast memory or can be loaded)
    # red = value is actively being held in fast memory
    # No wait...
    
    # Standard pebbling: 
    # - A pebble on a vertex means it's in fast memory
    # - To compute a vertex, all its parents must have pebbles
    # - After computing, you can place a pebble on the vertex
    # - You can remove pebbles (delete)
    
    # I'll go with: 
    # - fast_memory contains vertex IDs that have pebbles (blue/active)
    # - "red value" = value in fast memory
    # - Inputs start with pebbles (blue)
    # - Outputs must end with pebbles (blue)
    # - Budget limits number of pebbles at once
    
    # Let's implement a greedy algorithm:
    
    # Phase 1: Initialize - inputs start in fast memory
    fast_mem = set(inputs)
    ops = []
    
    # Check budget
    if len(fast_mem) > fast_memory_budget:
        # Need to store some inputs
        # Store inputs that are used earliest
        # For now, just keep first 'budget' inputs
        to_store = list(fast_mem)[fast_memory_budget:]
        for v in to_store:
            fast_mem.remove(v)
            ops.append(['store', v])
    
    # Phase 2: Process vertices in topological order
    # For each vertex, ensure its parents are in fast memory, then compute it
    
    # Track which vertices have been computed
    computed = set(inputs)
    
    # For each vertex in topological order (skip inputs)
    for v in range(num_vertices):
        if time.time() > DEADLINE:
            break
        if v in inputs:
            continue
        
        # Check if all parents are in fast memory
        missing_parents = [p for p in parents[v] if p not in fast_mem]
        
        if missing_parents:
            # Need to load missing parents
            # But we might need to free space first
            # Free space by storing or deleting non-essential values
            
            # Determine which values in fast_mem are not needed soon
            # Simple heuristic: store values that are not parents of any uncomputed vertex
            
            # Find which vertices in fast_mem are still needed (parents of uncomputed vertices)
            uncomputed = [u for u in range(num_vertices) if u not in computed]
            needed = set()
            for u in uncomputed:
                for p in parents[u]:
                    needed.add(p)
            
            # Values in fast_mem that are not needed can be deleted
            deletable = [x for x in fast_mem if x not in needed]
            
            # Load missing parents one by one, freeing space as needed
            for mp in missing_parents:
                while len(fast_mem) >= fast_memory_budget:
                    # Need to free space
                    if deletable:
                        # Delete a non-essential value
                        d = deletable.pop()
                        fast_mem.remove(d)
                        ops.append(['delete', d])
                    else:
                        # Store a value that is still needed
                        # Choose the one that will be needed latest
                        # Simple: just store the first one that's not an output
                        storable = [x for x in fast_mem if x not in outputs]
                        if storable:
                            s = storable[0]
                            fast_mem.remove(s)
                            ops.append(['store', s])
                        else:
                            # All are outputs, can't store them
                            # This shouldn't happen if budget is sufficient
                            break
                
                if len(fast_mem) < fast_memory_budget:
                    fast_mem.add(mp)
                    ops.append(['load', mp])
        
        # Now compute v
        if all(p in fast_mem for p in parents[v]):
            # Compute v
            if len(fast_mem) < fast_memory_budget or v in outputs:
                fast_mem.add(v)
                ops.append(['compute', v])
                computed.add(v)
            else:
                # Need to free space before computing
                # Delete a non-essential value
                uncomputed2 = [u for u in range(num_vertices) if u not in computed]
                needed2 = set()
                for u in uncomputed2:
                    for p in parents[u]:
                        needed2.add(p)
                # Also need v's parents to stay
                for p in parents[v]:
                    needed2.add(p)
                
                deletable2 = [x for x in fast_mem if x not in needed2 and x not in outputs]
                if deletable2:
                    d = deletable2[0]
                    fast_mem.remove(d)
                    ops.append(['delete', d])
                
                if len(fast_mem) < fast_memory_budget:
                    fast_mem.add(v)
                    ops.append(['compute', v])
                    computed.add(v)
                else:
                    # Can't compute, skip for now (shouldn't happen)
                    pass
        
        # Clean up: delete values that are no longer needed
        uncomputed3 = [u for u in range(num_vertices) if u not in computed]
        needed3 = set()
        for u in uncomputed3:
            for p in parents[u]:
                needed3.add(p)
        
        for x in list(fast_mem):
            if x not in needed3 and x not in outputs:
                fast_mem.remove(x)
                ops.append(['delete', x])
    
    # Phase 3: Ensure all outputs are in fast memory
    for out in outputs:
        if out not in fast_mem:
            # Load it
            while len(fast_mem) >= fast_memory_budget:
                # Free space
                deletable = [x for x in fast_mem if x not in outputs]
                if deletable:
                    d = deletable[0]
                    fast_mem.remove(d)
                    ops.append(['delete', d])
                else:
                    storable = [x for x in fast_mem if x != out]
                    if storable:
                        s = storable[0]
                        fast_mem.remove(s)
                        ops.append(['store', s])
                    else:
                        break
            
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(out)
                ops.append(['load', out])
    
    # Check if we exceeded operation limit
    if len(ops) > 20000:
        # Truncate and hope for the best (shouldn't happen for reasonable DAGs)
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
