"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def _get_dag_info(dag):
    """Extract structure from dag object.
    
    Supports common formats:
    - dag is a dict with 'parents' (list of lists), 'outputs' (list of vertex ids), 'n' (num vertices)
    - dag has attributes: .parents, .outputs, .n
    """
    # Try attribute access first
    if hasattr(dag, 'parents'):
        parents = dag.parents
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dag['parents']
    else:
        raise ValueError("Cannot extract parents from dag")
    
    if hasattr(dag, 'outputs'):
        outputs = dag.outputs
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = dag['outputs']
    else:
        outputs = []
    
    if hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        n = len(parents)
    
    return parents, outputs, n

def _build_children(parents):
    """Build children adjacency list from parents."""
    n = len(parents)
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children

def _compute_naive_transfers(dag, fast_memory_budget):
    """Count transfers in naive topological schedule."""
    parents, outputs, n = _get_dag_info(dag)
    
    # Naive: process in topological order (already given), load each vertex when needed
    # A vertex is "loaded" if it needs to be in fast memory
    # In naive schedule: load each input when first needed, compute, keep in memory if budget allows
    # Simplest naive: load/store every vertex that isn't an input (inputs start blue/free)
    # Actually, let's think about what "naive" means here.
    
    # Naive schedule: process vertices in topological order.
    # For each vertex v:
    #   - If v is an input (no parents), it's already in fast memory (blue), no load needed
    #   - For each parent p of v: if p is not in fast memory, load p
    #   - Compute v (requires all parents in fast memory)
    #   - If v is not an output and fast memory is full, store something
    #   - Actually, the naive approach typically just does minimal management
    
    # Let's use a simple definition: naive = load each non-input vertex's parents as needed,
    # compute, and store v if it's not an output and we need space.
    
    # More standard naive: just process in topo order, load parents when not present,
    # compute, and evict oldest non-output when full.
    
    in_memory = set()
    transfers = 0
    
    for v in range(n):
        # Ensure all parents are in memory
        for p in parents[v]:
            if p not in in_memory:
                # Need to load p
                # If memory is full, evict something
                if len(in_memory) >= fast_memory_budget:
                    # Evict a non-output, non-current-parent vertex if possible
                    evicted = False
                    for x in list(in_memory):
                        if x not in parents[v] and x not in outputs:
                            in_memory.discard(x)
                            transfers += 1  # store
                            evicted = True
                            break
                    if not evicted:
                        # Evict first non-output
                        for x in list(in_memory):
                            if x not in outputs:
                                in_memory.discard(x)
                                transfers += 1
                                break
                in_memory.add(p)
                transfers += 1  # load
        
        # Compute v
        # v is now in memory (it was just computed)
        if v not in in_memory:
            in_memory.add(v)
        
        # If v is not an output, we might need to evict it later
        # But in naive, we keep it until space is needed
    
    # Final: all outputs must be in memory (blue) at end
    # Count any stores needed to make room for outputs at the end
    for o in outputs:
        if o not in in_memory:
            if len(in_memory) >= fast_memory_budget:
                for x in list(in_memory):
                    if x not in outputs:
                        in_memory.discard(x)
                        transfers += 1
                        break
            in_memory.add(o)
            transfers += 1
    
    return transfers

def pebble(dag, fast_memory_budget):
    """Pebbling DAG to minimize transfers.
    
    Returns list of [operation, vertex] pairs.
    Operations: 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    
    parents, outputs, n = _get_dag_info(dag)
    children = _build_children(parents)
    
    # Topological order is 0..n-1 (given as topologically ordered)
    # Vertices 0..n-1, inputs are those with no parents
    
    input_vertices = set(v for v in range(n) if len(parents[v]) == 0)
    output_set = set(outputs)
    
    # Naive baseline
    naive_transfers = _compute_naive_transfers(dag, fast_memory_budget)
    
    if fast_memory_budget >= n:
        # All vertices fit in fast memory
        ops = []
        for v in range(n):
            # Ensure parents are in memory (they are, since all fit)
            ops.append(['compute', v])
        return ops
    
    # Greedy strategy:
    # Process in topological order.
    # Keep a priority queue of vertices in memory based on how many children they have
    # (more children = more valuable to keep).
    # When memory is full and we need to load a new vertex, evict the least valuable.
    
    # Better approach: use a heuristic that minimizes reloads.
    # For each vertex, compute how many times it will be needed (number of children).
    # Keep high-usage vertices in memory.
    
    # Let's try a simpler but effective approach:
    # Process in topological order.
    # Maintain a set of vertices in fast memory.
    # When we need to compute v:
    #   1. For each parent p of v not in memory:
    #      - If memory is full, evict the vertex with the fewest remaining uses
    #      - Load p
    #   2. Compute v, add v to memory
    #   3. If v is not an output and memory is over budget, evict least-used
    
    # Precompute remaining uses
    # remaining[v] = number of children of v that haven't been processed yet
    
    def greedy_pebble():
        ops = []
        in_memory = set()
        # Track how many times each vertex in memory will still be needed
        # remaining_children[v] = number of children of v not yet processed
        
        remaining_children = [len(children[v]) for v in range(n)]
        
        # Process in topological order
        for v in range(n):
            # Check deadline
            if time.time() - start_time > DEADLINE - 2:
                break
            
            # Ensure all parents of v are in memory
            for p in parents[v]:
                if p not in in_memory:
                    # Need to load p
                    if len(in_memory) >= fast_memory_budget:
                        # Evict least valuable vertex
                        # Value = remaining_children count (how many more times it's needed)
                        best_evict = None
                        best_val = float('inf')
                        for x in in_memory:
                            if x == v or x in parents[v]:
                                continue  # Don't evict what we need now
                            val = remaining_children[x]
                            if val < best_val:
                                best_val = val
                                best_evict = x
                        if best_evict is None:
                            # Evict any non-current-parent
                            for x in in_memory:
                                if x not in parents[v] and x != v:
                                    best_evict = x
                                    break
                        if best_evict is not None:
                            in_memory.discard(best_evict)
                            ops.append(['store', best_evict])
                    
                    in_memory.add(p)
                    ops.append(['load', p])
            
            # Compute v
            ops.append(['compute', v])
            
            # v is now in memory
            if v not in in_memory:
                in_memory.add(v)
            
            # Update remaining children for v's parents
            for p in parents[v]:
                remaining_children[p] -= 1
            
            # If v is not an output, check if we need to evict to stay within budget
            # Actually, we should keep v if it has remaining children or is an output
            # If v has no remaining children and is not an output, we can evict it to save space
            if v not in output_set and remaining_children[v] == 0:
                # v is no longer needed, evict it to save space
                in_memory.discard(v)
                ops.append(['delete', v])
            
            # If memory is over budget, evict extra
            while len(in_memory) > fast_memory_budget:
                # Evict least valuable
                best_evict = None
                best_val = float('inf')
                for x in in_memory:
                    val = remaining_children[x]
                    if val < best_val:
                        best_val = val
                        best_evict = x
                if best_evict is not None:
                    in_memory.discard(best_evict)
                    ops.append(['store', best_evict])
                else:
                    break
        
        # Ensure all outputs are in memory at the end
        for o in output_set:
            if o not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    # Evict something
                    best_evict = None
                    best_val = float('inf')
                    for x in in_memory:
                        if x not in output_set:
                            val = remaining_children[x]
                            if val < best_val:
                                best_val = val
                                best_evict = x
                    if best_evict is None:
                        for x in in_memory:
                            if x not in output_set:
                                best_evict = x
                                break
                    if best_evict is not None:
                        in_memory.discard(best_evict)
                        ops.append(['store', best_evict])
                in_memory.add(o)
                ops.append(['load', o])
        
        return ops
    
    ops = greedy_pebble()
    
    # Verify no more than 20000 operations
    if len(ops) > 20000:
        # Truncate (shouldn't happen for reasonable DAGs)
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
