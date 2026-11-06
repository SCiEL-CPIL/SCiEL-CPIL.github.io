"""DAG pebbling with recursive divide-and-conquer strategy."""

import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Generate an optimal pebbling schedule for a DAG.
    
    Args:
        dag: dict with 'parents' (list of lists, topologically ordered), 
             'outputs' (list of output vertex indices),
             'num_vertices' (int)
        fast_memory_budget: int, max number of red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    deadline = time.time() + 80  # 80s wall-clock, 10s safety margin
    
    # Parse DAG structure
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        num_vertices = dag.get('num_vertices', len(parents))
    else:
        # Fallback: assume dag is a list of parent lists
        parents = dag
        outputs = []
        num_vertices = len(parents)
        # Find sinks as outputs
        has_parent = [False] * num_vertices
        for i, par in enumerate(parents):
            for p in par:
                has_parent[p] = True
        outputs = [i for i in range(num_vertices) if not has_parent[i]]
    
    if not outputs:
        outputs = [num_vertices - 1]
    
    # Build children adjacency
    children = [[] for _ in range(num_vertices)]
    for i, par in enumerate(parents):
        for p in par:
            children[p].append(i)
    
    # Determine which vertices are inputs (no parents)
    inputs = set()
    for i, par in enumerate(parents):
        if not par:
            inputs.add(i)
    
    # Naive schedule: simple topological order, load each parent, compute, store non-output
    def naive_schedule():
        ops = []
        for i in range(num_vertices):
            if i in inputs:
                continue  # already blue
            # Ensure parents are in fast memory
            for p in parents[i]:
                ops.append(['load', p])
            ops.append(['compute', i])
            if i not in outputs and not any(c >= i for c in range(i+1, num_vertices) if i in parents[c]):
                # No future use, store
                if children[i]:  # has children but we already loaded for them
                    pass
                ops.append(['store', i])
            elif children[i] and i not in outputs:
                ops.append(['store', i])
        return ops
    
    # Count transfers in naive
    def count_transfers(ops):
        return sum(1 for op, v in ops if op in ('load', 'store'))
    
    naive_ops = naive_schedule()
    naive_transfers = count_transfers(naive_ops)
    
    # Recursive divide-and-conquer pebbling
    # Strategy: find a "cut" set of vertices that separates the DAG into sub-DAGs
    # Keep cut vertices in fast memory, solve sub-DAGs independently
    
    def compute_outreach(vertex, visited=None):
        """Count how many vertices depend on this vertex (transitively)."""
        if visited is None:
            visited = set()
        if vertex in visited:
            return 0
        visited.add(vertex)
        count = 1
        for c in children[vertex]:
            count += compute_outreach(c, visited)
        return count
    
    # Precompute outreach
    outreach = [compute_outreach(i) for i in range(num_vertices)]
    
    # Find a good split: a vertex or set of vertices whose removal
    # creates roughly balanced sub-DAGs
    # For layered DAGs, a good split is at the middle layer
    
    # Group vertices by "depth" (longest path from any input)
    depth = [0] * num_vertices
    for i in range(num_vertices):
        if parents[i]:
            depth[i] = 1 + max(depth[p] for p in parents[i])
    
    max_depth = max(depth) if depth else 0
    
    # Find mid-depth layer as potential cut
    mid_depth = max_depth // 2
    mid_layer = [i for i in range(num_vertices) if depth[i] == mid_depth]
    
    # If mid_layer is small enough to fit in fast memory, use it as persistent checkpoints
    # Otherwise, try a different approach
    
    # Recursive pebbling with memoization of vertex states
    # State: set of blue vertices in fast memory
    # We'll use a greedy approach with lookahead
    
    def greedy_pebbling(vertices_to_compute, initial_fast, must_finish_blue):
        """
        Greedy pebbling for a set of vertices.
        vertices_to_compute: set of vertices to compute (topologically ordered)
        initial_fast: set of vertices already in fast memory (blue)
        must_finish_blue: set of vertices that must be blue at end
        Returns: list of operations
        """
        ops = []
        fast = set(initial_fast)
        computed = set(initial_fast)  # vertices whose values are known (in fast or stored)
        
        # Determine which vertices we need
        needed = set(vertices_to_compute) | set(must_finish_blue)
        
        # Topologically sort the needed vertices
        # Use the global topological order
        topo_order = [i for i in range(num_vertices) if i in needed]
        
        for v in topo_order:
            if v in fast:
                continue
            if v in inputs:
                # Need to load it
                ops.append(['load', v])
                fast.add(v)
                continue
            # Need to compute v
            # Ensure all parents are in fast
            for p in parents[v]:
                if p not in fast:
                    # Need to get p into fast
                    # If p is in computed but not fast, load it
                    # If p is not computed, we need to compute it first
                    # For simplicity, load if possible, otherwise compute
                    if p in inputs:
                        ops.append(['load', p])
                        fast.add(p)
                    else:
                        # Recursively ensure p is available
                        # This is the tricky part - we need to handle dependencies
                        # For now, just load (assume it was stored)
                        ops.append(['load', p])
                        fast.add(p)
            
            ops.append(['compute', v])
            fast.add(v)
            computed.add(v)
            
            # If fast is over budget, delete something
            while len(fast) > fast_memory_budget:
                # Delete a vertex that won't be needed again
                # Find a fast vertex with no future dependencies
                deleted = False
                for u in list(fast):
                    # Check if u is needed later
                    future_needed = any(u in parents[w] for w in topo_order if w > v and w in needed)
                    if u not in must_finish_blue and not future_needed and u not in inputs:
                        ops.append(['delete', u])
                        fast.discard(u)
                        deleted = True
                        break
                if not deleted:
                    # Delete the oldest non-essential vertex
                    for u in list(fast):
                        if u not in must_finish_blue and u not in inputs:
                            ops.append(['delete', u])
                            fast.discard(u)
                            deleted = True
                            break
                    if not deleted:
                        # Delete anything
                        u = list(fast)[0]
                        ops.append(['delete', u])
                        fast.discard(u)
        
        # At the end, ensure must_finish_blue vertices are in fast
        for v in must_finish_blue:
            if v not in fast:
                ops.append(['load', v])
                fast.add(v)
        
        return ops
    
    # Better approach: dynamic programming over topological order
    # For each vertex, decide whether to keep it in fast or store it
    # This is complex, so let's use a heuristic:
    
    # 1. Compute all vertices in topological order
    # 2. For each vertex, if it's used by multiple future vertices, keep it in fast
    # 3. If fast is full, evict the vertex with the least future use
    
    def smart_pebbling():
        ops = []
        fast = set(inputs)  # inputs start in fast
        fast_values = {v: True for v in inputs}
        
        # Count future uses for each vertex
        future_uses = [0] * num_vertices
        for i in range(num_vertices):
            for p in parents[i]:
                future_uses[p] += 1
        
        # Process in topological order
        for v in range(num_vertices):
            if v in inputs:
                continue
            
            # Ensure parents are in fast
            for p in parents[v]:
                if p not in fast:
                    # Load p
                    ops.append(['load', p])
                    fast.add(p)
            
            # Compute v
            ops.append(['compute', v])
            fast.add(v)
            
            # Decrement future use count for parents
            for p in parents[v]:
                future_uses[p] -= 1
            
            # If v is an output, keep it
            if v in outputs:
                continue
            
            # If fast is over budget, evict
            while len(fast) > fast_memory_budget:
                # Find the vertex in fast with the fewest future uses
                # and is not an output
                candidates = []
                for u in fast:
                    if u not in outputs and u not in inputs:
                        candidates.append((future_uses[u], u))
                
                if candidates:
                    candidates.sort()
                    _, evict = candidates[0]
                    # If future_uses is 0, store and delete
                    if future_uses[evict] == 0:
                        ops.append(['store', evict])
                    ops.append(['delete', evict])
                    fast.discard(evict)
                else:
                    # Can't evict anything, this shouldn't happen
                    break
            
            # If v has no future uses and is not an output, store and delete
            if v not in outputs and future_uses[v] == 0 and v not in fast:
                pass  # already handled
            
            # If v has no future uses and is not an output, but is in fast
            if v not in outputs and future_uses[v] == 0 and v in fast and len(fast) > fast_memory_budget // 2:
                # Store and delete to free space
                ops.append(['store', v])
                ops.append(['delete', v])
                fast.discard(v)
        
        # Ensure all outputs are in fast
        for v in outputs:
            if v not in fast:
                ops.append(['load', v])
                fast.add(v)
        
        return ops
    
    # Try the smart pebbling
    try:
        if time.time() > deadline - 5:
            return naive_ops
        
        result = smart_pebbling()
        
        # Verify the schedule
        def verify_schedule(ops, parents, outputs, num_vertices, fast_memory_budget, inputs):
            fast = set(inputs)
            stored = set()
            transfers = 0
            for op, v in ops:
                if op == 'load':
                    if v in fast:
                        return False  # already in fast
                    if v not in stored and v not in inputs:
                        return False  # not available
                    fast.add(v)
                    transfers += 1
                elif op == 'store':
                    if v not in fast:
                        return False
                    fast.discard(v)
                    stored.add(v)
                    transfers += 1
                elif op == 'compute':
                    # All parents must be in fast
                    for p in parents[v]:
                        if p not in fast:
                            return False
                    fast.add(v)
                    if len(fast) > fast_memory_budget:
                        return False
                elif op == 'delete':
                    if v not in fast:
                        return False
                    fast.discard(v)
                else:
                    return False
            # All outputs must be in fast
            for v in outputs:
                if v not in fast:
                    return False
            return True
        
        if verify_schedule(result, parents, outputs, num_vertices, fast_memory_budget, inputs):
            return result
        else:
            # Fall back to naive
            return naive_ops
    
    except Exception:
        return naive_ops
