# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict
from functools import lru_cache
import math

_DEADLINE_SECONDS = 80.0

def _toposort_vertices(dag):
    """Extract vertices and build adjacency from the dag structure."""
    if 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif 'nodes' in dag:
        vertices = list(dag['nodes'])
    else:
        all_v = set()
        parents = dag.get('parents', {})
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        vertices = sorted(all_v)

    if 'parents' in dag:
        parents = dag['parents']
    elif 'edges' in dag:
        parents = {}
        for edge in dag['edges']:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                p, c = edge
                parents.setdefault(c, []).append(p)
            elif isinstance(edge, dict):
                parents[edge['child']] = [edge['parent']]
    else:
        parents = {}

    if 'outputs' in dag:
        outputs = set(dag['outputs'])
    elif 'output' in dag:
        outputs = set(dag['output']) if isinstance(dag['output'], (list, set, tuple)) else {dag['output']}
    else:
        children = {}
        for v, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(v)
        outputs = {v for v in vertices if v not in children}

    if 'inputs' in dag:
        inputs = set(dag['inputs'])
    elif 'blue' in dag:
        inputs = set(dag['blue'])
    elif 'blue_vertices' in dag:
        inputs = set(dag['blue_vertices'])
    else:
        inputs = {v for v in vertices if not parents.get(v)}

    return vertices, parents, outputs, inputs

def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))

def _validate_ops(ops, dag, fast_memory_budget):
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    blue = set(inputs)
    fast_mem = set()
    available = set(inputs)
    
    for i, (op, v) in enumerate(ops):
        if op == 'load':
            if v not in inputs:
                return False
            if v in fast_mem:
                return False
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'store':
            if v not in fast_mem and v not in blue:
                return False
            if v in fast_mem:
                fast_mem.remove(v)
            blue.add(v)
        elif op == 'compute':
            ps = parents.get(v, [])
            for p in ps:
                if p not in available:
                    return False
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'delete':
            if v not in fast_mem:
                return False
            fast_mem.remove(v)
            if v not in blue:
                available.discard(v)
        else:
            return False
    
    if not outputs.issubset(blue):
        return False
    
    return True

def _get_relevant_subgraph(dag):
    """Identify vertices that are ancestors of any output."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    # BFS/DFS from outputs to find ancestors
    relevant = set(outputs)
    queue = deque(list(outputs))
    while queue:
        v = queue.popleft()
        for p in parents.get(v, []):
            if p not in relevant:
                relevant.add(p)
                queue.append(p)
                
    # Filter
    filtered_vertices = [v for v in vertices if v in relevant]
    filtered_parents = {v: [p for p in parents.get(v, []) if p in relevant] for v in filtered_vertices}
    filtered_outputs = outputs
    filtered_inputs = {v for v in inputs if v in relevant}
    
    return filtered_vertices, filtered_parents, filtered_outputs, filtered_inputs

def _dfs_pebble(dag, fast_memory_budget):
    """
    Depth-First Search pebbling strategy.
    Minimizes transfers by only loading relevant inputs and storing outputs.
    Uses O(depth) fast memory.
    """
    # Use relevant subgraph
    vertices, parents, outputs, inputs = _get_relevant_subgraph(dag)
    
    # Re-build children for relevant subgraph
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    # Topological order is not strictly needed for DFS, but we need to ensure
    # we process children before deleting parents.
    
    ops = []
    fast_mem = set()
    
    # We need to know which vertices are "done" (all children processed)
    # to decide when to delete.
    # Since it's a DAG, a node might have multiple parents.
    # But in DFS, we enter a node, process its children, then leave.
    # If a node is shared, this simple DFS might fail if we delete it
    # before another parent's subtree is done?
    # Actually, if we treat the DAG as a tree of dependencies from the outputs,
    # we can just recurse.
    
    # However, a node can be a parent to multiple children.
    # If we recurse on child 1, then child 2, the parent stays in memory.
    # After all children are done, we delete the parent (if not output).
    
    # What if a node is a parent to children in different branches?
    # e.g. A -> B, A -> C.
    # DFS(A):
    #   Load A
    #   DFS(B)
    #   DFS(C)
    #   Delete A
    # This works.
    
    # What if A and D are parents of B?
    # DFS(B) needs A and D.
    # If we are coming from A, A is in memory. D might not be.
    # We need to ensure all parents are in memory before computing B.
    
    # Strategy:
    # process(v):
    #   if v in fast_mem: return
    #   if v in inputs:
    #      load v
    #   else:
    #      for p in parents[v]:
    #         process(p)
    #      # Now all parents are in fast_mem?
    #      # If a parent was deleted after its other children were done, it's gone.
    #      # This is the problem with DAGs.
    #      
    #      # To handle DAGs safely with low memory, we might need to re-compute or re-load.
    #      # But re-loading increases transfers.
    #      
    #      # For "layered DAGs" which are often trees or have limited sharing,
    #      # the DFS tree approach works if we don't delete a node until ALL its children are done.
    #      # But if a node is a parent to children in different subtrees,
    #      # we keep it in memory throughout.
    
    # Let's implement a recursive DFS that keeps nodes in memory until all children are processed.
    
    def process(v):
        if v in fast_mem:
            return
        
        # Ensure space
        # We don't evict arbitrarily. We rely on the structure.
        # If we run out of space, this simple strategy fails.
        # But for layered DAGs, depth is usually small.
        
        if v in inputs:
            ops.append(('load', v))
            fast_mem.add(v)
        else:
            # Compute v
            # All parents must be in fast_mem
            for p in parents.get(v, []):
                process(p)
            
            # Check if v can be computed (parents in fast_mem)
            # If a parent was evicted, this fails.
            # But we only delete after all children are done.
            # So if v is a child of p, and we are processing v, p is still in memory
            # UNLESS p was deleted because its other children were done and v was not one of them?
            # No, v IS a child of p. So p is not deleted until v is done.
            
            # So, if we process children in a way that keeps parents alive, it works.
            
            ops.append(('compute', v))
            fast_mem.add(v)
        
        # Process children
        for c in children.get(v, []):
            process(c)
        
        # Post-order: v's children are done
        if v in outputs:
            if v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
        else:
            if v in fast_mem:
                ops.append(('delete', v))
                fast_mem.discard(v)
                
    # Start from all outputs
    for o in outputs:
        process(o)
        
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    try:
        # Try DFS pebbling on relevant subgraph
        ops = _dfs_pebble(dag, fast_memory_budget)
        
        if _validate_ops(ops, dag, fast_memory_budget) and len(ops) <= 20000:
            return ops
            
        # Fallback to naive if DFS fails (e.g. memory overflow)
        # But naive might also fail if memory is tight.
        # For now, return the DFS ops even if invalid? No, must be valid.
        # If DFS is invalid, it likely exceeded memory.
        # In that case, we might need a more sophisticated eviction.
        # But given the 1.0 baseline, the naive schedule is valid.
        # So memory is likely sufficient for the naive schedule.
        # DFS uses less memory than naive (which loads all inputs).
        # So DFS should be valid if naive is valid.
        
    except Exception:
        pass
        
    # Fallback: Naive schedule (filtered to relevant inputs)
    vertices, parents, outputs, inputs = _get_relevant_subgraph(dag)
    ops = []
    for v in vertices:
        if v in inputs:
            ops.append(('load', v))
        else:
            ops.append(('compute', v))
    for v in outputs:
        ops.append(('store', v))
        
    if _validate_ops(ops, dag, fast_memory_budget):
        return ops
        
    # Last resort: Original naive
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    ops = []
    for v in vertices:
        if v in inputs:
            ops.append(('load', v))
        else:
            ops.append(('compute', v))
    for v in outputs:
        ops.append(('store', v))
        
    return ops

# EVOLVE-BLOCK-END
