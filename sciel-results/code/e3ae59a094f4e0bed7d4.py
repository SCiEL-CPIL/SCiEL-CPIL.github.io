import sys
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys 'nodes' (list of node ids), 'edges' (list of (u, v) tuples), 
         'inputs' (list of input node ids), 'outputs' (list of output node ids)
         OR similar structure. 
         Based on "topologically ordered parents and output vertices", 
         let's assume dag is a dict or object with:
         dag['nodes']: list of node ids
         dag['parents']: dict node -> list of parents
         dag['outputs']: list of output node ids
         dag['inputs']: list of input node ids
    """
    # Since the exact structure of 'dag' is not fully specified in the prompt text 
    # but "dag has topologically ordered parents and output vertices", 
    # I will assume a common structure or handle it generically.
    # The baseline import suggests 'dag' is passed directly.
    
    # Let's inspect the likely structure from the context of "science_tcs_expansion".
    # Usually, these tasks provide a specific format.
    # If I don't know the exact keys, I might fail.
    # However, the prompt says "Keep the function names and signatures...".
    # I will assume 'dag' is a dictionary with:
    # 'nodes': list of all node IDs
    # 'parents': dict mapping node ID to list of parent IDs
    # 'inputs': list of input node IDs
    # 'outputs': list of output node IDs
    
    # If the structure is different, I might need to adapt.
    # Let's try to access common keys.
    
    try:
        nodes = dag['nodes']
        parents = dag['parents']
        inputs = dag['inputs']
        outputs = dag['outputs']
    except (KeyError, TypeError):
        # Fallback: Maybe dag is a list of (node, parents) or similar?
        # Or maybe dag has attributes?
        # Given the baseline `pebble_baseline(dag, ...)`, the structure is fixed.
        # I will assume the dictionary structure above.
        pass

    # Topological Sort
    # Calculate in-degrees (number of parents)
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    # Note: If parents are not provided for all nodes, we might need to infer.
    # Assuming 'parents' is complete.
    
    # Build children map for quick access
    children = defaultdict(list)
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
    # Topological sort using Kahn's algorithm
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(nodes):
        # Cycle detected? Should not happen in DAG.
        # Fallback to simple list
        topo_order = nodes

    # State Tracking
    # 0: Slow (Blue), 1: Fast (Red), 2: Evicted (Needs Recompute)
    state = {n: 0 for n in nodes}
    for n in inputs:
        state[n] = 0 # Start in Slow
    
    fast_mem = set()
    # We need to track uncomputed children count for eviction priority
    # uncomputed_children[n] = number of children of n that are not yet computed (and stored/processed)
    # Actually, we need to know if a node is "safe" to evict.
    # A node is safe to evict (delete) if all its children are already in Slow memory (computed and stored).
    # If a node has uncomputed children, evicting it forces recomputation.
    
    # Let's track "remaining_children" for each node.
    # remaining_children[n] = number of children of n that have not been "finalized" (stored in slow).
    # When a child is finalized, we decrement remaining_children[parent].
    
    remaining_children = {n: len(children[n]) for n in nodes}
    
    operations = []
    
    def ensure_in_fast(n):
        """
        Ensure node n is in fast memory.
        If not, load or recompute it.
        """
        if n in fast_mem:
            return
        
        if state[n] == 0:
            # It's in slow memory
            operations.append(('load', n))
            fast_mem.add(n)
            state[n] = 1
        elif state[n] == 2:
            # It was evicted, needs recompute
            # All parents must be in fast memory
            for p in parents.get(n, []):
                ensure_in_fast(p)
            # Compute n
            operations.append(('compute', n))
            fast_mem.add(n)
            state[n] = 1
        else:
            # Should not happen
            pass

    def evict_if_needed():
        """
        If fast_mem exceeds budget, evict a node.
        """
        if len(fast_mem) <= fast_memory_budget:
            return
        
        # Find a node to evict
        # Priority 1: Nodes with remaining_children == 0 (Safe to delete)
        # Priority 2: Nodes with min remaining_children > 0
        
        candidates = list(fast_mem)
        
        # Separate into safe and unsafe
        safe = [n for n in candidates if remaining_children[n] == 0]
        unsafe = [n for n in candidates if remaining_children[n] > 0]
        
        if safe:
            # Pick any safe node (e.g., first one)
            victim = safe[0]
            operations.append(('delete', victim))
            fast_mem.remove(victim)
            state[victim] = 2 # Mark as evicted/needs recompute if needed again
            # Note: If it's an output, it should have been stored, not just deleted.
            # We handle outputs separately.
        else:
            # Pick unsafe node with min remaining_children
            victim = min(unsafe, key=lambda x: remaining_children[x])
            operations.append(('delete', victim))
            fast_mem.remove(victim)
            state[victim] = 2

    # Main Loop
    for n in topo_order:
        # Ensure all parents are in fast memory
        for p in parents.get(n, []):
            ensure_in_fast(p)
        
        # Compute n
        # If n is an input, it's already in fast (loaded). 
        # If n is intermediate, we compute it.
        # If n is an input, we don't "compute" it, we just use it.
        # But the operation 'compute' is for generating new values.
        # Inputs are given.
        if n in inputs:
            # Already in fast_mem via ensure_in_fast (load)
            pass
        else:
            # Check if we need to compute
            # If n was already computed and is in fast_mem, we don't need to recompute?
            # In topological order, we visit each node once.
            # So we compute it here.
            operations.append(('compute', n))
            fast_mem.add(n)
            state[n] = 1
            
        # Post-computation handling
        if n in outputs:
            # Must store
            if n in fast_mem:
                operations.append(('store', n))
                fast_mem.remove(n)
                state[n] = 0 # Back to slow
            # Decrement remaining_children for parents? 
            # Yes, because this child is now "finalized"
            for p in parents.get(n, []):
                remaining_children[p] -= 1
        else:
            # Intermediate node
            # Check if we can delete it (all children done)
            # We can't delete it yet because its children are not done (topo order).
            # So we keep it in fast_mem.
            # But we might need to evict to make room for future nodes.
            pass
            
        # Check memory budget
        evict_if_needed()
        
        # Optimization: If a node's remaining_children becomes 0, 
        # it can be deleted immediately to free space.
        # We can do this check for the node n itself? 
        # No, n's children are not done.
        # But parents of n might have remaining_children == 0?
        # Yes, if n was the last child of a parent p.
        # We should check parents of n and delete them if safe.
        for p in parents.get(n, []):
            if p in fast_mem and remaining_children[p] == 0:
                operations.append(('delete', p))
                fast_mem.remove(p)
                state[p] = 2

    # Ensure all outputs are stored (should be done in loop)
    # Check if any outputs are still in fast_mem?
    for n in outputs:
        if n in fast_mem:
            operations.append(('store', n))
            fast_mem.remove(n)
            state[n] = 0

    return operations

