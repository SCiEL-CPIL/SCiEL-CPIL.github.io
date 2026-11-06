"""Rewritten program: heuristic pebbling with recompute-aware scheduling."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    dag: dict or object with keys:
      'vertices': list of vertex ids (topologically ordered)
      'parents': dict vertex -> list of parent vertices (topologically ordered)
      'inputs': list of input vertices
      'outputs': list of output vertices
    fast_memory_budget: int, max red pebbles allowed at any time
    
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}
    """
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    else:
        # Try attribute access
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        inputs = getattr(dag, 'inputs', [])
        outputs = getattr(dag, 'outputs', [])
    
    if not vertices:
        return []
    
    # Build adjacency: children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine in-degree for topological order (should already be topo-ordered)
    # Compute reverse topological order
    rev_vertices = list(reversed(vertices))
    
    # State tracking
    # blue: vertex has value in slow memory (input or stored)
    # red: vertex has value in fast memory
    blue = set(inputs)
    red = set()
    
    operations = []
    
    def is_input(v):
        return v in inputs
    
    def is_output(v):
        return v in outputs
    
    def can_compute(v):
        """Check if all parents of v are in red memory."""
        for p in parents.get(v, []):
            if p not in red:
                return False
        return True
    
    def compute_cost(v):
        """Number of operations needed to make v available in red."""
        # v must be computed, which requires all parents in red
        # We need to ensure parents are in red
        cost = 1  # the compute operation itself
        for p in parents.get(v, []):
            if p not in red:
                if p in blue:
                    cost += 1  # load
                else:
                    # Need to recompute p
                    cost += compute_cost(p)
        return cost
    
    def make_red(v):
        """Make vertex v red (in fast memory). Returns list of operations."""
        ops = []
        if v in red:
            return ops
        # Ensure all parents are red
        for p in parents.get(v, []):
            ops.extend(make_red(p))
        # Now compute v
        ops.append(['compute', v])
        red.add(v)
        blue.discard(v)  # Once computed, we might still keep it blue if stored, but for now mark as red
        return ops
    
    def delete_vertex(v):
        """Delete vertex v from fast memory (if red)."""
        if v in red:
            red.discard(v)
            # It's now only in blue if it was an input or was stored
            # For non-input vertices, if we delete from red and don't store, it's gone
            # But we only delete red pebbles; blue means in slow memory
            # If v was computed, it's not in blue unless we stored it
            # So after delete, v is not in red. It's in blue only if it was an input or we stored it.
            # We track blue separately.
            pass
        # Return the operation
        return ['delete', v]
    
    def store_vertex(v):
        """Store vertex v in slow memory (blue)."""
        if v in red:
            blue.add(v)
            return ['store', v]
        return []
    
    def load_vertex(v):
        """Load vertex v from slow memory to fast memory."""
        if v in blue and v not in red:
            red.add(v)
            return ['load', v]
        return []
    
    # Strategy: Process vertices in topological order.
    # For each vertex, ensure it's computed.
    # After computing a vertex, decide whether to keep it in red or store/delete.
    # To minimize transfers, we want to keep values in red as long as they're needed.
    
    # Better strategy: 
    # 1. Load all inputs into red (up to budget)
    # 2. Process vertices in topological order
    # 3. For each vertex, if not all parents are red, load/recompute missing parents
    # 4. Compute the vertex
    # 5. After computing, if fast memory is full and vertex is not needed soon, store it
    
    # Let's use a simpler approach:
    # Greedy: process in topo order, keep a window of recent vertices in red.
    
    # Initialize: load inputs
    for v in inputs:
        if len(red) >= fast_memory_budget:
            # Need to make space. Store the least recently used non-output vertex?
            # For simplicity, store the first non-output vertex in red
            for r in list(red):
                if r not in outputs:
                    operations.append(['store', r])
                    blue.add(r)
                    red.discard(r)
                    break
        operations.append(['load', v])
        red.add(v)
        blue.add(v)  # Inputs are already blue
    
    # Process vertices in topological order
    # For each vertex, ensure it can be computed
    for v in vertices:
        if v in inputs:
            continue  # Already handled
        
        # Check if all parents are in red
        need_to_load = []
        for p in parents.get(v, []):
            if p not in red:
                need_to_load.append(p)
        
        # Load missing parents
        for p in need_to_load:
            if len(red) >= fast_memory_budget:
                # Make space: store a red vertex that's not an output and not needed immediately
                # Simple heuristic: store the first red vertex that's not an output
                for r in list(red):
                    if r not in outputs and r != v:
                        operations.append(['store', r])
                        blue.add(r)
                        red.discard(r)
                        break
            if p in blue:
                operations.append(['load', p])
                red.add(p)
            else:
                # Need to recompute p
                ops = make_red(p)
                operations.extend(ops)
        
        # Now compute v
        operations.append(['compute', v])
        red.add(v)
        blue.discard(v)  # v is now in red, not necessarily blue
        
        # If fast memory is full and v is not an output, consider storing it
        # But we'll handle this when we need space for future loads
    
    # Ensure all outputs are in red (they should be, since we computed them)
    # But we need them to be blue at the end
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            blue.add(v)
            red.discard(v)
    
    return operations
# EVOLVE-BLOCK-END
