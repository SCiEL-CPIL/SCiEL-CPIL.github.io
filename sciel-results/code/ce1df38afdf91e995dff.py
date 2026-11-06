"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def _topological_sort(dag):
    """
    Returns topological ordering of vertices.
    dag structure: we assume dag has 'vertices', 'parents' (dict vertex -> list of parents),
    'outputs' (list of output vertices).
    If dag is a dict with 'vertices', 'parents', 'outputs', use that.
    """
    if hasattr(dag, 'vertices'):
        vertices = dag.vertices
        parents = dag.parents
        outputs = dag.outputs
    elif isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
    else:
        # Try to infer
        raise ValueError("Unknown dag format")
    
    if not vertices:
        return [], {}, []
    
    # Build in-degree map
    children = defaultdict(list)
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            in_degree[v] += 1
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(vertices):
        raise ValueError("Cycle detected or invalid DAG")
    
    return topo, parents, outputs

def _get_inputs(dag, parents):
    """Input vertices are those with no parents."""
    if hasattr(dag, 'vertices'):
        vertices = dag.vertices
    elif isinstance(dag, dict):
        vertices = dag.get('vertices', [])
    else:
        vertices = []
    return [v for v in vertices if not parents.get(v, [])]

def _compute_naive_cost(dag, parents, outputs, topo):
    """
    Naive schedule: compute each vertex once, store results as needed.
    For each vertex, load parents from slow memory if not in fast, compute, store to slow.
    Transfers = number of loads + stores.
    Actually, let's think about what "naive schedule" means in terms of transfers.
    
    In a naive approach:
    - Start with all inputs blue (in slow memory, not in fast).
    - For each vertex in topo order:
      - For each parent: if not in fast, load it (1 transfer)
      - Compute the vertex (no transfer for compute itself, but the result goes to fast)
      - Store the result to slow memory (1 transfer)
      - If parent is no longer needed, it could be deleted, but naive might not delete.
    
    Let's define naive more precisely:
    - All inputs start in slow memory.
    - We process vertices in topological order.
    - For each vertex v:
      - For each parent p of v:
        - If p is not in fast memory, load p from slow (1 transfer).
      - Compute v (result is in fast memory, consumes 1 slot).
      - Store v to slow memory (1 transfer).
      - Now v is in both fast and slow.
    - At the end, all outputs must be blue (in fast memory).
    
    But this uses O(n) fast memory, which may exceed budget. The naive schedule assumes unlimited fast memory.
    
    Number of transfers:
    - Each vertex v with k parents: k loads (assuming parents not already in fast) + 1 store.
    - But if a parent is used by multiple children, it's loaded only once.
    
    Let's compute:
    - For each vertex, it's loaded once (when first needed) and stored once.
    - Inputs are never loaded (they start in slow, but we load them when first needed).
    - Actually, inputs start in slow memory. They need to be loaded when first used.
    
    Total transfers = (number of vertices that are loaded) + (number of vertices that are stored)
    - Every vertex except inputs is stored (computed and stored).
    - Every vertex is loaded at most once (when first needed as a parent, or if it's an output and needs to be in fast at the end... but outputs are stored too).
    
    Wait, let's think again. In the naive schedule with unlimited memory:
    - We load each vertex at most once (from slow to fast).
    - We store each computed vertex once (from fast to slow).
    - Inputs: loaded once when first used.
    - Non-inputs: computed, so they appear in fast, then stored to slow. They don't need to be loaded from slow (they were just computed).
    
    So:
    - Loads: one for each input vertex (when first used).
    - Stores: one for each non-input vertex.
    - Total = num_inputs + num_non_inputs = num_vertices.
    
    Hmm, that seems too simple. Let me reconsider.
    
    Actually, for a vertex v with parents p1, p2, ..., pk:
    - We need p1, p2, ..., pk in fast memory to compute v.
    - If any pi is not in fast, we load it.
    - After computing v, we store v to slow.
    - We keep pi in fast if it's needed by other children.
    
    In the naive schedule (unlimited memory), we never delete anything. So:
    - Each vertex is loaded exactly once (when first needed).
    - Each non-input vertex is stored exactly once.
    - Total transfers = num_vertices (loads for inputs + stores for non-inputs).
    
    But wait, inputs are loaded, and non-inputs are stored. What about vertices that are both? No, inputs have no parents, so they're never computed.
    
    So naive_cost = len(vertices).
    
    But this might not be right. Let's think of a simple example:
    - A -> B -> C
    - A is input, B and C are non-inputs, C is output.
    - Load A (1), compute B (B in fast), store B (1), load B? No, B is already in fast. Compute C (C in fast), store C (1).
    - Total: 3 transfers. num_vertices = 3. OK.
    
    - A -> B, A -> C, B -> D, C -> D
    - A is input, B, C, D are non-inputs, D is output.
    - Load A (1), compute B, store B (1), compute C, store C (1), compute D (B and C in fast), store D (1).
    - Total: 4 transfers. num_vertices = 4. OK.
    
    So naive_cost = len(vertices).
    """
    if hasattr(dag, 'vertices'):
        vertices = dag.vertices
    elif isinstance(dag, dict):
        vertices = dag.get('vertices', [])
    else:
        vertices = []
    return len(vertices)

def pebble(dag, fast_memory_budget):
    """
    Pebble the DAG with the given fast memory budget.
    Returns a list of [operation, vertex] operations.
    """
    topo, parents, outputs = _topological_sort(dag)
    if not topo:
        return []
    
    if hasattr(dag, 'vertices'):
        vertices = dag.vertices
    elif isinstance(dag, dict):
        vertices = dag.get('vertices', [])
    else:
        vertices = []
    
    inputs = _get_inputs(dag, parents)
    naive_cost = _compute_naive_cost(dag, parents, outputs, topo)
    
    # Build children map
    children = defaultdict(list)
    for v in topo:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # For each vertex, compute:
    # - num_children: number of children
    # - last_child_index: the index of the last child in topological order
    
    # Strategy: We need to keep vertices in fast memory if they will be needed again.
    # A vertex v is "needed again" if it has more than one child, and we haven't processed all its children yet.
    
    # Simple heuristic:
    # - Process vertices in topological order.
    # - Keep in fast memory:
    #   - Vertices that have unprocessed children.
    #   - Outputs (must be in fast at the end).
    # - When a vertex's last child is processed, we can delete it from fast memory.
    
    # Let's track:
    # - fast_set: set of vertices currently in fast memory.
    # - remaining_children: for each vertex, how many children are left to process.
    
    fast_set = set()
    remaining_children = {v: len(children[v]) for v in topo}
    operations = []
    
    # Inputs start in slow memory. They need to be loaded when first used.
    # We'll load them when we need them.
    
    for v in topo:
        # Load parents if not in fast
        for p in parents.get(v, []):
            if p not in fast_set:
                operations.append(["load", p])
                fast_set.add(p)
        
        # Compute v
        operations.append(["compute", v])
        fast_set.add(v)
        
        # Store v to slow memory
        operations.append(["store", v])
        
        # Decrement remaining children count for parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
            # If this was the last child, we can delete p from fast memory
            # But only if p is not an output and we have memory pressure
            if remaining_children[p] == 0 and p not in outputs:
                # Check if we should delete p
                # We can delete p if fast_set size exceeds budget, or if we want to free space
                # For now, delete if over budget
                if len(fast_set) > fast_memory_budget:
                    operations.append(["delete", p])
                    fast_set.discard(p)
        
        # If v is not an output and has no children, we can delete it
        if v not in outputs and remaining_children[v] == 0:
            if len(fast_set) > fast_memory_budget:
                operations.append(["delete", v])
                fast_set.discard(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_set:
            operations.append(["load", o])
            fast_set.add(o)
    
    # If we still exceed budget, we need to delete non-output vertices
    # This is a fallback; the above should handle most cases
    
    return operations
# EVOLVE-BLOCK-END
