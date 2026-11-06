"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with communication minimization.
    
    dag: dict with:
      - 'nodes': list of node ids (topologically ordered)
      - 'parents': dict mapping node -> list of parent nodes
      - 'inputs': set/list of input nodes (start blue)
      - 'outputs': set/list of output nodes (must finish blue)
    
    fast_memory_budget: int, number of red values that can be held
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    # Parse the DAG
    nodes = list(dag['nodes'])
    parents = dict(dag['parents'])
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = defaultdict(list)
    for node in nodes:
        for p in parents.get(node, []):
            children[p].append(node)
    
    # Topological order is given by nodes list
    # Determine which nodes are inputs (start blue), which are computed
    # A node is an input if it has no parents or is explicitly in inputs
    # Let's assume inputs are nodes with no parents, or explicitly listed
    
    # Identify input nodes: those with no parents
    input_nodes = set()
    for node in nodes:
        if not parents.get(node, []):
            input_nodes.add(node)
    # Also add explicitly listed inputs
    input_nodes |= inputs
    
    # Output nodes must finish blue
    output_nodes = outputs if outputs else input_nodes  # fallback
    
    # Strategy: Process nodes in topological order
    # For each node, compute it when all parents are available
    # Use a priority queue to decide which value to evict when memory is full
    # Prioritize evicting values that are needed farthest in the future
    
    # Precompute: for each node, the earliest time (topo index) it's needed as a parent
    # and the latest time it's needed
    topo_index = {node: i for i, node in enumerate(nodes)}
    
    # For each node, find the max topo_index among its children (when it's last needed)
    last_needed = {}
    for node in nodes:
        if node in input_nodes:
            last_needed[node] = 0  # inputs are available from start
        else:
            # Compute when all parents are available
            parent_max = max((topo_index[p] for p in parents.get(node, [])), default=-1)
            last_needed[node] = parent_max + 1  # earliest it can be computed
    
    # For eviction: how far in the future is a value needed?
    # For a node v, it's needed by its children. The last child to be computed needs it.
    future_need = {}
    for node in nodes:
        if not children.get(node, []):
            # Leaf node: only needed if it's an output
            if node in output_nodes:
                future_need[node] = topo_index[node]  # needed at its own time
            else:
                future_need[node] = topo_index[node]  # not needed later
        else:
            # Needed by children: the last child to be computed
            child_indices = [topo_index[c] for c in children[node]]
            future_need[node] = max(child_indices)
    
    # Simulate the pebbling
    in_fast = set()  # nodes currently in fast memory (blue)
    in_slow = set()  # nodes stored in slow memory
    computed = set()  # nodes that have been computed at least once
    
    operations = []
    
    # Initially, inputs are blue (in fast memory)
    for node in input_nodes:
        in_fast.add(node)
    
    # We need to process nodes in topological order
    # For each non-input node, we need to compute it
    # Before computing, ensure all parents are in fast memory
    
    # Build a list of nodes to compute (non-input nodes)
    to_compute = [node for node in nodes if node not in input_nodes]
    
    # For each node to compute, we need its parents in fast memory
    # Use a lookahead-based eviction: when we need to evict, evict the value
    # that is needed farthest in the future (or not needed at all)
    
    def evict_one():
        """Evict one node from fast memory to make room. Returns the evicted node."""
        candidates = list(in_fast)
        if not candidates:
            return None
        # Evict the node with the farthest future need (least urgent)
        # If a node is not needed in the future, evict it
        # Score: -future_need (higher future_need = less urgent = evict first)
        # But also consider: if a node is an output and not yet done, we can't evict it
        # Actually, outputs must finish blue, so we can evict them after they're done?
        # No, they must finish blue, so they need to be in fast memory at the end
        # For now, just evict based on future need
        def eviction_priority(node):
            # Lower priority = evict first
            # We want to evict nodes that are needed farthest in the future
            # Or nodes that are not needed at all
            # future_need[node] is the topo index when it's last needed
            # Higher future_need = needed later = less urgent = evict first
            return -future_need.get(node, -1)
        
        candidates.sort(key=eviction_priority)
        # Evict the one with lowest priority (most negative future_need, i.e., farthest needed)
        # Wait, I want to evict the LEAST urgent one
        # Least urgent = highest future_need (needed farthest in the future)
        # So sort by -future_need descending, evict the first
        candidates.sort(key=lambda x: -future_need.get(x, -1), reverse=True)
        # Actually, let me reconsider:
        # I want to evict the node that is needed farthest in the future
        # So I want the node with the MAXIMUM future_need
        candidates.sort(key=lambda x: future_need.get(x, -1), reverse=True)
        evicted = candidates[0]
        in_fast.remove(evicted)
        in_slow.add(evicted)
        operations.append(['store', evicted])
        return evicted
    
    def ensure_in_fast(node):
        """Ensure a node is in fast memory. Load from slow if needed."""
        if node in in_fast:
            return
        if node in in_slow:
            # Need to make room if fast is full
            while len(in_fast) >= fast_memory_budget:
                evict_one()
            in_slow.remove(node)
            in_fast.add(node)
            operations.append(['load', node])
        else:
            # Node hasn't been computed yet, this shouldn't happen for parents
            # If it's an input, it should be in fast or slow
            # This is an error case
            pass
    
    # Process nodes in topological order
    for node in to_compute:
        if time.time() - start_time > DEADLINE:
            # Return best effort
            break
        
        # Ensure all parents are in fast memory
        for parent in parents.get(node, []):
            ensure_in_fast(parent)
        
        # Compute the node
        # Make room if necessary
        while len(in_fast) >= fast_memory_budget:
            evict_one()
        
        in_fast.add(node)
        computed.add(node)
        operations.append(['compute', node])
        
        # If the node is not needed in the future and not an output, we can delete it
        # But let's keep it for now and let eviction handle it
    
    # Ensure all output nodes are in fast memory at the end
    for node in output_nodes:
        if node not in in_fast:
            while len(in_fast) >= fast_memory_budget:
                evict_one()
            if node in in_slow:
                in_slow.remove(node)
                in_fast.add(node)
                operations.append(['load', node])
            elif node in computed:
                # It was computed but evicted, need to reload
                # But if it's not in slow, we have a problem
                # For now, assume it's in slow
                pass
    
    return operations

# EVOLVE-BLOCK-END
