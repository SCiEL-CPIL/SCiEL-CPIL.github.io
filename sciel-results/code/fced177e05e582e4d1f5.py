"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """
    Returns a list of vertices in topological order.
    vertices: list of vertex IDs
    parents: dict mapping vertex -> list of parent IDs
    """
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return vertices
    return order

def _validate_schedule(dag, ops, fast_memory_budget):
    """
    Validates a schedule by simulating the state machine.
    Returns True if valid, False otherwise.
    """
    vertices = set(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    blue = set(inputs)
    red = set()
    
    for op, v in ops:
        if v not in vertices:
            return False
        
        if op == 'load':
            if v not in blue or v in red:
                return False
            if len(red) >= fast_memory_budget:
                return False
            red.add(v)
            blue.remove(v)
            
        elif op == 'store':
            if v not in red or v in blue:
                return False
            red.remove(v)
            blue.add(v)
            
        elif op == 'compute':
            for p in parents.get(v, []):
                if p not in red:
                    return False
            
            if v not in red:
                if len(red) >= fast_memory_budget:
                    return False
                red.add(v)
            
        elif op == 'delete':
            if v not in red:
                return False
            red.remove(v)
            
        else:
            return False
            
    for v in outputs:
        if v not in blue:
            return False
            
    return True

def _count_transfers(ops):
    """
    Counts the number of transfers (loads + stores).
    """
    count = 0
    for op, v in ops:
        if op == 'load' or op == 'store':
            count += 1
    return count

def _simple_schedule(dag, fast_memory_budget):
    """
    A robust simple schedule:
    1. Load all inputs.
    2. Compute vertices in topological order.
    3. When memory is full, delete vertices that are no longer needed.
    4. Store outputs when they are no longer needed as parents, or at the end.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    ops = []
    in_memory = set()
    
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # 1. Load all inputs
    for v in order:
        if not parents.get(v, []):
            if len(in_memory) >= fast_memory_budget:
                return None
            ops.append(['load', v])
            in_memory.add(v)
            
    # 2. Compute in topological order
    for v in order:
        if not parents.get(v, []):
            continue
            
        if len(in_memory) >= fast_memory_budget:
            freed = False
            for u in list(in_memory):
                if u == v:
                    continue
                if remaining_children[u] == 0:
                    if u in outputs:
                        ops.append(['store', u])
                    else:
                        ops.append(['delete', u])
                    in_memory.remove(u)
                    freed = True
                    break
                    
            if not freed:
                return None
                
        ops.append(['compute', v])
        in_memory.add(v)
        
        for p in parents[v]:
            remaining_children[p] -= 1
            
    # 3. Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def _bottom_up_schedule(dag, fast_memory_budget):
    """
    A bottom-up pebbling strategy.
    This is conceptually different but in a DAG with inputs at top and outputs at bottom,
    "bottom-up" might not be straightforward if we need inputs to compute outputs.
    However, the prompt mentions "bottom-up pebbling strategy that computes vertices in reverse topological order".
    This is usually for pebbling games where you place pebbles.
    Here, we must load inputs (blue) to compute.
    If we process in reverse topological order, we would be computing outputs first?
    No, outputs depend on parents.
    
    Let's stick to a refined topological approach but optimize eviction.
    The "bottom-up" inspiration might refer to processing children before parents in some context,
    but for computation flow, we must go parents -> children.
    
    Maybe the inspiration implies a different strategy for memory management?
    "storing only the final output values in fast memory and discarding intermediate results immediately after their last consumer is processed"
    This is exactly what _simple_schedule does.
    
    Let's try to improve _simple_schedule by being smarter about eviction.
    Instead of just picking the first available vertex, pick the one that minimizes future loads?
    Or simply ensure we don't evict an output too early if it might be needed?
    Wait, outputs are the final vertices (usually). If an output has children, it's not a final output in the DAG sense, 
    but the task says "output vertices". Usually outputs are sinks.
    If outputs are sinks, they have no children. So remaining_children[output] == 0 immediately upon computation.
    
    Let's try a strategy that keeps "hot" vertices (those with many remaining children) and evicts "cold" ones.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    ops = []
    in_memory = set()
    
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Load inputs
    for v in order:
        if not parents.get(v, []):
            if len(in_memory) >= fast_memory_budget:
                return None
            ops.append(['load', v])
            in_memory.add(v)
            
    # Compute
    for v in order:
        if not parents.get(v, []):
            continue
            
        if len(in_memory) >= fast_memory_budget:
            # Evict the vertex with the fewest remaining children (excluding v itself, though v isn't in memory yet)
            # And prefer evicting non-outputs if possible?
            # Actually, if remaining_children is 0, it's safe.
            # Among safe vertices, pick one.
            candidates = [u for u in in_memory if u != v and remaining_children[u] == 0]
            if candidates:
                # Pick the first one, or the one that is not an output?
                # If it's an output, we must store it. If not, delete.
                # Storing is a transfer. Deleting is not.
                # We want to minimize transfers.
                # So we prefer to delete non-outputs over storing outputs?
                # But if we store an output, it's done. If we delete a non-output, it's done.
                # Both free space.
                # However, if we store an output, we incur a cost.
                # If we delete a non-output, we incur no cost (in terms of transfers).
                # So we should prefer to delete non-outputs if possible.
                
                non_outputs = [u for u in candidates if u not in outputs]
                if non_outputs:
                    to_evict = non_outputs[0]
                    ops.append(['delete', to_evict])
                else:
                    to_evict = candidates[0]
                    ops.append(['store', to_evict])
                
                in_memory.remove(to_evict)
            else:
                return None
                
        ops.append(['compute', v])
        in_memory.add(v)
        
        for p in parents[v]:
            remaining_children[p] -= 1
            
    # Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0
    
    try:
        # Try the simple schedule
        schedule1 = _simple_schedule(dag, fast_memory_budget)
        if schedule1 is not None and _validate_schedule(dag, schedule1, fast_memory_budget):
            cost1 = _count_transfers(schedule1)
        else:
            cost1 = float('inf')
            schedule1 = None
            
        # Try the bottom-up (refined eviction) schedule
        schedule2 = _bottom_up_schedule(dag, fast_memory_budget)
        if schedule2 is not None and _validate_schedule(dag, schedule2, fast_memory_budget):
            cost2 = _count_transfers(schedule2)
        else:
            cost2 = float('inf')
            schedule2 = None
            
        # Pick the best
        if cost1 <= cost2:
            return schedule1 if schedule1 is not None else []
        else:
            return schedule2 if schedule2 is not None else []
            
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
