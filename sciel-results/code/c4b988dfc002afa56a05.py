"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Try optimal Dijkstra-based search for small problems
    if len(needed_vertices) <= 12 and fast_memory_budget <= 4:
        result = _optimal_pebble(needed_vertices, parent_dict, needed_set,
                                 inputs, outputs, fast_memory_budget, deadline)
        if result is not None:
            return result

    # Fallback to improved greedy approach
    result = _improved_greedy_pebble(needed_vertices, parent_dict, needed_set,
                                     inputs, outputs, fast_memory_budget, deadline,
                                     vertex_to_idx, children)
    
    # Try local refinement to reduce transfers
    if time.time() < deadline - 5:
        refined = _local_refine(result, needed_vertices, parent_dict, needed_set,
                                inputs, outputs, fast_memory_budget, deadline)
        if refined is not None:
            result = refined
    
    return result


def _optimal_pebble(needed_vertices, parent_dict, needed_set,
                    inputs, outputs, fast_memory_budget, deadline):
    """
    Dijkstra-based search over memory states to find minimum-transfer pebbling.
    State: (frozenset of vertices in fast memory, frozenset of computed vertices)
    Cost: number of transfers (load/store operations)
    """
    # Initial state: no fast memory, no computed vertices
    init_fast = frozenset()
    init_computed = frozenset()
    init_state = (init_fast, init_computed)

    # Dijkstra's algorithm
    # dist[state] = minimum transfers to reach this state
    dist = {init_state: 0}
    # parent tracking: state -> (prev_state, operation, vertex)
    parent = {}
    
    # Priority queue: (cost, fast_frozen, computed_frozen)
    pq = [(0, init_fast, init_computed)]
    
    # Track number of states explored for timeout
    states_explored = 0
    max_states = 50000  # Limit to avoid timeout
    
    while pq:
        if time.time() > deadline - 2:
            return None
        if states_explored > max_states:
            return None
            
        cost, fast, computed = heapq.heappop(pq)
        
        if cost > dist.get((fast, computed), float('inf')):
            continue
            
        states_explored += 1
        
        # Check if we've reached goal: all outputs are in fast memory
        if outputs.issubset(fast):
            # Reconstruct path
            ops = []
            state = (fast, computed)
            while state in parent:
                prev_state, op, vertex = parent[state]
                ops.append([op, vertex])
                state = prev_state
            ops.reverse()
            return ops
        
        # Generate successor states
        # Actions: load, store, compute, delete
        
        # 1. LOAD: bring a vertex from slow memory to fast
        #    Can load: inputs not in fast, or computed vertices not in fast
        #    Must have space in fast memory
        if len(fast) < fast_memory_budget:
            # Vertices that can be loaded: inputs not in fast, or computed not in fast
            loadable = (inputs | computed) - fast
            for v in loadable:
                if v not in needed_set:
                    continue
                new_fast = fast | {v}
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'load', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
        
        # 2. STORE: move a vertex from fast to slow memory
        #    Can store: any vertex in fast that is either an input or computed
        for v in fast:
            if v in inputs or v in computed:
                new_fast = fast - {v}
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'store', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
        
        # 3. COMPUTE: compute a vertex whose parents are all in fast memory
        for v in needed_set:
            if v in computed:
                continue
            v_parents = [p for p in parent_dict.get(v, []) if p in needed_set]
            if all(p in fast for p in v_parents):
                if len(fast) < fast_memory_budget:
                    new_fast = fast | {v}
                    new_computed = computed | {v}
                    new_state = (new_fast, new_computed)
                    new_cost = cost  # compute doesn't cost a transfer
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent[new_state] = ((fast, computed), 'compute', v)
                        heapq.heappush(pq, (new_cost, new_fast, new_computed))
        
        # 4. DELETE: remove a vertex from fast memory without storing
        for v in fast:
            if v in computed and v not in inputs:
                new_fast = fast - {v}
                new_state = (new_fast, computed)
                new_cost = cost  # delete doesn't cost a transfer
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'delete', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
    
    return None  # No solution found within limits


def _improved_greedy_pebble(needed_vertices, parent_dict, needed_set,
                            inputs, outputs, fast_memory_budget, deadline,
                            vertex_to_idx, children):
    """Improved greedy pebbling with multiple eviction strategies."""
    # Precompute usage positions for each vertex
    use_positions = defaultdict(list)
    for i, v in enumerate(needed_vertices):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                use_positions[p].append(i)
    
    output_set = set(outputs)
    
    # Precompute next use and reuse count helpers
    def get_next_use(u, pos):
        uses = use_positions.get(u, [])
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uses):
            return uses[lo]
        return float('inf')

    def get_reuse_count(u, pos):
        uses = use_positions.get(u, [])
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return len(uses) - lo

    def run_greedy(eviction_strategy):
        in_fast = set()
        in_slow = set(inputs)
        computed = set()
        operations = []
        current_pos = 0
        
        def load(v):
            operations.append(['load', v])
            in_fast.add(v)
            
        def store(v):
            operations.append(['store', v])
            in_fast.discard(v)
            in_slow.add(v)
            
        def compute(v):
            operations.append(['compute', v])
            computed.add(v)
            in_fast.add(v)
            
        def delete(v):
            operations.append(['delete', v])
            in_fast.discard(v)

        def evict_one(protect_set, pos):
            if not in_fast:
                return None
            
            best_evict = None
            best_score = float('-inf')
            
            for u in list(in_fast):
                if u in protect_set:
                    continue
                
                nu = get_next_use(u, pos)
                reuse = get_reuse_count(u, pos)
                
                score = 0.0
                if eviction_strategy == 'LFU':
                    # Least Frequently Used
                    score = -reuse
                    if nu == float('inf'):
                        score += 1000
                    if u in output_set:
                        score -= 1000
                elif eviction_strategy == 'LRU':
                    # Least Recently Used (approximated by next use distance)
                    score = 0.0
                    if nu == float('inf'):
                        score += 1000
                    else:
                        score = nu
                    if u in output_set:
                        score -= 1000
                else: # 'HYBRID' (current)
                    if nu == float('inf'):
                        score += 1000
                    else:
                        score += (nu - pos) * 10
                    score -= reuse * 50
                    if u in output_set:
                        score -= 500
                    if u in inputs:
                        score += 10
                
                if score > best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                for u in in_fast:
                    if u not in output_set:
                        best_evict = u
                        break
                if best_evict is None:
                    best_evict = next(iter(in_fast))
            
            if best_evict is not None:
                if best_evict in in_slow:
                    delete(best_evict)
                else:
                    store(best_evict)
            
            return best_evict

        # Process vertices in topological order
        for v in needed_vertices:
            if time.time() > deadline:
                return None
            
            current_pos = vertex_to_idx.get(v, current_pos)
            parents = [p for p in parent_dict.get(v, []) if p in needed_set]

            for p in parents:
                if p in in_fast:
                    continue
                
                if len(in_fast) >= fast_memory_budget:
                    remaining_parents = set(parents) - {p}
                    evict_one(remaining_parents | {v}, current_pos)
                
                load(p)

            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_one(set(parents) | {v}, current_pos)
                compute(v)

        # Finalize outputs
        for v in outputs:
            if v in needed_set and v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_one(output_set, current_pos)
                
                if v in in_slow or v in computed:
                    load(v)
        
        for v in outputs:
            if v in needed_set and v not in in_fast:
                if len(in_fast) < fast_memory_budget:
                    load(v)
                else:
                    evicted = False
                    for u in list(in_fast):
                        if u not in output_set:
                            if u in in_slow:
                                delete(u)
                            else:
                                store(u)
                            evicted = True
                            break
                    if evicted and v in (in_slow | computed):
                        load(v)

        # Verify
        if not outputs.issubset(in_fast):
            return None
            
        return operations

    # Run multiple strategies and pick the best
    best_ops = None
    best_transfers = float('inf')
    
    strategies = ['HYBRID', 'LFU', 'LRU']
    for strat in strategies:
        if time.time() > deadline - 2:
            break
        ops = run_greedy(strat)
        if ops is not None:
            # Count transfers
            transfers = sum(1 for op, v in ops if op in ['load', 'store'])
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops

    return best_ops if best_ops is not None else []


def _local_refine(operations, needed_vertices, parent_dict, needed_set,
                  inputs, outputs, fast_memory_budget, deadline):
    """
    Try to reduce transfers by swapping adjacent operations or removing
    unnecessary load/store pairs.
    """
    if not operations:
        return operations
    
    # Simulate the schedule to verify it's valid and count transfers
    def simulate(ops):
        in_fast = set()
        in_slow = set(inputs)
        computed = set()
        transfers = 0
        
        for op, v in ops:
            if op == 'load':
                if v not in (in_slow | computed) or v in in_fast:
                    return None, 0  # Invalid
                if len(in_fast) >= fast_memory_budget:
                    return None, 0  # No space
                in_fast.add(v)
                transfers += 1
            elif op == 'store':
                if v not in in_fast:
                    return None, 0
                if v not in inputs and v not in computed:
                    return None, 0
                in_fast.discard(v)
                in_slow.add(v)
                transfers += 1
            elif op == 'compute':
                if v in computed or v in in_fast:
                    return None, 0
                parents = [p for p in parent_dict.get(v, []) if p in needed_set]
                if not all(p in in_fast for p in parents):
                    return None, 0
                if len(in_fast) >= fast_memory_budget:
                    return None, 0
                in_fast.add(v)
                computed.add(v)
            elif op == 'delete':
                if v not in in_fast:
                    return None, 0
                if v in inputs:
                    return None, 0
                in_fast.discard(v)
            else:
                return None, 0
        
        # Check all outputs are in fast
        if not outputs.issubset(in_fast):
            return None, 0
        
        return in_fast, transfers
    
    # Get baseline transfer count
    _, baseline_transfers = simulate(operations)
    if baseline_transfers == 0:
        return operations  # Invalid or no transfers
    
    # Try removing redundant load/store pairs
    def try_remove_redundant(ops):
        new_ops = []
        i = 0
        while i < len(ops):
            if i + 1 < len(ops):
                op1, v1 = ops[i]
                op2, v2 = ops[i + 1]
                if (op1 == 'load' and op2 == 'store' and v1 == v2) or \
                   (op1 == 'store' and op2 == 'load' and v1 == v2):
                    i += 2
                    continue
            new_ops.append(ops[i])
            i += 1
        return new_ops
    
    best_ops = operations
    best_transfers = baseline_transfers
    
    refined = try_remove_redundant(best_ops)
    _, refined_transfers = simulate(refined)
    if refined_transfers > 0 and refined_transfers < best_transfers:
        best_ops = refined
        best_transfers = refined_transfers
    
    # Try swapping adjacent operations
    if len(best_ops) <= 200 and time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            i = rng.randint(0, len(best_ops) - 2)
            candidate = best_ops[:]
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
            
            _, cand_transfers = simulate(candidate)
            if cand_transfers > 0 and cand_transfers < best_transfers:
                best_ops = candidate
                best_transfers = cand_transfers
    
    return best_ops
# EVOLVE-BLOCK-END
