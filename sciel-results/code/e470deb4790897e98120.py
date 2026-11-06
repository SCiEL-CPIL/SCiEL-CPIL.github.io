"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    if not input_vertices:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Build vertex index for bitset representation
    vertex_list = sorted(all_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    n_vertices = len(vertex_list)
    
    # Precompute bitmasks
    input_mask = 0
    for v in input_vertices:
        input_mask |= (1 << vertex_to_idx[v])
    
    output_mask = 0
    for v in outputs:
        output_mask |= (1 << vertex_to_idx[v])
    
    # Precompute parent masks for each vertex
    parent_masks = {}
    for v in all_vertices:
        mask = 0
        for p in parents.get(v, []):
            mask |= (1 << vertex_to_idx[p])
        parent_masks[v] = mask
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    q = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while q:
        v = q.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                q.append(c)
    
    # For each vertex, compute which vertices in topo_order come after it
    # and need it as a parent (i.e., it's a transitive dependency)
    # Actually, we need: for each vertex u, the set of vertices w such that
    # u is in parents(w) and w comes after u in topo order (i.e., w is uncomputed
    # when u is in fast memory)
    # Simpler: for each u, the set of direct children that come after u
    needed_by_mask = {}
    for u in all_vertices:
        mask = 0
        for w in children[u]:
            # w is a child of u, so u is a parent of w
            # If w hasn't been computed yet (comes after u in topo), u is needed
            mask |= (1 << vertex_to_idx[w])
        needed_by_mask[u] = mask

    # Greedy Solution
    def generate_greedy():
        slow_mem = set(input_vertices)
        fast_mem = set()
        ops = []
        
        computed = set()
        for v in topo_order:
            if v in input_vertices:
                continue
            
            # Load parents
            for p in parents.get(v, []):
                if p not in fast_mem:
                    if p in slow_mem:
                        ops.append(['load', p])
                        fast_mem.add(p)
                        slow_mem.discard(p)
                    else:
                        pass
            
            # Compute
            ops.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)
            
            # If output, store immediately
            if v in outputs:
                ops.append(['store', v])
                slow_mem.add(v)
                fast_mem.discard(v)
                
            # Evict if over budget
            while len(fast_mem) > fast_memory_budget:
                evict_candidate = None
                best_score = -1
                
                for u in fast_mem:
                    score = 0
                    if u in outputs and u not in slow_mem:
                        score -= 10000
                    elif u in outputs:
                        score += 100
                    else:
                        # Check if u is a parent of any uncomputed vertex
                        needed = False
                        for w in children[u]:
                            if w not in computed:
                                needed = True
                                break
                        if needed:
                            score -= 100
                        else:
                            score += 100
                
                    if score > best_score:
                        best_score = score
                        evict_candidate = u
                        
                if evict_candidate is None:
                    for u in fast_mem:
                        if u not in outputs:
                            evict_candidate = u
                            break
                    if evict_candidate is None:
                        evict_candidate = next(iter(fast_mem))
                        
                ops.append(['delete', evict_candidate])
                fast_mem.discard(evict_candidate)

        for o in outputs:
            if o in fast_mem:
                ops.append(['store', o])
                slow_mem.add(o)
                fast_mem.discard(o)
                
        return ops

    best_ops = generate_greedy()
    best_cost = sum(1 for op in best_ops if op[0] in ['load', 'store'])
    
    # A* Search with bitset states
    # State: (F_mask, S_mask) where F_mask = fast memory, S_mask = slow memory (stored)
    # Note: inputs start in S, so initial S = input_mask
    # Goal: all outputs in S, i.e., (S_mask & output_mask) == output_mask
    
    # Stronger heuristic: minimum number of loads needed
    # For each output not yet in S, we need to eventually store it, which requires
    # computing it, which requires loading its parents if not in S.
    # Lower bound: count of outputs not in S, plus some dependency-based estimate.
    
    def heuristic(F_mask, S_mask):
        # Count outputs not in S
        outputs_not_in_S = output_mask & ~S_mask
        if outputs_not_in_S == 0:
            return 0
        
        # For each output not in S, we need at least 1 store operation.
        # Additionally, if its parents aren't all in S or F, we may need loads.
        # Simple admissible heuristic: number of outputs not in S
        # (each requires at least one store, and stores cost 1)
        return bin(outputs_not_in_S).count('1')
    
    # Even stronger: consider that to compute a vertex, all parents must be in F.
    # If a parent is not in S and not in F, it must be loaded (cost 1) or recomputed.
    # For now, use the simple heuristic but improve neighbor generation.
    
    start_F = 0
    start_S = input_mask
    
    # Heap: (priority, cost, F_mask, S_mask, path)
    heap = [(heuristic(start_F, start_S), 0, start_F, start_S, [])]
    visited = {}  # (F_mask, S_mask) -> min_cost
    
    # Precompute which vertices are computable given a state
    # A vertex v is computable if:
    #   v not in F, v not in inputs, and all parents of v are in F
    
    # For efficiency, precompute for each vertex its parent mask
    # and check (parent_masks[v] & ~F_mask) == 0
    
    max_expansions = 20000
    expansions = 0
    last_time_check = start_time
    time_check_interval = 0.5  # Check time every 0.5 seconds
    
    while heap and expansions < max_expansions:
        # Time check with interval
        if time.time() - last_time_check > time_check_interval:
            last_time_check = time.time()
            if time_check():
                break
        
        priority, cost, F_mask, S_mask, path = heapq.heappop(heap)
        expansions += 1
        
        key = (F_mask, S_mask)
        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost
        
        # Check goal: all outputs in S
        if (S_mask & output_mask) == output_mask:
            if cost < best_cost:
                best_cost = cost
                best_ops = list(path)
            continue
        
        # Generate neighbors efficiently
        
        # 1. Compute (Cost 0): vertices whose parents are all in F
        #    and which are not in F and not inputs
        #    Also, the result must fit in F (but we handle eviction via delete)
        #    Actually, compute adds to F, so we need len(F) < budget or we'll need to delete
        #    For now, allow compute and let delete handle overflow
        
        # Find computable vertices
        for v in all_vertices:
            if v in input_vertices:
                continue
            v_idx = vertex_to_idx[v]
            if F_mask & (1 << v_idx):
                continue  # Already in F
            # Check parents all in F
            p_mask = parent_masks[v]
            if p_mask & ~F_mask:
                continue  # Some parent not in F
            
            # Can compute v
            new_F = F_mask | (1 << v_idx)
            new_S = S_mask
            new_cost = cost
            new_path = path + [['compute', v]]
            new_key = (new_F, new_S)
            if new_key not in visited or visited[new_key] > new_cost:
                heapq.heappush(heap, (new_cost + heuristic(new_F, new_S), new_cost, new_F, new_S, new_path))
        
        # 2. Delete (Cost 0): remove from F if not needed
        #    Can delete if v is in S (safe) or v is not an output (will recompute if needed)
        #    Actually, we can always delete non-output vertices.
        #    For outputs, only delete if already in S.
        F_set = set()
        fm = F_mask
        while fm:
            lsb = fm & (-fm)
            idx = lsb.bit_length() - 1
            F_set.add(vertex_list[idx])
            fm ^= lsb
        
        for v in F_set:
            v_idx = vertex_to_idx[v]
            # Can delete if v in S or v not an output
            if (S_mask & (1 << v_idx)) or not (output_mask & (1 << v_idx)):
                new_F = F_mask & ~(1 << v_idx)
                new_S = S_mask
                new_cost = cost
                new_path = path + [['delete', v]]
                new_key = (new_F, new_S)
                if new_key not in visited or visited[new_key] > new_cost:
                    heapq.heappush(heap, (new_cost + heuristic(new_F, new_S), new_cost, new_F, new_S, new_path))
        
        # 3. Load (Cost 1): move from S to F
        #    v in S, not in F
        S_set = set()
        sm = S_mask
        while sm:
            lsb = sm & (-sm)
            idx = lsb.bit_length() - 1
            S_set.add(vertex_list[idx])
            sm ^= lsb
        
        for v in S_set:
            v_idx = vertex_to_idx[v]
            if F_mask & (1 << v_idx):
                continue  # Already in F
            new_F = F_mask | (1 << v_idx)
            new_S = S_mask & ~(1 << v_idx)
            new_cost = cost + 1
            new_path = path + [['load', v]]
            new_key = (new_F, new_S)
            if new_key not in visited or visited[new_key] > new_cost:
                heapq.heappush(heap, (new_cost + heuristic(new_F, new_S), new_cost, new_F, new_S, new_path))
        
        # 4. Store (Cost 1): move from F to S
        for v in F_set:
            v_idx = vertex_to_idx[v]
            if S_mask & (1 << v_idx):
                continue  # Already in S
            new_F = F_mask & ~(1 << v_idx)
            new_S = S_mask | (1 << v_idx)
            new_cost = cost + 1
            new_path = path + [['store', v]]
            new_key = (new_F, new_S)
            if new_key not in visited or visited[new_key] > new_cost:
                heapq.heappush(heap, (new_cost + heuristic(new_F, new_S), new_cost, new_F, new_S, new_path))

    return best_ops

# EVOLVE-BLOCK-END
