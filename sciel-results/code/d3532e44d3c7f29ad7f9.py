"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _topological_order(dag):
    """Extract vertices and compute topological order."""
    if isinstance(dag, dict):
        vertices = list(dag.keys())
        parents_map = {v: list(p) for v, p in dag.items()}
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            parents_map = {}
            vertices = []
            for item in dag:
                v = item[0]
                ps = list(item[1]) if len(item) > 1 else []
                parents_map[v] = ps
                vertices.append(v)
        else:
            vertices = list(dag)
            parents_map = {v: [] for v in vertices}
    else:
        vertices = list(dag) if hasattr(dag, '__iter__') else [dag]
        parents_map = {v: [] for v in vertices}
    
    in_degree = {v: 0 for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map[v]:
            children[p].append(v)
            in_degree[v] += 1
    
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        queue.sort()
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    remaining = [v for v in vertices if v not in set(topo_order)]
    topo_order.extend(remaining)
    
    return topo_order, parents_map, vertices

def _compute_depths(topo_order, parents_map):
    depth = {}
    for v in topo_order:
        if not parents_map[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents_map[v]) + 1
    return depth

def _compute_heights(topo_order, parents_map, vertices):
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map[v]:
            children[p].append(v)
    height = {}
    for v in reversed(topo_order):
        if not children[v]:
            height[v] = 0
        else:
            height[v] = max(height[c] for c in children[v]) + 1
    return height

def _simulate_schedule(schedule, dag, fast_memory_budget, max_ops=20000):
    topo_order, parents_map, vertices = _topological_order(dag)
    
    state = {v: None for v in vertices}
    fast_mem = set()
    disk = set()
    transfers = 0
    ops_count = 0
    
    # Inputs (sources) start blue
    for v in vertices:
        if not parents_map[v]:
            fast_mem.add(v)
            state[v] = 'blue'
    
    for op, v in schedule:
        if ops_count >= max_ops:
            return transfers, False, "Too many operations"
        ops_count += 1
        
        if op == 'load':
            if v not in disk:
                return transfers, False, f"Load {v}: not on disk"
            if v in fast_mem:
                return transfers, False, f"Load {v}: already in fast memory"
            if len(fast_mem) >= fast_memory_budget:
                return transfers, False, f"Load {v}: fast memory full"
            fast_mem.add(v)
            disk.discard(v)
            transfers += 1
            state[v] = 'blue'
            
        elif op == 'store':
            if v not in fast_mem:
                return transfers, False, f"Store {v}: not in fast memory"
            fast_mem.discard(v)
            disk.add(v)
            transfers += 1
            state[v] = 'red'
            
        elif op == 'compute':
            for p in parents_map[v]:
                if state[p] is None:
                    return transfers, False, f"Compute {v}: parent {p} not computed"
            if len(fast_mem) >= fast_memory_budget:
                return transfers, False, f"Compute {v}: fast memory full"
            fast_mem.add(v)
            state[v] = 'blue'
            
        elif op == 'delete':
            if v not in fast_mem:
                return transfers, False, f"Delete {v}: not in fast memory"
            fast_mem.discard(v)
            state[v] = None
            
        else:
            return transfers, False, f"Unknown op {op}"
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map[v]:
            children[p].append(v)
    
    outputs = [v for v in vertices if not children[v]]
    for v in outputs:
        if state[v] != 'blue':
            return transfers, False, f"Output {v} not blue at end"
    
    return transfers, True, None

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0
    
    topo_order, parents_map, vertices = _topological_order(dag)
    depth = _compute_depths(topo_order, parents_map)
    height = _compute_heights(topo_order, parents_map, vertices)
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map[v]:
            children[p].append(v)
    
    usage_count = {v: len(children[v]) for v in vertices}
    
    schedule = []
    fast_mem = set()
    disk = set()
    computed = set()
    transfers = 0
    
    # Sources start in fast memory
    for v in vertices:
        if not parents_map[v]:
            fast_mem.add(v)
            computed.add(v)
    
    def keep_priority(v):
        return (usage_count[v], height[v], -depth[v])
    
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        # Skip if already computed (shouldn't happen in topo order)
        if v in computed:
            continue
        
        # Ensure parents are in fast memory
        for p in parents_map[v]:
            if p in computed and p not in fast_mem:
                if p in disk:
                    if len(fast_mem) >= fast_memory_budget:
                        needed_parents = set(parents_map[v]) - {p}
                        candidates = [x for x in fast_mem if x not in needed_parents]
                        if not candidates:
                            candidates = list(fast_mem)
                        if candidates:
                            candidates.sort(key=keep_priority)
                            to_store = candidates[0]
                            schedule.append(('store', to_store))
                            fast_mem.discard(to_store)
                            disk.add(to_store)
                            transfers += 1
                    
                    schedule.append(('load', p))
                    fast_mem.add(p)
                    disk.discard(p)
                    transfers += 1
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            remaining = set(topo_order[idx+1:])
            for to_store in list(fast_mem):
                if to_store not in remaining:
                    schedule.append(('store', to_store))
                    fast_mem.discard(to_store)
                    disk.add(to_store)
                    transfers += 1
                    break
            if len(fast_mem) >= fast_memory_budget:
                candidates = list(fast_mem)
                candidates.sort(key=keep_priority)
                if candidates:
                    to_store = candidates[0]
                    schedule.append(('store', to_store))
                    fast_mem.discard(to_store)
                    disk.add(to_store)
                    transfers += 1
        
        schedule.append(('compute', v))
        fast_mem.add(v)
        computed.add(v)
        
        # Heuristic: store v to disk if it's low priority and not needed soon
        if fast_memory_budget > 1 and usage_count[v] > 0:
            all_in_mem = list(fast_mem)
            all_in_mem.sort(key=keep_priority)
            if len(fast_mem) >= fast_memory_budget * 0.7:
                if all_in_mem.index(v) < len(all_in_mem) * 0.3:
                    next_vertices = set(topo_order[idx+1 : idx+10])
                    if v not in next_vertices:
                        schedule.append(('store', v))
                        fast_mem.discard(v)
                        disk.add(v)
                        transfers += 1
    
    # Ensure outputs are in fast memory
    for v in topo_order:
        if time.time() > deadline:
            break
        if not children[v] and v in computed and v not in fast_mem:
            if v in disk:
                if len(fast_mem) >= fast_memory_budget:
                    for to_store in list(fast_mem):
                        if to_store not in [x for x in topo_order if not children[x]]:
                            schedule.append(('store', to_store))
                            fast_mem.discard(to_store)
                            disk.add(to_store)
                            transfers += 1
                            break
                schedule.append(('load', v))
                fast_mem.add(v)
                disk.discard(v)
                transfers += 1
    
    num_transfers, is_valid, error = _simulate_schedule(schedule, dag, fast_memory_budget)
    
    if not is_valid:
        # Fallback: simple topological order with minimal stores
        fallback = []
        fast_mem2 = set()
        disk2 = set()
        computed2 = set()
        for v in vertices:
            if not parents_map[v]:
                fast_mem2.add(v)
                computed2.add(v)
        
        for v in topo_order:
            if v in computed2:
                continue
            for p in parents_map[v]:
                if p in computed2 and p not in fast_mem2:
                    if p in disk2:
                        if len(fast_mem2) >= fast_memory_budget:
                            for to_store in list(fast_mem2):
                                fallback.append(('store', to_store))
                                fast_mem2.discard(to_store)
                                disk2.add(to_store)
                                break
                        fallback.append(('load', p))
                        fast_mem2.add(p)
                        disk2.discard(p)
            fallback.append(('compute', v))
            fast_mem2.add(v)
            computed2.add(v)
        
        # Ensure outputs
        for v in topo_order:
            if not children[v] and v in computed2 and v not in fast_mem2:
                if v in disk2:
                    if len(fast_mem2) >= fast_memory_budget:
                        for to_store in list(fast_mem2):
                            fallback.append(('store', to_store))
                            fast_mem2.discard(to_store)
                            disk2.add(to_store)
                            break
                    fallback.append(('load', v))
                    fast_mem2.add(v)
                    disk2.discard(v)
        
        return fallback
    
    return schedule
# EVOLVE-BLOCK-END

