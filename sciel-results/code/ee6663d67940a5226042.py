"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = list(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    v_index = {v: i for i, v in enumerate(vertices)}
    
    # Compute depth for layering
    depth = {}
    for v in vertices:
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    
    max_depth = max(depth.values()) if depth else 0
    layers = defaultdict(list)
    for v in vertices:
        layers[depth[v]].append(v)
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    fast_mem = set()
    slow_mem = set()
    for v in vertices:
        if not parents[v]:
            slow_mem.add(v)
    
    ops = []
    op_count = 0
    MAX_OPS = 20000
    
    def emit(op, v):
        nonlocal op_count
        ops.append([op, v])
        op_count += 1
    
    def can_add():
        return len(fast_mem) < fast_memory_budget
    
    def load(v):
        if v in fast_mem:
            return True
        if v not in slow_mem:
            return False
        if not can_add():
            return False
        emit('load', v)
        fast_mem.add(v)
        return True
    
    def store(v):
        if v not in fast_mem:
            return False
        emit('store', v)
        slow_mem.add(v)
        fast_mem.discard(v)
        return True
    
    def delete(v):
        if v not in fast_mem:
            return False
        emit('delete', v)
        fast_mem.discard(v)
        return True
    
    def compute(v):
        if v in fast_mem:
            return True
        for p in parents[v]:
            if p not in fast_mem:
                return False
        emit('compute', v)
        fast_mem.add(v)
        return True
    
    def free_space_for(target_set):
        """Free fast memory space by deleting/storing items not in target_set."""
        for v2 in list(fast_mem):
            if can_add():
                return
            if v2 not in target_set:
                if v2 in outputs:
                    store(v2)
                else:
                    delete(v2)
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() > DEADLINE:
            break
        if d not in layers:
            continue
        
        layer = layers[d]
        
        # Parents needed for this layer
        needed = set()
        for v in layer:
            for p in parents[v]:
                needed.add(p)
        
        # Load needed parents
        for p in sorted(needed, key=lambda x: v_index[x]):
            if p in fast_mem:
                continue
            if p in slow_mem:
                free_space_for(needed)
                load(p)
        
        # Compute layer vertices
        for v in layer:
            if time.time() > DEADLINE or op_count >= MAX_OPS:
                break
            if v in fast_mem:
                continue
            for p in parents[v]:
                if p not in fast_mem and p in slow_mem:
                    free_space_for(set(parents[v]))
                    load(p)
            compute(v)
        
        # Cleanup: keep only what's needed by future layers or is output
        future_needed = set()
        for d2 in range(d + 1, max_depth + 1):
            if d2 in layers:
                for v in layers[d2]:
                    for p in parents[v]:
                        future_needed.add(p)
        
        for v in list(fast_mem):
            if v in future_needed or v in outputs:
                if len(fast_mem) > fast_memory_budget:
                    store(v)
            else:
                if v in outputs:
                    store(v)
                else:
                    delete(v)
    
    # Ensure outputs are blue
    for v in outputs:
        if v in fast_mem:
            store(v)
    
    return ops

# EVOLVE-BLOCK-END
