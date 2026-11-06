from collections import defaultdict

def pebble(dag, fast_memory_budget):
    N = len(dag)
    if N == 0:
        return []
        
    parents = dag
    children = [[] for _ in range(N)]
    for i in range(N):
        for p in parents[i]:
            children[p].append(i)
            
    # Compute depths
    depth = [-1] * N
    for i in range(N):
        if len(parents[i]) == 0:
            depth[i] = 0
        else:
            depth[i] = max(depth[p] for p in parents[i]) + 1
            
    max_depth = max(depth) if N > 0 else 0
    depth_groups = defaultdict(list)
    for i in range(N):
        depth_groups[depth[i]].append(i)
        
    in_fast = set()
    in_slow = set(i for i in range(N) if len(parents[i]) == 0)
    ops = []
    
    prev_layer = []
    
    for d in range(max_depth + 1):
        curr_layer = depth_groups[d]
        
        if not curr_layer:
            continue
            
        # Calculate chunk size
        # We keep prev_layer in Fast.
        # We add curr_chunk to Fast.
        # Total <= B
        if not prev_layer:
            chunk_size = fast_memory_budget
        else:
            # If prev_layer size >= B, we can't add anything.
            # But we must process.
            # This implies we need to store prev_layer first?
            # But we need prev_layer to compute curr_layer.
            # So if |prev| >= B, we are stuck unless we recompute?
            # Assuming |prev| < B for solvability.
            available = fast_memory_budget - len(prev_layer)
            if available < 1:
                # We must store some of prev_layer?
                # But we need them.
                # This case should not happen for valid instances.
                chunk_size = 1
            else:
                chunk_size = available
                
        for i in range(0, len(curr_layer), chunk_size):
            chunk = curr_layer[i:i+chunk_size]
            for v in chunk:
                # Load parents
                for p in parents[v]:
                    if p not in in_fast:
                        if p in in_slow:
                            ops.append(['load', p])
                            in_fast.add(p)
                        else:
                            # Should not happen
                            pass
                ops.append(['compute', v])
                in_fast.add(v)
                
        # Store prev_layer
        if prev_layer:
            for p in prev_layer:
                if p in in_fast:
                    if p not in in_slow:
                        ops.append(['store', p])
                        in_slow.add(p)
                    ops.append(['delete', p])
                    in_fast.remove(p)
                    
        prev_layer = curr_layer
        
    # Finalize
    for v in prev_layer:
        if v in in_fast:
            if v not in in_slow:
                ops.append(['store', v])
                in_slow.add(v)
            ops.append(['delete', v])
            in_fast.remove(v)
            
    return ops
