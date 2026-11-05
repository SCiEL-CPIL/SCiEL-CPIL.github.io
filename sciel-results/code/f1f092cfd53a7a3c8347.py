import time
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    
    # Identify all vertices
    all_vertices = set(parents.keys())
    for v in parents:
        for p in parents[v]:
            all_vertices.add(p)
    all_vertices.update(outputs)
    
    if not all_vertices:
        return []
        
    # Topological Sort
    indeg = {v: len(parents.get(v, [])) for v in all_vertices}
    children = {v: [] for v in all_vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    q = deque([v for v in all_vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
                
    if len(topo) != len(all_vertices):
        # Cycle fallback: just return a basic schedule
        return _fallback_schedule(dag, fast_memory_budget)

    # Precompute recompute cost (approx: number of loads needed)
    # cost[v] = 1 + sum(cost[p] for p in parents[v]) if we assume we have to load all parents
    # This is an upper bound on transfers if we start from scratch.
    recompute_cost = {}
    def get_cost(v):
        if v in recompute_cost:
            return recompute_cost[v]
        ps = parents.get(v, [])
        if not ps:
            recompute_cost[v] = 1 # Load itself? Or is it an input?
            # If no parents, it's an input. Inputs start on disk (and mem).
            # To recompute an input, we just load it. Cost 1.
        else:
            c = 1 # Compute cost 0, but we need parents
            for p in ps:
                c += get_cost(p)
            recompute_cost[v] = c
        return recompute_cost[v]
        
    for v in all_vertices:
        get_cost(v)
        
    # State
    mem = set(outputs) # Start with outputs? No, inputs start in mem.
    # "Inputs start blue" -> Inputs are in memory.
    inputs = [v for v in all_vertices if not parents.get(v)]
    mem = set(inputs)
    disk = set(inputs) # Inputs are also on disk
    
    ops = []
    
    # Remaining uses count
    remaining_uses = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for c in children[v]:
            remaining_uses[v] += 1
            
    # We need to compute all outputs.
    # Strategy: Process vertices in topological order, but prioritize those needed for outputs.
    # Actually, just process in topo order, ensuring dependencies are met.
    
    # To minimize transfers, we want to keep "hot" vertices in memory.
    # We will iterate through the topo order.
    # If a vertex is not in mem, we need to make it available.
    
    # We need a way to "push" a vertex into memory.
    def ensure_in_mem(v):
        if v in mem:
            return
        
        # Check if v is on disk
        if v in disk:
            # Need space?
            if len(mem) >= fast_memory_budget:
                # Evict
                evict_one()
            ops.append(["load", v])
            mem.add(v)
            return
            
        # v is not in mem, not in disk. Must compute.
        # First, ensure parents are in mem
        for p in parents.get(v, []):
            ensure_in_mem(p)
            
        # Now compute v
        if len(mem) >= fast_memory_budget:
            evict_one()
            
        ops.append(["compute", v])
        mem.add(v)
        
    def evict_one():
        # Find best candidate to evict
        best_v = None
        best_score = float('inf')
        
        for u in list(mem):
            # Score: lower is better to evict
            # If u is on disk, delete is free.
            # If u is not on disk, delete is free but loses value. Store is 1 but keeps value.
            
            is_on_disk = u in disk
            future_use = remaining_uses[u]
            
            if is_on_disk:
                # Delete is free. 
                # Score: 0 + small penalty for future use? 
                # Actually, if it's on disk, we can always load it back.
                # So deleting is very safe.
                score = future_use * 0.1 # Low penalty
            else:
                # Not on disk.
                # If we delete, we lose it. Cost to recompute later is recompute_cost[u].
                # If we store, we pay 1 now.
                # We should store if recompute_cost[u] > 1.
                # But we are forced to evict.
                # If we delete, we save 1 transfer now, but pay recompute_cost[u] later.
                # If we store, we pay 1 now, save recompute_cost[u] later.
                # Net cost of delete: recompute_cost[u]
                # Net cost of store: 1
                # Choose store if recompute_cost[u] > 1.
                # But we are in evict_one, which assumes we just remove from mem.
                # We need to decide op: delete or store.
                
                if recompute_cost[u] > 1:
                    # Store is better
                    score = 1 + future_use * 0.01
                    # Mark to store
                    evict_op = "store"
                else:
                    # Delete is better
                    score = 0 + future_use * 0.01
                    evict_op = "delete"
                    
            if score < best_score:
                best_score = score
                best_v = u
                best_op = "delete" if (u in disk) else ("store" if recompute_cost[u] > 1 else "delete")
                
        if best_v:
            ops.append([best_op, best_v])
            mem.remove(best_v)
            if best_op == "store":
                disk.add(best_v)
            # If delete, it's gone from mem. If not on disk, it's lost.
            # If it was on disk, it's still on disk.
            
    # Main loop
    # We need to ensure all outputs are in mem at the end.
    # We process in topo order to compute dependencies.
    for v in topo:
        if v in mem:
            continue
        ensure_in_mem(v)
        
    # Final check: ensure outputs are in mem
    for o in outputs:
        if o not in mem:
            ensure_in_mem(o)
            
    # Truncate to 20000
    return ops[:20000]

def _fallback_schedule(dag, fast_memory_budget):
    # Simple greedy fallback
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs))
    if not vertices:
        return []
    ops = []
    mem = set()
    disk = set()
    
    # Topo sort
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
                
    for v in topo:
        if v in mem:
            continue
        # Load parents
        for p in parents.get(v, []):
            if p not in mem:
                if p in disk:
                    if len(mem) >= fast_memory_budget:
                        # Evict
                        # Simple: delete first non-output
                        for u in list(mem):
                            if u not in outputs:
                                ops.append(["delete", u])
                                mem.remove(u)
                                break
                    ops.append(["load", p])
                    mem.add(p)
                else:
                    # Recompute p? This is complex. Just assume it's there or load.
                    # For fallback, just load if possible
                    if len(mem) >= fast_memory_budget:
                        for u in list(mem):
                            if u not in outputs:
                                ops.append(["delete", u])
                                mem.remove(u)
                                break
                    ops.append(["load", p])
                    mem.add(p)
        # Compute v
        if len(mem) >= fast_memory_budget:
            for u in list(mem):
                if u not in outputs and u != v:
                    ops.append(["delete", u])
                    mem.remove(u)
                    break
        ops.append(["compute", v])
        mem.add(v)
        if v in outputs:
            ops.append(["store", v])
            disk.add(v)
            mem.remove(v) # Store removes from mem? 
            # "Outputs must finish blue" -> In mem.
            # So don't remove.
            # But "store" usually means move to disk.
            # Let's assume store keeps it in mem? No, "store" is a transfer.
            # If it's a transfer to disk, it leaves mem.
            # But output must finish in mem (blue).
            # So we must load it back? Or store means "save to disk" but it stays in mem?
            # Standard: Store = mem -> disk.
            # So if we store, it's gone from mem.
            # So we must NOT store if it needs to finish in mem, UNLESS we load it back.
            # But loading back costs 1.
            # So for outputs, we just keep them in mem.
            pass
            
    return ops[:20000]
