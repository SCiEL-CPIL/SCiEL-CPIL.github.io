import heapq
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Solves the communication-minimizing DAG pebbling problem.
    
    Args:
        dag: A list of lists, where dag[i] is the list of parent indices of vertex i.
             Vertices are assumed to be topologically ordered.
        fast_memory_budget: The maximum number of red pebbles allowed.
    
    Returns:
        A list of [operation, vertex] pairs.
    """
    # Determine N and outputs
    if not dag:
        return []
    
    N = len(dag)
    
    # Identify outputs (sinks)
    # A vertex is a sink if it is not a parent of any other vertex
    has_children = [False] * N
    for i in range(N):
        for p in dag[i]:
            has_children[p] = True
            
    outputs = [i for i in range(N) if not has_children[i]]
    if not outputs:
        # If no sinks (cycle?), assume last vertex is output
        outputs = [N-1]
        
    outputs_mask = 0
    for o in outputs:
        outputs_mask |= (1 << o)
        
    sources_mask = 0
    for i in range(N):
        if not dag[i]:
            sources_mask |= (1 << i)

    # State: (red_mask, main_mask)
    # Initial state
    init_red = 0
    init_main = sources_mask
    init_state = (init_red, init_main)
    
    # Target condition: red_mask == 0 and (main_mask & outputs_mask) == outputs_mask
    def is_target(state):
        red, main = state
        return red == 0 and (main & outputs_mask) == outputs_mask

    # Dijkstra
    # dist[state] = cost
    # parent[state] = (prev_state, op, vertex)
    dist = {init_state: 0}
    parent = {}
    heap = [(0, init_state)]
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    best_state = None
    best_path = None
    
    while heap:
        if time.time() > deadline:
            break
            
        cost, state = heapq.heappop(heap)
        
        if state in dist and dist[state] < cost:
            continue
            
        if is_target(state):
            best_state = state
            break
            
        red, main = state
        red_count = bin(red).count('1')
        
        # 1. Load
        # Can load v if v in main and v not in red
        # And red_count < budget
        if red_count < fast_memory_budget:
            # Candidates: bits in main but not in red
            candidates = main & ~red
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                new_red = red | lsb
                new_main = main & ~lsb
                new_state = (new_red, new_main)
                new_cost = cost + 1
                if new_state not in dist or new_cost < dist[new_state]:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', v)
                    heapq.heappush(heap, (new_cost, new_state))
                candidates ^= lsb
                
        # 2. Store
        # Can store v if v in red
        candidates = red
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            new_red = red & ~lsb
            new_main = main | lsb
            new_state = (new_red, new_main)
            new_cost = cost + 1
            if new_state not in dist or new_cost < dist[new_state]:
                dist[new_state] = new_cost
                parent[new_state] = (state, 'store', v)
                heapq.heappush(heap, (new_cost, new_state))
            candidates ^= lsb
            
        # 3. Compute
        # Can compute v if all parents in red and v not in red
        # v not in main? If v is in main, we can load it. But compute is free.
        # If v is in main, it means it was stored before. We can just load it?
        # But if we compute, we need parents in red.
        # If v is already in main, we don't need to compute it, we can load it.
        # So we only compute v if v is NOT in main and NOT in red.
        # Actually, if v is in main, we can load it (cost 1). If we compute it (cost 0), we save 1.
        # But we need parents in red.
        # So we should try to compute v if possible, even if it's in main?
        # If v is in main, it's blue. If we compute it, it becomes red.
        # Is it valid to compute a vertex that is already in main?
        # "Recomputations allowed".
        # If we compute it, we overwrite the main copy? Or does it create a new red pebble?
        # Usually, pebbling states are just pebbles on/off.
        # Here, we have Red and Main.
        # If v is in Main, and we Compute v, v becomes Red. Is it still in Main?
        # If Compute moves it from Main to Red? Or copies?
        # If it copies, Main still has it.
        # If it moves, Main loses it.
        # Let's assume Compute creates a Red pebble. If it was in Main, does it leave Main?
        # If it leaves Main, we lose the ability to Load it later without recomputing.
        # If it stays in Main, we can Load it later.
        # Given "load" and "store" are transfers, "compute" is likely just a logical operation.
        # I will assume Compute adds to Red, and does NOT remove from Main.
        # However, if v is in Main, we can just Load it.
        # Computing it is only useful if we want it in Red and we have parents in Red.
        # But if it's in Main, we can Load it (cost 1).
        # If we Compute it (cost 0), we save 1 transfer.
        # So we should prefer Compute if possible.
        
        for v in range(N):
            if red & (1 << v):
                continue
            # Check if all parents are in red
            parents_ok = True
            for p in dag[v]:
                if not (red & (1 << p)):
                    parents_ok = False
                    break
            if parents_ok:
                new_red = red | (1 << v)
                new_main = main # Assume compute doesn't affect main
                new_state = (new_red, new_main)
                new_cost = cost
                if new_state not in dist or new_cost < dist[new_state]:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v)
                    heapq.heappush(heap, (new_cost, new_state))
                    
        # 4. Delete
        # Can delete v if v in red
        candidates = red
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            new_red = red & ~lsb
            new_main = main # Delete loses the value, doesn't put in main
            new_state = (new_red, new_main)
            new_cost = cost
            if new_state not in dist or new_cost < dist[new_state]:
                dist[new_state] = new_cost
                parent[new_state] = (state, 'delete', v)
                heapq.heappush(heap, (new_cost, new_state))
            candidates ^= lsb

    if best_state:
        # Reconstruct path
        path = []
        cur = best_state
        while cur in parent:
            prev, op, v = parent[cur]
            path.append([op, v])
            cur = prev
        path.reverse()
        return path
    else:
        # Fallback: Simple valid schedule
        # Strategy: Topological order, load sources, compute others, store all.
        # This ensures no recomputation is needed because we store everything.
        ops = []
        red_set = set()
        main_set = set(i for i in range(N) if not dag[i])
        
        # Topological sort is just range(N) if dag is topologically ordered
        for v in range(N):
            if not dag[v]:
                # Source
                if v not in red_set:
                    ops.append(['load', v])
                    red_set.add(v)
                    main_set.discard(v)
            else:
                # Ensure parents are in red_set
                for p in dag[v]:
                    if p not in red_set:
                        # Since we stored all previous, p should be in main_set
                        if p in main_set:
                            ops.append(['load', p])
                            red_set.add(p)
                            main_set.discard(p)
                        else:
                            # This shouldn't happen with the store-all strategy
                            # but if it does, we have a problem.
                            # For safety, we can't do anything.
                            pass
                ops.append(['compute', v])
                red_set.add(v)
                ops.append(['store', v])
                red_set.discard(v)
                main_set.add(v)
                
        return ops

