"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 50.0  # Wall-clock limit with safety margin (90s total, leave buffer)
    
    def time_exceeded():
        return time.time() - start_time > DEADLINE

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        if not parents and not outputs:
            return []
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order using Kahn's algorithm
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = sorted([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    queue.sort()
        
        if len(topo_order) != len(all_vertices):
            remaining = all_vertices - set(topo_order)
            topo_order.extend(sorted(remaining))
        
        topo_index = {v: i for i, v in enumerate(topo_order)}
        n = len(topo_order)
        
        # Pre-compute static properties
        # 1. Recomputation cost (number of ancestors)
        recomp_cost = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                recomp_cost[v] = 1 # Cost to load an input is 1 transfer
            else:
                # Approximate cost as max cost of parents + 1 (compute op)
                # Or sum? Sum is a better proxy for total work if we have to redo everything.
                # But memory constraint means we might not redo everything at once.
                # Let's use the number of ancestors as a proxy for "expensiveness".
                max_anc = 0
                for p in ps:
                    if p in recomp_cost:
                        max_anc = max(max_anc, recomp_cost[p])
                recomp_cost[v] = max_anc + 1
        
        # 2. Next use and Last use
        next_use = {}
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                uses = [topo_index[c] for c in children[v]]
                next_use[v] = min(uses)
                last_use[v] = max(uses)
            else:
                next_use[v] = n # Never used again (or leaf)
                last_use[v] = topo_index.get(v, n - 1)
        
        # Outputs must be kept until the end
        for o in outputs:
            last_use[o] = n - 1
            if o in children and children[o]:
                 # If output has children (shouldn't usually, but just in case)
                 pass
            else:
                 next_use[o] = n-1 # Effectively needed at the end

        # State
        fast_mem = set()
        ops = []
        
        def get_keep_score(v, current_idx):
            """
            Calculate how valuable it is to keep vertex v in fast memory.
            Higher score = more valuable to keep.
            """
            if v in outputs:
                return 10000 # Keep outputs always if possible
            
            # Distance to next use
            dist = next_use[v] - current_idx
            if dist < 0:
                dist = 0
            
            # If it's not used again, score is low (unless it's an output)
            if next_use[v] >= n:
                return recomp_cost[v] * 0.1 # Cheap to evict if not needed again
            
            # Value components:
            # 1. Expensive to recompute? (High recomp_cost -> High value)
            # 2. Needed soon? (Low dist -> High value, because evicting now forces immediate reload)
            # 3. Frequency? (Not easily computed dynamically, skip for simplicity)
            
            # Heuristic:
            # If dist is small, we want to keep it.
            # If recomp_cost is high, we want to keep it.
            
            # Smooth surrogate for "urgency": 1 / (1 + dist)
            urgency = 1.0 / (1.0 + dist)
            
            # Normalize recomp cost
            max_recomp = max(recomp_cost.values()) if recomp_cost else 1
            norm_recomp = recomp_cost[v] / max_recomp
            
            # Weighted sum
            # We prioritize keeping things that are expensive to recompute AND needed soon.
            score = (norm_recomp * 10.0) + (urgency * 5.0)
            
            # Bonus if it's on the critical path to the current vertex?
            # Hard to determine dynamically without path tracking.
            
            return score
        
        def make_space(needed=1, current_idx=0):
            """Free up 'needed' slots in fast_mem by evicting vertices with lowest keep score."""
            evicted = 0
            while len(fast_mem) + needed > fast_memory_budget and evicted < needed:
                if not fast_mem:
                    break
                
                # Find candidate with lowest keep score
                best_candidate = None
                best_score = float('inf')
                
                for v in fast_mem:
                    s = get_keep_score(v, current_idx)
                    if s < best_score:
                        best_score = s
                        best_candidate = v
                
                if best_candidate is None:
                    break
                
                ops.append(['delete', best_candidate])
                fast_mem.discard(best_candidate)
                evicted += 1
            
            return evicted >= needed
        
        def ensure_in_fast_mem(v, current_idx):
            """
            Ensure vertex v is in fast_mem.
            Returns True if successful.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            if not ps:
                # Input vertex
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space(1, current_idx):
                        return False
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex: need to recompute it
                # First ensure all parents are in fast_mem
                for p in ps:
                    if p not in fast_mem:
                        if not ensure_in_fast_mem(p, current_idx):
                            return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space(1, current_idx):
                        return False
                
                ops.append(['compute', v])
                fast_mem.add(v)
                return True
        
        # Main loop: process vertices in topological order
        for i, v in enumerate(topo_order):
            if time_exceeded():
                break
            
            if v not in fast_mem:
                if not ensure_in_fast_mem(v, i):
                    break
            
            # Proactively delete vertices that are no longer needed
            # A vertex u is no longer needed if last_use[u] < i and u is not an output
            # We do this to keep memory clean for future needs
            to_delete = []
            for u in list(fast_mem):
                if u != v and u not in outputs and last_use[u] < i:
                    to_delete.append(u)
            
            for u in to_delete:
                ops.append(['delete', u])
                fast_mem.discard(u)
            
            # Optional: Proactive loading of upcoming expensive vertices
            # Look ahead a few steps
            # If we have space, and a vertex in the next few steps has high recomp_cost, load it now?
            # This is risky as it might evict something more important.
            # Given the greedy nature, we rely on the ensure_in_fast_mem to handle it.
            
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                if not ensure_in_fast_mem(o, n - 1):
                    # Force evict if needed
                    if len(fast_mem) >= fast_memory_budget:
                        # Evict lowest score
                        best_candidate = None
                        best_score = float('inf')
                        for v in fast_mem:
                            s = get_keep_score(v, n-1)
                            if s < best_score:
                                best_score = s
                                best_candidate = v
                        if best_candidate:
                            ops.append(['delete', best_candidate])
                            fast_mem.discard(best_candidate)
                    ensure_in_fast_mem(o, n - 1)
        
        # Validate: all outputs must be in fast_mem
        for o in outputs:
            if o not in fast_mem:
                return []
        
        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]
            
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
