"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _pebble_greedy_cost(dag, budget):
    """
    Greedy pebbling with cost-based eviction.
    Minimizes transfers (loads + stores) by evicting vertices with lowest recomputation/load cost.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    
    # 1. Precompute recompute cost C_v
    # C_v = 1 + sum(C_p for p in parents(v))
    # If no parents, C_v = 1 (cost to load)
    C = [0] * n
    def calc_cost(v):
        i = idx[v]
        if C[i] != 0:
            return C[i]
        ps = parents.get(v, [])
        if not ps:
            C[i] = 1
        else:
            s = 0
            for p in ps:
                s += calc_cost(p)
            C[i] = 1 + s
        return C[i]
    
    for v in vertices:
        calc_cost(v)
        
    # Topological sort
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
                
    if len(topo) != n:
        # Cycle fallback
        topo = vertices
        
    outset = set(outputs)
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # State
    in_memory = set()
    on_disk = set()
    ops = []
    
    # Initialize memory
    # "Inputs start blue"
    # If budget allows, put all inputs in memory.
    # Otherwise, put the most important ones? Or just as many as fit.
    # Let's put as many inputs as fit, prioritizing those with higher C (though all are 1).
    # Actually, if we don't put them in memory, we load them later.
    # Loading an input costs 1.
    # If we start with them in memory, cost 0.
    # So we should start with as many as possible.
    if len(inputs) <= budget:
        in_memory = set(inputs)
    else:
        # Pick first 'budget' inputs?
        # Better: Pick inputs that are parents of high-cost vertices?
        # For simplicity, just take first 'budget'.
        in_memory = set(inputs[:budget])
        
    # Vertices not in memory are considered "on disk" if they are inputs or computed.
    # For inputs not in memory, they are on disk.
    for v in inputs:
        if v not in in_memory:
            on_disk.add(v)
            
    # Track which vertices are "done" (outputs that are in memory)
    # We need all outputs in memory at the end.
    
    # Helper: is v ready to compute?
    def is_ready(v):
        if v in in_memory:
            return False
        ps = parents.get(v, [])
        if not ps:
            return False # Inputs are handled separately
        return all(p in in_memory for p in ps)
        
    # Helper: is v ready to load?
    def is_loadable(v):
        if v in in_memory:
            return False
        if v in on_disk:
            return True
        if not parents.get(v, []):
            return True # Input not in memory
        return False
        
    # Main Loop
    # We want to reach a state where all outputs are in in_memory.
    # We stop when all outputs are in in_memory.
    
    # We need to be careful about infinite loops.
    # Max ops 20000.
    
    max_ops = 20000
    op_count = 0
    
    while op_count < max_ops:
        # Check goal
        if all(o in in_memory for o in outputs):
            break
            
        # 1. Try to compute a ready vertex
        # Prefer vertices with higher C (more expensive to recompute) to keep them in memory?
        # Actually, computing adds to memory.
        # If memory is full, we must evict first.
        
        ready_to_compute = [v for v in vertices if is_ready(v)]
        
        if ready_to_compute:
            # Pick the one with highest C?
            # High C means expensive to recompute, so we want to compute it and keep it?
            # But computing it puts it in memory.
            # If memory is full, we evict someone else.
            # Let's pick the one with highest C to maximize value in memory.
            v_to_compute = max(ready_to_compute, key=lambda x: C[idx[x]])
            
            # Check if we need to evict
            if len(in_memory) >= budget:
                # Evict a vertex
                # Candidates: in_memory vertices
                # Loss(u) = min(2, C_u) if u not in outset else min(2, C_u) + penalty?
                # If u is an output, we really don't want to evict it.
                # But if we must, we do.
                
                # Score for eviction:
                # We want to minimize the cost of getting it back.
                # Cost to get back = 1 (load) if on disk, or C_u (recompute) if deleted.
                # If we store it, it goes to disk. Cost to get back = 1 (load).
                # If we delete it, it is gone. Cost to get back = C_u (recompute).
                # So if we evict u:
                #   If we store: future cost 1.
                #   If we delete: future cost C_u.
                # We choose the min.
                # So effective loss = min(1, C_u) = 1 (since C_u >= 1).
                # Wait, if we store, we pay 1 now (store op). Future load is 1. Total 2.
                # If we delete, we pay 0 now. Future compute is C_u. Total C_u.
                # So we compare 2 vs C_u.
                # If C_u > 2, we store (cost 2).
                # If C_u <= 2, we delete (cost C_u).
                
                best_evict = None
                best_loss = float('inf')
                
                for u in in_memory:
                    # Don't evict if it's the one we want to compute? No, u is in memory, v is not.
                    # Calculate loss
                    c_u = C[idx[u]]
                    if c_u > 2:
                        loss = 2 # store + load
                        op_type = "store"
                    else:
                        loss = c_u # delete + recompute
                        op_type = "delete"
                        
                    # Add penalty if u is an output
                    if u in outset:
                        loss += 1000 # Strong penalty to avoid evicting outputs
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_evict = u
                        evict_op = op_type
                        
                if best_evict is not None:
                    # Perform eviction
                    if evict_op == "store":
                        ops.append(["store", best_evict])
                        on_disk.add(best_evict)
                    else:
                        ops.append(["delete", best_evict])
                        # If deleted, it's not on disk. It's gone.
                        # We don't add to on_disk.
                    in_memory.remove(best_evict)
                    op_count += 1
                    continue # Re-evaluate ready set? Or just proceed?
                    # Proceeding is fine, we just made space.
                    
            # Now compute
            ops.append(["compute", v_to_compute])
            in_memory.add(v_to_compute)
            # If it was on disk, remove from on_disk?
            # If we computed it, it's in memory.
            if v_to_compute in on_disk:
                on_disk.remove(v_to_compute)
            op_count += 1
            
        else:
            # No vertex is ready to compute.
            # We need to load a parent.
            # Find a vertex v not in memory such that loading it would make some child ready?
            # Or just load any loadable vertex that is a parent of a "soon-to-be-ready" vertex.
            
            # Let's find all vertices v not in memory.
            # If v is an input, it's loadable.
            # If v is on disk, it's loadable.
            # We want to load a vertex that helps us progress.
            
            # Strategy: Find a vertex v such that all parents of v are in memory, but v is not in memory.
            # This is the definition of is_ready. But is_ready returned empty.
            # This means for all v not in memory, at least one parent is not in memory.
            # So we need to load a parent.
            
            # Find candidates: vertices u not in memory, such that u is a parent of some v.
            # And v is "close" to being ready.
            
            # Simple heuristic: Load the vertex with the highest C among those that are loadable?
            # No, we want to load what we need.
            
            # Let's find any vertex v not in memory.
            # Check its parents.
            # If a parent p is not in memory, p is a candidate to load.
            
            candidates = set()
            for v in vertices:
                if v in in_memory:
                    continue
                ps = parents.get(v, [])
                if not ps:
                    # v is an input
                    # If v is not in memory, it's a candidate to load
                    # But we only load inputs if we need them?
                    # Actually, if v is an input and not in memory, and it's not ready to compute (it is an input),
                    # it's just sitting there.
                    # We should load it if it's a parent of something we need.
                    pass
                else:
                    for p in ps:
                        if p not in in_memory:
                            # p is a parent of v, and p is not in memory.
                            # If we load p, does it help?
                            # Only if all other parents of v are in memory.
                            others_ready = all(pp in in_memory for pp in ps if pp != p)
                            if others_ready:
                                # Loading p makes v ready.
                                candidates.add(p)
                                
            if not candidates:
                # No direct parent load will make a vertex ready.
                # We might need to load a "higher up" parent.
                # Or we are stuck in a cycle of needs.
                # Fallback: Load any input not in memory.
                for v in inputs:
                    if v not in in_memory:
                        candidates.add(v)
                        break
                        
            if not candidates:
                # Nothing to load?
                # Maybe we are done?
                break
                
            # Pick a candidate
            # Prefer candidates with high C?
            # Or candidates that are inputs?
            # Let's pick the one with the highest C.
            v_to_load = max(candidates, key=lambda x: C[idx[x]])
            
            # Check if we need to evict
            if len(in_memory) >= budget:
                best_evict = None
                best_loss = float('inf')
                for u in in_memory:
                    c_u = C[idx[u]]
                    if c_u > 2:
                        loss = 2
                        op_type = "store"
                    else:
                        loss = c_u
                        op_type = "delete"
                    if u in outset:
                        loss += 1000
                    if loss < best_loss:
                        best_loss = loss
                        best_evict = u
                        evict_op = op_type
                        
                if best_evict is not None:
                    if evict_op == "store":
                        ops.append(["store", best_evict])
                        on_disk.add(best_evict)
                    else:
                        ops.append(["delete", best_evict])
                    in_memory.remove(best_evict)
                    op_count += 1
                    continue
                    
            # Load
            ops.append(["load", v_to_load])
            in_memory.add(v_to_load)
            if v_to_load in on_disk:
                on_disk.remove(v_to_load)
            op_count += 1
            
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50
    try:
        result = _pebble_greedy_cost(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    # Fallback to original naive
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        # Very basic fallback
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
        ops = []
        for v in vertices:
            for p in parents.get(v, []):
                ops.append(["load", p])
            ops.append(["compute", v])
            if v in outputs:
                ops.append(["store", v])
        return ops[:20000]
# EVOLVE-BLOCK-END
