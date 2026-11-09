"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list
            
            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        
        return parents, outputs, inputs, vertices, n
    
    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n
    
    return None, None, None, None, None

def _build_children(parents, n):
    """Build children list."""
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children

def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    
    for o in outputs:
        if o not in stored:
            return False
    
    return True

def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))

def _greedy_keep_live(parents, outputs, fast_memory_budget, deadline):
    """
    Greedy pebbling that keeps intermediate values in fast memory as long as possible.
    Evicts based on future usage (furthest next use first).
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    # Precompute next use for each vertex at each step? 
    # Simpler: precompute the list of children.
    # When evicting u, check if u has any children c such that c is not yet processed.
    # We process in topological order 0..n-1.
    # So "not yet processed" means c > current_index.
    
    ops = []
    in_fast = set()
    
    def get_next_use(u, current_idx):
        """Find the smallest child index of u that is > current_idx. Return -1 if none."""
        next_u = -1
        for c in children[u]:
            if c > current_idx:
                if next_u == -1 or c < next_u:
                    next_u = c
        return next_u

    def evict_one(current_idx):
        """Evict one vertex from in_fast. Returns True if successful."""
        if not in_fast:
            return False
        
        # Find candidate with furthest next use (or -1 for no future use)
        # We want to keep vertices that are needed soon.
        # So we evict the one with the largest next_use (or -1).
        best_candidate = None
        best_next_use = -2 # -1 is "no future use", we want to evict those first? 
                           # Actually, if no future use, we can Delete (cost 0).
                           # If future use, we must Store (cost 1).
                           # We prefer to Delete (0 cost) over Store (1 cost).
                           # So we should evict "No Future Use" vertices first.
                           # Among "Future Use" vertices, evict the one with furthest use.
        
        candidates = []
        for u in in_fast:
            if u in is_output:
                # Outputs must be stored when processed. If they are in in_fast, 
                # it means they were computed but not stored yet? 
                # In our algo, we store outputs immediately after compute.
                # So outputs should not be in in_fast for long.
                # But if they are, we must store them.
                candidates.append((0, u, -1)) # Priority 0: Store immediately
            else:
                nu = get_next_use(u, current_idx)
                if nu == -1:
                    candidates.append((1, u, -1)) # Priority 1: Delete (free)
                else:
                    candidates.append((2, u, nu)) # Priority 2: Store (cost 1), sort by nu desc
        
        if not candidates:
            return False
        
        # Sort: 
        # 1. Outputs (must store)
        # 2. No future use (delete, cost 0)
        # 3. Future use (store, cost 1), furthest first
        
        # We want to pick the "best" one to evict.
        # Best is:
        # - If any output, store it (mandatory).
        # - Else, if any no-future-use, delete one (free).
        # - Else, store the one with furthest future use.
        
        outputs_in_fast = [u for u in in_fast if u in is_output]
        if outputs_in_fast:
            u = outputs_in_fast[0]
            ops.append(['store', u])
            in_fast.discard(u)
            return True
            
        no_future = [u for u in in_fast if get_next_use(u, current_idx) == -1]
        if no_future:
            u = no_future[0]
            ops.append(['delete', u])
            in_fast.discard(u)
            return True
            
        # All have future use. Pick furthest.
        best_u = None
        max_nu = -1
        for u in in_fast:
            nu = get_next_use(u, current_idx)
            if nu > max_nu:
                max_nu = nu
                best_u = u
        
        if best_u is not None:
            ops.append(['store', best_u])
            in_fast.discard(best_u)
            return True
            
        return False

    for v in range(n):
        if time.time() > deadline:
            return None
        
        # 1. Ensure parents are in fast
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p
                # If memory full, evict
                while len(in_fast) >= fast_memory_budget:
                    if not evict_one(v):
                        # Cannot evict? Should not happen if budget > 0 and we can delete
                        # If we can't evict, we fail
                        return None 
                ops.append(['load', p])
                in_fast.add(p)
        
        # 2. Compute v
        if parents[v]:
            # v is a computed vertex
            while len(in_fast) >= fast_memory_budget:
                if not evict_one(v):
                    return None
            ops.append(['compute', v])
            in_fast.add(v)
        else:
            # v is an input
            # It should have been loaded in step 1 if it has children?
            # No, if v is input, it has no parents.
            # If v is used, it will be loaded when its child is processed.
            # But if v is an input AND an output, we need to load and store it.
            # If v is just an input, we don't need to load it now unless it's an output.
            # Wait, if v is an input, it's in slow memory initially.
            # We only load it when we need to compute a child.
            # So for input v, we do nothing here, UNLESS v is an output.
            pass

        # 3. Handle Output
        if v in is_output:
            # If v is an input-output, we need to load it first if not in fast
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not evict_one(v):
                        return None
                ops.append(['load', v])
                in_fast.add(v)
            
            # Now v is in fast. Store it.
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # 4. Cleanup: Delete vertices that are no longer needed?
        # The evict_one function handles this when memory is full.
        # But we can proactively delete vertices that have no future use to save memory slots?
        # No, deleting doesn't save memory slots unless we are full.
        # Keeping them doesn't cost transfers.
        # So we only evict when we need space.

    # Final check: ensure all outputs are stored
    # They should have been stored in step 3.
    
    return ops

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    best_ops = None
    best_transfers = float('inf')
    
    # Try Greedy Keep-Live
    try:
        ops = _greedy_keep_live(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    if best_ops is not None:
        return best_ops
    
    # Fallback to simple layer-by-layer if greedy failed
    # (Re-using logic from previous version as backup)
    try:
        ops = []
        in_fast = set()
        children = _build_children(parents, n)
        is_output = set(outputs)
        
        for v in range(n):
            if time.time() > deadline:
                break
            
            for p in parents[v]:
                if p not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            if parents[v]:
                while len(in_fast) >= fast_memory_budget:
                    evicted = False
                    for u in list(in_fast):
                        if u not in is_output:
                            ops.append(['delete', u])
                            in_fast.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            if v in is_output:
                if v not in in_fast and not parents[v]:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', v])
                        in_fast.add(v)
                
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            
            for p in parents[v]:
                if p in in_fast and p not in is_output:
                    still_needed = False
                    for c in children[p]:
                        if c > v:
                            still_needed = True
                            break
                    if not still_needed:
                        ops.append(['delete', p])
                        in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    return []

# EVOLVE-BLOCK-END
