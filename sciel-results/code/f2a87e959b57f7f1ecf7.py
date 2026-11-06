# EVOLVE-BLOCK-START
import sys
import os
import json
import time

def _get_module():
    """Import the task module to inspect DAG structure and baseline."""
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        import autoresearch_suite.science_tcs_expansion as mod
        return mod
    except Exception:
        return None

def _inspect_dag_structure(dag):
    """
    Inspect the dag object to understand its structure.
    Returns a dict with keys: type, keys, sample
    """
    info = {"type": type(dag).__name__}
    if isinstance(dag, dict):
        info["keys"] = list(dag.keys())[:20]
        for k in list(dag.keys())[:5]:
            v = dag[k]
            info[f"sample_{k}"] = (type(v).__name__, str(v)[:200])
    elif isinstance(dag, (list, tuple)):
        info["len"] = len(dag)
        for i in range(min(3, len(dag))):
            info[f"item_{i}"] = (type(dag[i]).__name__, str(dag[i])[:200])
    else:
        # Try common attributes
        for attr in ['vertices', 'nodes', 'edges', 'parents', 'children', 'inputs', 'outputs', 'n', 'num_vertices']:
            if hasattr(dag, attr):
                v = getattr(dag, attr)
                info[attr] = (type(v).__name__, str(v)[:200])
    return info

def _safe_baseline(dag, fast_memory_budget):
    """Run baseline, return its operations list."""
    mod = _get_module()
    if mod is not None and hasattr(mod, 'pebble_baseline'):
        try:
            return mod.pebble_baseline(dag, fast_memory_budget)
        except Exception as e:
            return None
    return None

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure (vertices, parents, inputs, outputs).
    2. Use a greedy topological pebbling with a memory-aware eviction policy.
    3. Minimize transfers by keeping values in fast memory when possible,
       and only storing/reloading when memory is full.
    """
    # First, try to understand the dag structure
    dag_info = _inspect_dag_structure(dag)
    
    # Try baseline first as a fallback
    baseline_ops = _safe_baseline(dag, fast_memory_budget)
    
    # Attempt to parse the dag into a usable form
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
    except Exception:
        # Fall back to baseline
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    if vertices is None:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Build the pebbling schedule
    ops = _pebble_greedy(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if ops is None or len(ops) == 0:
        if baseline_ops is not None:
            return baseline_ops
        return []
    
    # Verify and limit
    if len(ops) > 20000:
        if baseline_ops is not None:
            return baseline_ops
        ops = ops[:20000]
    
    return ops


def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
    vertices: list of vertex ids in topological order (or indexable)
    parents: dict vertex -> list of parent vertex ids
    inputs: set of input vertex ids
    outputs: set of output vertex ids
    """
    if isinstance(dag, dict):
        # Try common key patterns
        # Pattern 1: {'vertices': [...], 'parents': {...}, ...}
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents = dag.get('parents', {})
            inputs = dag.get('inputs', set())
            outputs = dag.get('outputs', set())
            if isinstance(inputs, list):
                inputs = set(inputs)
            if isinstance(outputs, list):
                outputs = set(outputs)
            return vertices, parents, inputs, outputs
        
        # Pattern 2: dag is a dict of vertex -> parents
        # Check if values are lists (parents)
        first_val = next(iter(dag.values())) if dag else None
        if isinstance(first_val, list):
            vertices = list(dag.keys())
            parents = {k: list(v) for k, v in dag.items()}
            # Determine inputs (no parents) and outputs (no children)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 3: {'nodes': [...], 'edges': [...]}
        if 'nodes' in dag and 'edges' in dag:
            nodes = dag['nodes']
            edges = dag['edges']
            vertices = nodes if isinstance(nodes, list) else list(nodes.keys())
            parents = {v: [] for v in vertices}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    src, dst = e[0], e[1]
                    if dst in parents:
                        parents[dst].append(src)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 4: dag has 'n' and 'parents' as arrays
        if 'n' in dag:
            n = dag['n']
            vertices = list(range(n))
            parents_raw = dag.get('parents', [])
            parents = {}
            for i in range(n):
                if i < len(parents_raw):
                    parents[i] = list(parents_raw[i])
                else:
                    parents[i] = []
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        # Maybe it's a list of (vertex, parents) tuples
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
            vertices = []
            parents = {}
            for item in dag:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    # Fallback: try to use the module's helper if available
    mod = _get_module()
    if mod is not None:
        for attr in ['parse_dag', 'get_dag_info', 'extract_dag']:
            if hasattr(mod, attr):
                try:
                    result = getattr(mod, attr)(dag)
                    if isinstance(result, tuple) and len(result) == 4:
                        return result
                except Exception:
                    pass
    
    return None, None, None, None


def _pebble_greedy(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Greedy pebbling with memory management using Next-Use heuristic.
    """
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Determine topological order (vertices should already be in topo order, but verify)
    # Use the given order as topo order
    topo_order = list(vertices)
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    # Pre-compute next use for each vertex
    # next_use[v] = index in topo_order of the next child of v
    # We can update this dynamically.
    # Initially, for each v, next_use[v] is the index of the first child in topo_order.
    next_use = {}
    for v in topo_order:
        ch = children.get(v, [])
        if ch:
            # Find the child with the smallest topological index
            min_idx = min(topo_index[c] for c in ch)
            next_use[v] = min_idx
        else:
            next_use[v] = len(topo_order) + 1 # No future uses
    
    # Count remaining uses: how many children still need this vertex
    remaining_uses = {}
    for v in topo_order:
        remaining_uses[v] = len(children.get(v, []))
    
    # Initial state: inputs are in fast memory (blue)
    fast_mem = set(inputs)
    slow_mem = set()
    computed = set(inputs)
    
    ops = []
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            # Already in fast memory
            # Update next_use for inputs? No, inputs are used by their children.
            # When we process a child, we should update the parent's next_use.
            continue
        
        # Ensure all parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Need to load
                    # Check if we have space
                    if len(fast_mem) >= fast_memory_budget:
                        # Need to evict something
                        evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, p, children, topo_order, next_use)
                        if evicted is not None:
                            fast_mem.discard(evicted)
                            # If evicted, it's no longer in fast mem.
                            # We don't need to update next_use for evicted vertex immediately,
                            # but it will be recalculated if needed? No, next_use is static per vertex.
                            # Wait, next_use is "next child in topo order". This is static.
                            # The heuristic "furthest next use" is valid.
                            ops.append(["delete", evicted])
                    ops.append(["load", p])
                    fast_mem.add(p)
                else:
                    # Parent not in slow either - need to recompute
                    # This is complex; for now, try to compute parent
                    # Recursively ensure parent's parents are in fast, then compute parent
                    _ensure_computed(p, fast_mem, slow_mem, computed, parents, children, 
                                    remaining_uses, outputs, fast_memory_budget, ops, topo_order, next_use)
        
        # Now compute v
        # Check space
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order, next_use)
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        
        ops.append(["compute", v])
        fast_mem.add(v)
        computed.add(v)
        
        # Update remaining uses and next_use for parents
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
                # Update next_use for p
                # The current vertex v is a child of p.
                # Since we are processing in topo order, v is the *next* child of p we are processing?
                # Not necessarily, if p has multiple children, we process them in topo order.
                # So v is the child we just used. The *next* use of p is the next child of p in topo order.
                # We can find it by looking at children[p] and finding the one with min topo_index > topo_index[v].
                # But we can just recalculate min index of remaining children.
                # For efficiency, we can maintain a pointer or just recalculate if small.
                # Given N is likely small enough for 60s, recalculation is fine.
                ch = children.get(p, [])
                remaining_ch = [c for c in ch if topo_index[c] > topo_index[v]]
                if remaining_ch:
                    next_use[p] = min(topo_index[c] for c in remaining_ch)
                else:
                    next_use[p] = len(topo_order) + 1
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, o, None, children, topo_order, next_use)
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                # Need to recompute
                _ensure_computed(o, fast_mem, slow_mem, computed, parents, children,
                                remaining_uses, outputs, fast_memory_budget, ops, topo_order, next_use)
    
    return ops


def _evict(fast_mem, slow_mem, remaining_uses, outputs, current_vertex, priority_parent, children, topo_order, next_use_index=None):
    """
    Choose a vertex to evict from fast memory.
    Uses 'Next Use' heuristic: Evict the vertex whose next usage is furthest in the future.
    If no next use, evict immediately.
    """
    if next_use_index is None:
        # Fallback to old behavior if next_use_index not provided
        next_use_index = {}
        
    candidates = []
    for v in fast_mem:
        if v == current_vertex:
            continue
        if v == priority_parent:
            continue
        if v in outputs:
            continue
        # If it's an input and we haven't processed all its children, it has uses.
        # next_use_index[v] is the index in topo_order of the next child of v.
        # If v is not in next_use_index, it has no future uses.
        nu = next_use_index.get(v, len(topo_order) + 1) # Infinity if no next use
        candidates.append((nu, v))
    
    if not candidates:
        # Last resort: evict anything that's not an output and not current
        for v in fast_mem:
            if v != current_vertex and v != priority_parent and v not in outputs:
                return v
        return None
    
    # Sort by next use index (furthest first)
    candidates.sort(reverse=True)
    return candidates[0][1]


def _ensure_computed(v, fast_mem, slow_mem, computed, parents, children, 
                     remaining_uses, outputs, fast_memory_budget, ops, topo_order, next_use=None):
    """
    Ensure vertex v is computed and in fast memory.
    Recursively compute parents if needed.
    """
    if next_use is None:
        next_use = {}
        
    if v in fast_mem:
        return
    
    if v in slow_mem:
        # Load it
        if len(fast_mem) >= fast_memory_budget:
            evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order, next_use)
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        ops.append(["load", v])
        fast_mem.add(v)
        return
    
    # Need to compute v from scratch
    ps = parents.get(v, [])
    for p in ps:
        _ensure_computed(p, fast_mem, slow_mem, computed, parents, children,
                        remaining_uses, outputs, fast_memory_budget, ops, topo_order, next_use)
    
    # Compute v
    if len(fast_mem) >= fast_memory_budget:
        evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, None, children, topo_order, next_use)
        if evicted is not None:
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
    
    ops.append(["compute", v])
    fast_mem.add(v)
    computed.add(v)

# EVOLVE-BLOCK-END
