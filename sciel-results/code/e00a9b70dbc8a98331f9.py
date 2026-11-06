# EVOLVE-BLOCK-START
import sys
import os
import json
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using layer-aware scheduling.
    
    Strategy:
    1. Parse the DAG structure.
    2. Identify layers (topological levels).
    3. Process layer by layer, keeping current and next layer in fast memory.
    4. Minimize transfers by deferring eviction and using lookahead.
    """
    try:
        vertices, parents, inputs, outputs = _parse_dag(dag)
        if vertices is None:
            return _fallback_baseline(dag, fast_memory_budget)
    except Exception:
        return _fallback_baseline(dag, fast_memory_budget)
    
    ops = _pebble_layered(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if ops is None or len(ops) == 0:
        return _fallback_baseline(dag, fast_memory_budget)
    
    if len(ops) > 20000:
        return _fallback_baseline(dag, fast_memory_budget)
    
    return ops


def _fallback_baseline(dag, fast_memory_budget):
    """Fallback to naive baseline if available."""
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return []


def _parse_dag(dag):
    """
    Parse dag into (vertices, parents, inputs, outputs).
    Returns None for any component if parsing fails.
    """
    if isinstance(dag, dict):
        # Pattern 1: {'vertices': [...], 'parents': {...}, 'inputs': [...], 'outputs': [...]}
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
        
        # Pattern 2: dag is a dict of vertex -> parents list
        first_val = next(iter(dag.values())) if dag else None
        if isinstance(first_val, list):
            vertices = list(dag.keys())
            parents = {k: list(v) for k, v in dag.items()}
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
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
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
        
        # Pattern 4: {'n': int, 'parents': [[...], ...]}
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
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        # List of (vertex, parents) tuples
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
            vertices = []
            parents = {}
            for item in dag:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            inputs = {v for v in vertices if not parents.get(v)}
            outputs = {v for v in vertices if not children.get(v)}
            return vertices, parents, inputs, outputs
    
    return None, None, None, None


def _compute_levels(vertices, parents):
    """Compute topological levels for each vertex."""
    level = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            level[v] = 0
        else:
            level[v] = max(level.get(p, 0) for p in ps) + 1
    return level


def _pebble_layered(vertices, parents, inputs, outputs, fast_memory_budget):
    """
    Layer-aware pebbling to minimize transfers.
    
    Key insight: Process vertices level by level. Keep current level results
    in fast memory until all children in the next level are computed.
    Only evict when necessary, and prefer evicting vertices whose last
    child has already been processed.
    """
    # Compute levels
    level = _compute_levels(vertices, parents)
    
    # Group vertices by level
    from collections import defaultdict
    level_to_vertices = defaultdict(list)
    for v in vertices:
        level_to_vertices[level[v]].append(v)
    
    max_level = max(level.values()) if level else 0
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many children each vertex has (total)
    total_children = {v: len(children[v]) for v in vertices}
    
    # Track which children have been processed for each vertex
    processed_children = defaultdict(int)
    
    # State
    fast_mem = set(inputs)
    slow_mem = set()
    computed = set(inputs)
    ops = []
    
    # Process level by level
    for lvl in range(0, max_level + 1):
        verts_at_level = level_to_vertices.get(lvl, [])
        
        # For this level, we need to ensure all parents are available
        # Process vertices in this level
        for v in verts_at_level:
            if v in inputs:
                continue
            
            ps = parents.get(v, [])
            
            # Ensure all parents are in fast memory
            for p in ps:
                if p not in fast_mem:
                    if p in slow_mem:
                        # Load from slow memory
                        if len(fast_mem) >= fast_memory_budget:
                            evicted = _find_evict_candidate(
                                fast_mem, slow_mem, processed_children, total_children,
                                outputs, v, p, level, level_to_vertices
                            )
                            if evicted is not None:
                                fast_mem.discard(evicted)
                                ops.append(["delete", evicted])
                        ops.append(["load", p])
                        fast_mem.add(p)
                    else:
                        # Need to recompute parent
                        _recompute_vertex(p, fast_mem, slow_mem, computed, parents,
                                         children, processed_children, total_children,
                                         outputs, fast_memory_budget, ops, level, level_to_vertices)
            
            # Compute v
            if len(fast_mem) >= fast_memory_budget:
                evicted = _find_evict_candidate(
                    fast_mem, slow_mem, processed_children, total_children,
                    outputs, v, None, level, level_to_vertices
                )
                if evicted is not None:
                    fast_mem.discard(evicted)
                    ops.append(["delete", evicted])
            
            ops.append(["compute", v])
            fast_mem.add(v)
            computed.add(v)
            
            # Mark this child as processed for each parent
            for p in ps:
                processed_children[p] += 1
        
        # After processing a level, consider storing vertices that won't be needed again
        # But only if memory is tight
        if fast_memory_budget > 0:
            for v in verts_at_level:
                if v in fast_mem and v not in outputs:
                    # Check if all children are processed
                    if processed_children[v] >= total_children[v]:
                        # This vertex won't be needed again
                        # Store it in slow memory to free up space
                        if len(fast_mem) > fast_memory_budget * 0.7:
                            ops.append(["store", v])
                            fast_mem.discard(v)
                            slow_mem.add(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            if o in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evicted = _find_evict_candidate(
                        fast_mem, slow_mem, processed_children, total_children,
                        outputs, o, None, level, level_to_vertices
                    )
                    if evicted is not None:
                        fast_mem.discard(evicted)
                        ops.append(["delete", evicted])
                ops.append(["load", o])
                fast_mem.add(o)
            else:
                _recompute_vertex(o, fast_mem, slow_mem, computed, parents,
                                 children, processed_children, total_children,
                                 outputs, fast_memory_budget, ops, level, level_to_vertices)
    
    return ops


def _find_evict_candidate(fast_mem, slow_mem, processed_children, total_children,
                          outputs, current_vertex, priority_parent, level, level_to_vertices):
    """
    Find the best vertex to evict from fast memory.
    Prefer vertices that are no longer needed (all children processed) and not outputs.
    """
    best_candidate = None
    best_score = float('inf')
    
    for v in fast_mem:
        if v == current_vertex or v == priority_parent:
            continue
        if v in outputs:
            continue
        
        # Score: lower is better
        # Priority 1: vertices with all children processed (0 remaining uses)
        remaining = total_children[v] - processed_children[v]
        
        # Priority 2: vertices at lower levels (further from current work)
        lvl = level.get(v, 0)
        cur_lvl = level.get(current_vertex, 0) if current_vertex else 0
        
        # Score: remaining uses * 1000 + (cur_lvl - lvl) * 10
        # We want to evict vertices with fewest remaining uses and lowest level
        score = remaining * 1000 + max(0, cur_lvl - lvl) * 10
        
        if score < best_score:
            best_score = score
            best_candidate = v
    
    return best_candidate


def _recompute_vertex(v, fast_mem, slow_mem, computed, parents, children,
                      processed_children, total_children, outputs,
                      fast_memory_budget, ops, level, level_to_vertices):
    """
    Recompute vertex v from scratch if not in fast or slow memory.
    Recursively ensure parents are available.
    """
    if v in fast_mem:
        return
    
    if v in slow_mem:
        if len(fast_mem) >= fast_memory_budget:
            evicted = _find_evict_candidate(
                fast_mem, slow_mem, processed_children, total_children,
                outputs, v, None, level, level_to_vertices
            )
            if evicted is not None:
                fast_mem.discard(evicted)
                ops.append(["delete", evicted])
        ops.append(["load", v])
        fast_mem.add(v)
        return
    
    # Need to compute from scratch
    ps = parents.get(v, [])
    for p in ps:
        _recompute_vertex(p, fast_mem, slow_mem, computed, parents, children,
                         processed_children, total_children, outputs,
                         fast_memory_budget, ops, level, level_to_vertices)
    
    if len(fast_mem) >= fast_memory_budget:
        evicted = _find_evict_candidate(
            fast_mem, slow_mem, processed_children, total_children,
            outputs, v, None, level, level_to_vertices
        )
        if evicted is not None:
            fast_mem.discard(evicted)
            ops.append(["delete", evicted])
    
    ops.append(["compute", v])
    fast_mem.add(v)
    computed.add(v)

# EVOLVE-BLOCK-END
