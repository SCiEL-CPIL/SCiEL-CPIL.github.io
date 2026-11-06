"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def _topo_sort(vertices, parents):
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    vset = set(vertices)
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)
    return topo

def _pebble_fallback(dag, budget):
    """Greedy baseline: topological pass with aggressive deletion."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    # children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    topo = _topo_sort(vertices, parents)
    # remaining children counter
    remaining = {v: len(children.get(v, [])) for v in vertices}
    ops = []
    in_fast = set()
    # process in topo order
    for v in topo:
        for p in parents.get(v, []):
            if p not in in_fast:
                ops.append(["load", p])
                in_fast.add(p)
        ops.append(["compute", v])
        in_fast.add(v)
        if v in outset:
            ops.append(["store", v])
            in_fast.discard(v)
        # delete parents that are no longer needed
        for p in parents.get(v, []):
            remaining[p] = remaining.get(p, 0) - 1
            if remaining[p] <= 0 and p not in outset and p in in_fast:
                ops.append(["delete", p])
                in_fast.discard(p)
    # store any remaining outputs still in fast
    for v in outset:
        if v in in_fast:
            ops.append(["store", v])
            in_fast.discard(v)
    return ops[:20000]

def _find_bridges(dag, budget, max_bridges):
    """
    Find a small set of 'bridge' vertices whose removal makes the DAG decompose
    into smaller independent sub-DAGs. Bridges are treated as persistent
    fast-memory checkpoints to avoid recomputation.
    Strategy: pick vertices with high "fan-out" (many descendants) that sit on
    many root-to-output paths. Removing them splits the graph.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0 or n <= max_bridges:
        return []
    vset = set(vertices)
    outset = set(outputs)
    # children
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    # compute descendants count and "betweenness"-like score
    # A bridge is valuable if removing it splits the graph into two large parts.
    # Approximation: score = (#descendants) * (#ancestors) — vertices in the "middle".
    # Compute ancestors and descendants via DFS.
    import sys
    sys.setrecursionlimit(100000)
    def count_desc(v):
        seen = set()
        stack = [v]
        while stack:
            u = stack.pop()
            for w in children.get(u, []):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen)
    def count_anc(v):
        seen = set()
        stack = [v]
        while stack:
            u = stack.pop()
            for p in parents.get(u, []):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return len(seen)
    # Only consider non-input, non-output vertices as bridges (inputs/outputs are fixed)
    candidates = [v for v in vertices if v not in outset and parents.get(v, [])]
    scored = []
    for v in candidates:
        d = count_desc(v)
        a = count_anc(v)
        # Value of splitting: product of the two sides (roughly balanced split is good)
        score = d * a
        scored.append((score, v))
    scored.sort(reverse=True)
    bridges = []
    for s, v in scored[:max_bridges]:
        bridges.append(v)
    return bridges

def _pebble_dag(dag, budget, deadline, bridges=None):
    """
    Recursive divide-and-conquer pebbling.
    If bridges are given, they are kept in fast memory persistently and the
    DAG is split into sub-DAGs by removing bridges; each sub-DAG is solved
    independently and the operations are interleaved so bridges are loaded once.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if time.time() > deadline:
        return None

    # If no bridges or small DAG, do a single topological pass with deletion.
    if not bridges or n <= 20:
        return _pebble_single(dag, budget, deadline)

    bset = set(bridges)
    # Build sub-DAGs: remove bridges, find connected components (via parents/children)
    remaining = [v for v in vertices if v not in bset]
    rset = set(remaining)
    # children within remaining
    rchildren = defaultdict(list)
    for v in remaining:
        for p in parents.get(v, []):
            if p in rset:
                rchildren[p].append(v)
    # find components (undirected reachability)
    visited = set()
    components = []
    for v in remaining:
        if v in visited:
            continue
        comp = []
        stack = [v]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for w in rchildren.get(u, []):
                if w not in visited:
                    stack.append(w)
            for p in parents.get(u, []):
                if p in rset and p not in visited:
                    stack.append(p)
        if comp:
            components.append(comp)

    # If removing bridges did not split (one component), just do single pass.
    if len(components) <= 1:
        return _pebble_single(dag, budget, deadline)

    # Order components by size (largest first)
    components.sort(key=len, reverse=True)

    # Plan: load all bridges once, then process each component, then delete bridges.
    # But bridges may be parents of vertices in multiple components, so they must
    # persist across components.
    ops = []
    in_fast = set()
    # Load bridges
    # Bridges may have parents that are also bridges or in components.
    # To keep it simple: load bridges in topological order among themselves.
    bridge_topo = _topo_sort(sorted(bset, key=lambda x: (x if isinstance(x, int) else 0)), parents)
    # For each bridge, ensure its parents (that are bridges) are loaded; non-bridge parents
    # are handled when their component is processed. This is tricky; to be safe, load
    # bridges along with their non-bridge parents on demand.
    # Simpler robust approach: process the whole DAG in topological order but keep
    # bridges in fast memory (never delete them) and delete everything else aggressively.
    # This still benefits from the "checkpoint" idea: bridges are never recomputed.
    return _pebble_keep_bridges(dag, budget, deadline, bset)

def _pebble_keep_bridges(dag, budget, deadline, bset):
    """
    Topological pass where bridges are kept in fast memory (never deleted) and
    all other vertices are deleted as soon as possible. This avoids recomputing
    bridges and reduces memory pressure on the rest.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    outset = set(outputs)
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    topo = _topo_sort(vertices, parents)
    remaining = {v: len(children.get(v, [])) for v in vertices}
    ops = []
    in_fast = set()
    for v in topo:
        if time.time() > deadline:
            break
        for p in parents.get(v, []):
            if p not in in_fast:
                ops.append(["load", p])
                in_fast.add(p)
        ops.append(["compute", v])
        in_fast.add(v)
        if v in outset:
            ops.append(["store", v])
            in_fast.discard(v)
        # delete parents that are no longer needed and not bridges
        for p in parents.get(v, []):
            remaining[p] = remaining.get(p, 0) - 1
            if remaining[p] <= 0 and p not in outset and p not in bset and p in in_fast:
                ops.append(["delete", p])
                in_fast.discard(p)
    # store remaining outputs
    for v in outset:
        if v in in_fast:
            ops.append(["store", v])
            in_fast.discard(v)
    # delete remaining bridges (not required, but frees memory; outputs already stored)
    for v in bset:
        if v in in_fast and v not in outset:
            ops.append(["delete", v])
            in_fast.discard(v)
    return ops[:20000]

def _pebble_single(dag, budget, deadline):
    """Single topological pass with aggressive deletion (no persistent bridges)."""
    return _pebble_keep_bridges(dag, budget, deadline, set())

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # safety margin before 60s
    try:
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
        n = len(vertices)
        if n == 0:
            return []
        # Choose number of bridges based on budget: keep a small set persistent.
        # If budget is large enough to hold all, no bridges needed.
        if fast_memory_budget >= n:
            return _pebble_single(dag, fast_memory_budget, deadline)
        # Pick a small set of bridges: at most min(5, n//4)
        max_bridges = max(1, min(5, n // 4))
        bridges = _find_bridges(dag, fast_memory_budget, max_bridges)
        result = _pebble_dag(dag, fast_memory_budget, deadline, bridges)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    # Fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return _pebble_fallback(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
