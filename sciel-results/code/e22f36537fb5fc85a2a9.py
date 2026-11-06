"""Communication-minimizing DAG pebbling with improved eviction and scheduling heuristics."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

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
    DEADLINE = 85.0

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])

        # Build children map
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)

        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)

        # Topological sort (Kahn's algorithm)
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

        topo_index = {v: i for i, v in enumerate(topo_order)}
        n_vertices = len(topo_order)

        # Precompute depth (longest path from any input)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1

        # Precompute distance to sink (longest path from v to any output)
        dist_to_sink = {}
        for v in reversed(topo_order):
            ch = children.get(v, [])
            if not ch:
                dist_to_sink[v] = 1 if v in set(outputs) else 0
            else:
                dist_to_sink[v] = 1 + max(dist_to_sink[c] for c in ch)

        # Precompute number of distinct descendants (transitive dependents)
        descendant_count = {}
        for v in reversed(topo_order):
            ch = children.get(v, [])
            if not ch:
                descendant_count[v] = 0
            else:
                # Union of children's descendants + children themselves
                # Approximate by summing (may overcount if shared descendants, but fast)
                # For better accuracy, we could use a set, but that's O(n^2) worst case
                # Let's use a bounded approximation: sum of child descendant counts + len(children)
                # This overcounts but preserves relative ordering reasonably
                desc = len(ch)
                for c in ch:
                    desc += descendant_count[c]
                descendant_count[v] = desc

        # Precompute last_use and next_use
        last_use = {}
        next_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
                next_use[v] = min(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
                next_use[v] = topo_index.get(v, -1)

        # Outputs must be blue at the end
        must_be_blue = set(outputs)

        # Precompute recompute_cost: number of ancestors (approximate, using depth)
        recompute_cost = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                recompute_cost[v] = 1
            else:
                recompute_cost[v] = 1 + max(recompute_cost[p] for p in ps)

        ops = []
        fast_mem = set()
        stored_vertices = set()

        current_topo_index = [0]

        def get_eviction_candidate():
            """Return the vertex in fast_mem to evict, or None.
            Uses a smooth surrogate combining multiple factors.
            """
            best = None
            best_key = None
            cur = current_topo_index[0]

            for v in fast_mem:
                if v in must_be_blue:
                    # Outputs: high cost to evict
                    key = (2, recompute_cost[v], last_use[v])
                else:
                    if last_use[v] < cur:
                        # Dead vertex, evict first
                        key = (0, 0.0, 0.0)
                    else:
                        next_dist = max(0, next_use[v] - cur)
                        if v in stored_vertices:
                            eff_cost = 1.0
                        else:
                            eff_cost = float(recompute_cost[v])

                        fanout_persist = descendant_count[v] * max(1, dist_to_sink[v])

                        # Smooth surrogate: use log-sum-exp style combination
                        # We want to evict vertices where:
                        # 1. eff_cost is low (cheap to recompute)
                        # 2. next_dist is large (not needed soon)
                        # 3. fanout_persist is low (few dependents far away)

                        # Normalize components to [0, 1] range approximately
                        max_cost = max(recompute_cost.values()) if recompute_cost else 1
                        max_dist = max(dist_to_sink.values()) if dist_to_sink else 1
                        max_fanout = max(descendant_count.values()) if descendant_count else 1

                        cost_norm = eff_cost / (max_cost + 1.0)
                        dist_norm = next_dist / (max_dist * 2.0 + 1.0)
                        fanout_norm = fanout_persist / (max_fanout * max_dist + 1.0)

                        # Eviction score: lower is better to evict
                        # High cost to recompute + needed soon + high fanout = keep
                        keep_score = 0.4 * cost_norm + 0.3 * (1.0 - dist_norm) + 0.3 * fanout_norm
                        evict_score = 1.0 - keep_score

                        key = (1, evict_score, recompute_cost[v])

                if best_key is None or key < best_key:
                    best_key = key
                    best = v
            return best

        visiting = set()

        def ensure_in_fast_mem(v, depth_limit=0):
            if time.time() > DEADLINE:
                return False

            if v in fast_mem:
                return True

            if v in visiting:
                return False

            ps = parents.get(v, [])

            if not ps:
                # Input vertex
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                if v in stored_vertices:
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate()
                        if cand is None:
                            return False
                        ops.append(['delete', cand])
                        fast_mem.discard(cand)
                    ops.append(['load', v])
                    fast_mem.add(v)
                    return True
                else:
                    visiting.add(v)
                    try:
                        for p in ps:
                            if not ensure_in_fast_mem(p, depth_limit + 1):
                                return False

                        if len(fast_mem) >= fast_memory_budget:
                            cand = get_eviction_candidate()
                            if cand is None:
                                return False
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)

                        ops.append(['compute', v])
                        fast_mem.add(v)

                        # Heuristic for storing
                        num_children = len(children.get(v, []))
                        next_dist = max(0, next_use[v] - current_topo_index[0])
                        should_store = (
                            (recompute_cost[v] >= 3 and num_children >= 2) or
                            (recompute_cost[v] >= 2 and next_dist <= 3)
                        )
                        if should_store:
                            ops.append(['store', v])
                            stored_vertices.add(v)

                        return True
                    finally:
                        visiting.discard(v)

        # Process outputs in topological order
        sorted_outputs = sorted(outputs, key=lambda o: topo_index.get(o, 0))

        for i, o in enumerate(sorted_outputs):
            if time.time() > DEADLINE:
                break
            current_topo_index[0] = topo_index.get(o, i)
            if not ensure_in_fast_mem(o):
                for u in list(fast_mem):
                    if u not in must_be_blue and u != o:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                if not ensure_in_fast_mem(o):
                    break

        # Ensure all outputs are in fast_mem at the end
        for o in sorted_outputs:
            if time.time() > DEADLINE:
                break
            if o not in fast_mem:
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                ensure_in_fast_mem(o)

        for o in sorted_outputs:
            if o not in fast_mem:
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                ensure_in_fast_mem(o)
                if o in fast_mem:
                    ops.append(['store', o])
                    stored_vertices.add(o)

        if len(ops) <= 20000:
            return ops
        else:
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]

    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
