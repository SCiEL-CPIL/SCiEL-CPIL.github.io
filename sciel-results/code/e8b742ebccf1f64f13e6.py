"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    # --- Parse the DAG structure (robust to several representations) ---
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    elif hasattr(dag, 'vertices'):
        n = len(dag.vertices)
    else:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    def get_parents(v):
        try:
            if hasattr(dag, 'parents'):
                return dag.parents[v]
            elif isinstance(dag, dict) and 'parents' in dag:
                return dag['parents'][v]
        except Exception:
            pass
        return []

    def get_outputs():
        try:
            if hasattr(dag, 'outputs'):
                return set(dag.outputs)
            elif isinstance(dag, dict) and 'outputs' in dag:
                return set(dag['outputs'])
        except Exception:
            pass
        return {n - 1}

    outputs = get_outputs()

    # --- Precompute static structure ---
    parents_list = [list(get_parents(v)) for v in range(n)]
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children[p].append(v)

    # next_child[v] = how many children of v still need to be computed
    # (i.e. how many times v's value will still be consumed).
    next_child = [len(children[v]) for v in range(n)]

    # --- Bounded look-ahead: which vertices are needed as parents within
    #     the next K topological positions. Protecting these from eviction
    #     avoids immediate reloads, the dominant transfer cost in layered DAGs.
    LOOKAHEAD = 8
    in_lookahead = [False] * n
    for i in range(n):
        for j in range(i, min(i + LOOKAHEAD, n)):
            for p in parents_list[j]:
                in_lookahead[p] = True

    red = set()
    ops = []
    MAX_OPS = 20000

    # --- Eviction priority heap: (priority, vertex). Lower = evict first.
    #     priority = next_child[v] + big_penalty_if_in_lookahead
    #     We use lazy deletion (pop until a valid, still-red entry is found).
    heap = []

    BIG = 1 << 30

    def push_eviction(v):
        pr = next_child[v] + (BIG if in_lookahead[v] else 0)
        heappush(heap, (pr, v))

    def top_eviction():
        while heap:
            pr, v = heap[0]
            if v in red:
                return v
            heappop(heap)
        return None

    def make_room(need=1):
        # Evict until we have at least `need` free slots, or heap is empty.
        while len(red) >= fast_memory_budget and need > 0:
            v = top_eviction()
            if v is None:
                break
            heappop(heap)
            ops.append(['store', v])
            red.discard(v)
            need -= 1

    # --- Main topological sweep: compute each vertex as soon as possible,
    #     keep only what the next layer's fan-in needs, evict the rest.
    for v in range(n):
        if len(ops) >= MAX_OPS:
            break
        if time.time() - start_time > deadline:
            break

        ps = parents_list[v]

        if not ps:
            # Input vertex: load into fast memory.
            if v not in red:
                make_room(1)
                if len(red) < fast_memory_budget:
                    ops.append(['load', v])
                    red.add(v)
                    push_eviction(v)
        else:
            # Internal vertex: ensure all parents are red, then compute.
            for p in ps:
                if p not in red:
                    make_room(1)
                    if len(red) < fast_memory_budget:
                        ops.append(['load', p])
                        red.add(p)
                        push_eviction(p)
            ops.append(['compute', v])
            red.add(v)
            push_eviction(v)

        # v has now been used as a parent by all its children in this sweep;
        # actually we decrement when each child is processed. Decrement here
        # for each child of v that we are about to (or have) consumed.
        # Since children are processed later in topological order, we instead
        # decrement next_child[v] now to reflect that v is "done being produced"
        # and its remaining uses equal the children not yet computed.
        # (We already set next_child[v] = total children; as each child is
        #  computed we should decrement. We do that at the start of each child.)
        # For simplicity and correctness of the priority, decrement now by 1
        # if v itself is a child (i.e. has parents) — but the cleanest is to
        # decrement at child-processing time. We handle it below.

    # The decrement of next_child must happen as each child is computed.
    # Re-do the sweep with proper bookkeeping (single pass, correct counts).
    # (The pass above was a structural check; we now run the real schedule.)
    ops = []
    red = set()
    heap = []
    next_child = [len(children[v]) for v in range(n)]

    for v in range(n):
        if len(ops) >= MAX_OPS:
            break
        if time.time() - start_time > deadline:
            break

        ps = parents_list[v]

        # Ensure parents are red
        if ps:
            for p in ps:
                if p not in red:
                    make_room(1)
                    if len(red) < fast_memory_budget:
                        ops.append(['load', p])
                        red.add(p)
                        push_eviction(p)
                # p is now consumed by v; decrement its remaining uses
                next_child[p] -= 1
                # If p's remaining uses just hit zero, it's safe to evict
                # (recompute-on-demand). Push with updated priority.
                push_eviction(p)

            ops.append(['compute', v])
            red.add(v)
            push_eviction(v)
        else:
            # Input vertex
            if v not in red:
                make_room(1)
                if len(red) < fast_memory_budget:
                    ops.append(['load', v])
                    red.add(v)
                    push_eviction(v)

        # If v is a leaf (no children) and not an output, we could delete it
        # immediately to free memory. But we let the heap handle eviction
        # naturally; an explicit delete of a never-again-needed vertex is
        # strictly beneficial.
        if not children[v] and v not in outputs:
            if v in red:
                ops.append(['delete', v])
                red.discard(v)

    # --- Finalize: all outputs must finish blue (stored to slow memory).
    for v in sorted(outputs):
        if v in red:
            ops.append(['store', v])
            red.discard(v)

    # --- Safety: if we ran out of operations or time, ensure a valid-ish
    #     tail. The evaluator's state-machine verification is strict, so if
    #     we cannot complete, return the baseline.
    if len(ops) > MAX_OPS:
        ops = ops[:MAX_OPS]
        for v in sorted(outputs):
            if v in red:
                ops.append(['store', v])
                red.discard(v)

    if time.time() - start_time > deadline:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            pass

    return ops
# EVOLVE-BLOCK-END
