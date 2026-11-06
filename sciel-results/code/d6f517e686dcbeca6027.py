"""DAG pebbling solver with LRU-based eviction and lookahead optimization."""
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Build adjacency
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    # Compute in-degrees and topological order
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1

    topo_order = []
    queue = deque(v for v in vertices if in_degree[v] == 0)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Identify inputs and outputs
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)

    # Precompute topo position for each vertex
    topo_pos = {v: i for i, v in enumerate(topo_order)}

    # For each vertex, precompute sorted list of child positions
    children_positions = {}
    for v in vertices:
        if v in children:
            children_positions[v] = sorted([topo_pos[c] for c in children[v]])
        else:
            children_positions[v] = []

    # Compute total fan-out
    fan_out = {v: len(children.get(v, [])) for v in vertices}

    # Compute number of parents (recomputation cost proxy)
    num_parents = {v: len(parents.get(v, [])) for v in vertices}

    # State
    state = {}
    for v in vertices:
        state[v] = 'blue' if v in input_set else 'red'

    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []

    # Track remaining uses (how many children still need this vertex)
    remaining_uses = {v: fan_out[v] for v in vertices}

    # For efficient next-use lookup, maintain a pointer index per vertex
    next_use_idx = {v: 0 for v in vertices}

    # LRU cache for eviction order
    lru_order = []
    lru_pos = {}

    def lru_touch(v):
        """Move v to most recently used position."""
        if v in lru_pos:
            lru_order.pop(lru_pos[v])
        lru_order.append(v)
        lru_pos[v] = len(lru_order) - 1

    def lru_remove(v):
        """Remove v from LRU order."""
        if v in lru_pos:
            pos = lru_pos[v]
            lru_order[pos] = lru_order[-1]
            lru_pos[lru_order[pos]] = pos
            lru_order.pop()
            del lru_pos[v]

    def get_next_use_pos(v, current_idx):
        """Get the topo position of the next use of vertex v after current_idx."""
        cps = children_positions.get(v, [])
        idx = next_use_idx[v]
        while idx < len(cps) and cps[idx] <= current_idx:
            idx += 1
        next_use_idx[v] = idx
        if idx < len(cps):
            return cps[idx]
        return None

    def eviction_score(v, current_idx):
        """
        Score for eviction: lower score = more likely to be evicted.
        We want to keep vertices that are:
        1. Outputs (never evict if possible)
        2. Needed soon (small distance to next use)
        3. Have high fan-out (expensive to reload)
        4. Hard to recompute (many parents)
        """
        if v in output_set:
            return float('inf')  # Never evict outputs
        
        rem = remaining_uses.get(v, 0)
        if rem <= 0:
            return -1000.0  # No more uses, evict immediately
        
        next_pos = get_next_use_pos(v, current_idx)
        if next_pos is None:
            return -500.0  # No future use, evict
        
        dist = next_pos - current_idx
        # Higher fan-out and smaller distance = more important to keep
        # Add recomputation cost as a tiebreaker
        recompute_cost = num_parents.get(v, 0)
        score = (fan_out.get(v, 1) + 1.0) / (1.0 + dist) + 0.1 * recompute_cost
        return score

    def evict_if_needed(exclude=None, current_idx=0):
        """Evict vertices until fast_mem is within budget."""
        nonlocal fast_mem, state, operations
        while len(fast_mem) > fast_memory_budget:
            if time.time() > deadline:
                return
            
            # Find best victim using eviction score, but prefer LRU order for ties
            best_victim = None
            best_score = float('inf')
            for u in lru_order:
                if u not in fast_mem:
                    continue
                if u == exclude:
                    continue
                sc = eviction_score(u, current_idx)
                if sc < best_score:
                    best_score = sc
                    best_victim = u
            
            if best_victim is None:
                # Fallback: pick any non-excluded vertex
                for u in fast_mem:
                    if u != exclude:
                        best_victim = u
                        break
                if best_victim is None:
                    break
            
            operations.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_mem.discard(best_victim)
            lru_remove(best_victim)

    # Initialize LRU order with inputs
    for v in input_set:
        lru_touch(v)

    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break

        parents_list = parents.get(v, [])

        # Load any red parents
        for p in parents_list:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
                lru_touch(p)

        # Evict if over budget (exclude v since we're about to compute it)
        evict_if_needed(exclude=v, current_idx=idx)

        # Compute v if not already blue
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
            lru_touch(v)

        # After computing v, decrement remaining_uses of its parents
        for p in parents_list:
            remaining_uses[p] -= 1

        # Evict if over budget after compute
        evict_if_needed(exclude=None, current_idx=idx)

    # Ensure all outputs are blue at the end
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
            lru_touch(v)
        evict_if_needed(exclude=None, current_idx=len(topo_order) - 1)

    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
