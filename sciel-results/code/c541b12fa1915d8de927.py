import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Build conflict graph: edge (u,v) if they share a bad triple
    # Use bitmasks for fast operations
    conflict = [0] * (N + 1)  # bitmask of conflicts for each node

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict[a] |= (1 << b)
                    conflict[a] |= (1 << c)
                    conflict[b] |= (1 << a)
                    conflict[b] |= (1 << c)
                    conflict[c] |= (1 << a)
                    conflict[c] |= (1 << b)

    if time.time() > deadline:
        return [1]

    # Greedy independent set (independent set in conflict graph = valid solution)
    def greedy_iset(order):
        selected = set()
        for v in order:
            if not (conflict[v] & sum(1 << x for x in selected)):
                # Check if v conflicts with any selected
                ok = True
                for s in selected:
                    if conflict[v] & (1 << s):
                        ok = False
                        break
                if ok:
                    selected.add(v)
        return selected

    def greedy_iset_fast(order, selected_mask):
        """selected_mask: bitmask of already selected"""
        selected = set()
        for v in order:
            if selected_mask & (1 << v):
                selected.add(v)
                continue
            if not (conflict[v] & selected_mask):
                selected.add(v)
                selected_mask |= (1 << v)
        return selected

    # Multiple greedy strategies to get initial lower bound
    candidates = []
    
    # Natural order
    candidates.append(greedy_iset_fast(list(range(1, N+1)), 0))
    # Reverse
    candidates.append(greedy_iset_fast(list(range(N, 0, -1)), 0))
    
    # Sorted by degree (ascending)
    deg = [bin(conflict[x]).count('1') for x in range(N+1)]
    order_deg_asc = sorted(range(1, N+1), key=lambda x: (deg[x], x))
    candidates.append(greedy_iset_fast(order_deg_asc, 0))
    
    # Sorted by degree (descending)
    order_deg_desc = sorted(range(1, N+1), key=lambda x: (-deg[x], x))
    candidates.append(greedy_iset_fast(order_deg_desc, 0))
    
    # Primes + 1
    primes = [1] + [p for p in range(2, N+1) if _is_prime(p)]
    candidates.append(greedy_iset_fast(primes, 0))
    
    # Random restarts
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        candidates.append(greedy_iset_fast(order, 0))
    
    # Weighted random
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        order_w = sorted(range(1, N+1), key=lambda x: (deg[x] + random.random() * 20, x))
        candidates.append(greedy_iset_fast(order_w, 0))
    
    # Pick best greedy
    best_set = max(candidates, key=len)
    best_size = len(best_set)
    best_mask = 0
    for x in best_set:
        best_mask |= (1 << x)
    
    # Branch and bound for maximum independent set
    # Color the candidate graph for upper bound
    def color_bound(candidates_list):
        """Greedy coloring of the subgraph induced by candidates_list.
        Returns number of colors (upper bound on max independent set)."""
        if not candidates_list:
            return 0
        color_class = []
        remaining = set(candidates_list)
        num_colors = 0
        while remaining:
            num_colors += 1
            # Greedy: pick vertices in order, assign to color class if no conflict with already in class
            this_class = []
            for v in sorted(remaining, key=lambda x: -deg[x]):
                # Check v has no conflict with any in this_class
                can_add = True
                for u in this_class:
                    if conflict[v] & (1 << u):
                        can_add = False
                        break
                if can_add:
                    this_class.append(v)
            for v in this_class:
                remaining.discard(v)
        return num_colors

    # Better coloring: process in order, assign smallest possible color
    def color_bound2(candidates_list):
        if not candidates_list:
            return 0
        # Sort by degree descending for tighter bound
        ordered = sorted(candidates_list, key=lambda x: -deg[x])
        color_of = {}
        color_classes = []  # list of sets
        for v in ordered:
            used = set()
            for u in color_classes:
                if any(conflict[v] & (1 << x) for x in u):
                    used.add(color_classes.index(u))
            # Find smallest available color
            c = 0
            while c in used:
                c += 1
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
            color_of[v] = c
        return len(color_classes)

    # Branch and bound
    best_solution = list(best_set)
    best_size_bb = best_size
    
    # Precompute: for each node, its neighbors as bitmask
    # Use recursive B&B with candidate list
    
    def bb_search(candidates, current_set, current_size, deadline):
        """Branch and bound. candidates: list of available nodes."""
        nonlocal best_size_bb, best_solution
        if time.time() > deadline:
            return
        if not candidates:
            if current_size > best_size_bb:
                best_size_bb = current_size
                best_solution = list(current_set)
            return
        # Upper bound: current_size + color bound on remaining
        ub = current_size + color_bound2(candidates)
        if ub <= best_size_bb:
            return
        
        # Pick a vertex with max degree in the subgraph
        # For efficiency, pick the one with most conflicts among candidates
        cand_set = set(candidates)
        best_v = None
        best_conf = -1
        for v in candidates:
            conf_in_cand = 0
            mask = 0
            for u in candidates:
                if u != v and (conflict[v] & (1 << u)):
                    conf_in_cand += 1
            if conf_in_cand > best_conf:
                best_conf = conf_in_cand
                best_v = v
        
        if best_v is None:
            return
        
        # Branch 1: exclude best_v
        bb_search([v for v in candidates if v != best_v], current_set, current_size, deadline)
        
        if time.time() > deadline:
            return
        
        # Branch 2: include best_v, remove all its neighbors
        current_set.add(best_v)
        neighbors_mask = conflict[best_v]
        new_cands = [v for v in candidates if v != best_v and not (neighbors_mask & (1 << v))]
        bb_search(new_cands, current_set, current_size + 1, deadline)
        current_set.discard(best_v)

    # Try B&B from the best greedy solution
    # Start with candidates = all nodes, but we can seed with best solution
    # Actually, let's do B&B from scratch but with the lower bound
    
    if time.time() < deadline - 10:
        # B&B on full problem
        all_nodes = list(range(1, N+1))
        bb_search(all_nodes, set(), 0, deadline - 5)
    
    # Local search: kick and refill
    def kick_and_refill(sel, deadline):
        sel_list = list(sel)
        n = len(sel_list)
        sel_set = set(sel_list)
        
        def refill(trial_set):
            trial_list = list(trial_set)
            for v in range(1, N+1):
                if v in trial_set:
                    continue
                # Check v doesn't conflict with any in trial
                ok = True
                for s in trial_set:
                    if conflict[v] & (1 << s):
                        ok = False
                        break
                if ok:
                    trial_set.add(v)
            return trial_set
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for kick_size in [1, 2, 3]:
                if time.time() > deadline:
                    break
                if n < kick_size:
                    continue
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    remove = random.sample(sel_list, min(kick_size, n))
                    trial = sel_set - set(remove)
                    trial = refill(trial)
                    if len(trial) > n:
                        sel_set = trial
                        sel_list = list(trial)
                        n = len(sel_list)
                        improved = True
                        break
        return sel_list
    
    if time.time() < deadline - 3:
        result_set = kick_and_refill(set(best_solution), deadline)
        if len(result_set) > best_size_bb:
            best_solution = list(result_set)
            best_size_bb = len(result_set)
    
    # Final validation
    def is_valid(sel):
        s = set(sel)
        if len(s) != len(sel) or len(s) == 0:
            return False
        for a, b, c in combinations(sorted(s), 3):
            if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                return False
        return True
    
    result = best_solution
    if not is_valid(result):
        # Fall back to best greedy
        if not is_valid(list(best_set)):
            return [1]
        result = list(best_set)
    
    if not result:
        return [1]
    
    return result
