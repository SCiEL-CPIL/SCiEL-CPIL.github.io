import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 120

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431213421324124312341231423214321"

    perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}

    visited = [False] * num_perms
    cycles = []
    for i in range(num_perms):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                p = perms[curr]
                next_p = p[1:] + (p[0],)
                curr = perm_to_idx[next_p]
            cycles.append(cycle)

    C = len(cycles)

    # Precompute distances between nodes of different cycles
    # dist_matrix[c1][c2][i1][i2] = dist(cycles[c1][i1], cycles[c2][i2])
    # This is too big.
    # Instead, precompute dist for all pairs of nodes?
    # We only need dist between nodes in different cycles.
    # But we don't know which pairs will be adjacent in the TSP.
    # So we need dist for ANY pair of nodes from ANY two cycles.
    # Actually, we need dist(node_a, node_b) for any a, b.
    # Since we only use it for adjacent cycles in the order, and the order changes,
    # we need dist for any pair of nodes that might be adjacent.
    # Any node can be adjacent to any other node.
    # So we need the full distance matrix?
    # 5040x5040 is 25M entries.
    # Let's try to precompute it using numpy if possible?
    # Or just compute on the fly if precomputation is too slow.
    
    # Let's estimate precomputation time again.
    # 25M distances.
    # If we use a loop in Python, it's slow.
    # But we can use the structure.
    # dist(a, b) depends on overlap.
    
    # Let's stick to on-the-fly but optimize the inner loop.
    
    def dist(a_idx, b_idx):
        a = perms[a_idx]
        b = perms[b_idx]
        max_k = 0
        # Unroll for n=7?
        # Check k=6, 5, 4, 3, 2, 1
        if a[1:] == b[:6]: return 1
        if a[2:] == b[:5]: return 2
        if a[3:] == b[:4]: return 3
        if a[4:] == b[:3]: return 4
        if a[5:] == b[:2]: return 5
        if a[6:] == b[:1]: return 6
        return 7

    order = list(range(C))
    entries = [0] * C
    exits = [0] * C

    def get_cost(order, entries, exits):
        total_cost = 0
        for c_idx in order:
            c = cycles[c_idx]
            e = entries[c_idx]
            x = exits[c_idx]
            total_cost += (x - e) % len(c)
        
        for i in range(C - 1):
            c1 = order[i]
            c2 = order[i + 1]
            x1 = cycles[c1][exits[c1]]
            e2 = cycles[c2][entries[c2]]
            total_cost += dist(x1, e2)
        return total_cost

    best_cost = get_cost(order, entries, exits)
    best_order = order[:]
    best_entries = entries[:]
    best_exits = exits[:]
    
    current_cost = best_cost
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline - 5:
        if random.random() < 0.5:
            i, j = random.sample(range(C), 2)
            new_order = order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_cost = get_cost(new_order, entries, exits)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                order = new_order
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = order[:]
                    best_entries = entries[:]
                    best_exits = exits[:]
        else:
            c_idx = random.choice(order)
            c_len = len(cycles[c_idx])
            if random.random() < 0.5:
                new_entries = entries[:]
                new_entries[c_idx] = random.randint(0, c_len - 1)
                new_cost = get_cost(order, new_entries, exits)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    entries = new_entries
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_order = order[:]
                        best_entries = entries[:]
                        best_exits = exits[:]
            else:
                new_exits = exits[:]
                new_exits[c_idx] = random.randint(0, c_len - 1)
                new_cost = get_cost(order, entries, new_exits)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    exits = new_exits
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_order = order[:]
                        best_entries = entries[:]
                        best_exits = exits[:]
                        
        T *= cooling
        if T < T_min:
            T = T_min

    word = []
    for i in range(len(best_order)):
        c_idx = best_order[i]
        c = cycles[c_idx]
        e = best_entries[c_idx]
        x = best_exits[c_idx]
        
        path_nodes = []
        curr = e
        while True:
            path_nodes.append(c[curr])
            if curr == x:
                break
            curr = (curr + 1) % len(c)
            
        for node_idx in path_nodes:
            p = perms[node_idx]
            if not word:
                word.extend(p)
            else:
                max_k = 0
                for k in range(1, n):
                    if word[-k:] == list(p[:k]):
                        max_k = k
                word.extend(list(p[max_k:]))

    return "".join(map(str, word))
