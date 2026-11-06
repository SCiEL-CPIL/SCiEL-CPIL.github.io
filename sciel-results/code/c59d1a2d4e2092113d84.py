"""Rewritten program: Simulated Annealing with fast bitmask triangle-free evaluation."""
import random
import time
import itertools

N = 18
ALL_MASK = (1 << N) - 1

def _edge_bit(u, v):
    if u > v:
        u, v = v, u
    return u * N + v

def _edge_exists(mask, u, v):
    if u > v:
        u, v = v, u
    return (mask >> (u * N + v)) & 1

def _toggle_edge(mask, u, v):
    if u > v:
        u, v = v, u
    return mask ^ (1 << (u * N + v))

def _build_adj(mask):
    """Build adjacency list from edge mask."""
    adj = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if _edge_exists(mask, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _is_k4_free(mask):
    """Check K4-free using bitmask operations. Returns True if no K4 exists."""
    adj = _build_adj(mask)
    for a in range(N):
        for b in range(a+1, N):
            if not (adj[a] >> b) & 1:
                continue
            # Find common neighbors of a and b (excluding a, b themselves)
            common = adj[a] & adj[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            # Check if any pair in common forms an edge
            while common:
                lsb = common & (-common)
                c = lsb.bit_length() - 1
                common ^= lsb
                rest = common & adj[c]
                if rest:
                    return False
    return True

def _max_triangle_free_size(mask):
    """Exact branch and bound: largest induced subgraph that is triangle-free."""
    best = [0]
    
    adj_mask = _build_adj(mask)
    
    def _can_add(selected_mask, v):
        """Check if vertex v can be added to selected_mask without creating a triangle."""
        common = selected_mask & adj_mask[v]
        # Check if any edge exists within common (which would form a triangle with v)
        while common:
            lsb = common & (-common)
            a = lsb.bit_length() - 1
            common ^= lsb
            if common & adj_mask[a]:
                return False
        return True
    
    def _branch(avail_mask, selected_mask):
        upper = bin(selected_mask).count('1') + bin(avail_mask).count('1')
        if upper <= best[0]:
            return
        if avail_mask == 0:
            sz = bin(selected_mask).count('1')
            if sz > best[0]:
                best[0] = sz
            return
        lsb = avail_mask & (-avail_mask)
        v = lsb.bit_length() - 1
        rest_avail = avail_mask ^ lsb
        if _can_add(selected_mask, v):
            _branch(rest_avail, selected_mask | lsb)
        _branch(rest_avail, selected_mask)
    
    _branch(ALL_MASK, 0)
    return best[0]

def _greedy_triangle_free(mask, trials=30):
    best = 0
    adj_mask = _build_adj(mask)
    for _ in range(trials):
        order = list(range(N))
        random.shuffle(order)
        selected = 0
        for v in order:
            common = selected & adj_mask[v]
            ok = True
            while common:
                lsb = common & (-common)
                a = lsb.bit_length() - 1
                common ^= lsb
                if common & adj_mask[a]:
                    ok = False
                    break
            if ok:
                selected |= (1 << v)
        size = bin(selected).count('1')
        if size > best:
            best = size
    return best

def _mask_to_edges(mask):
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if _edge_exists(mask, u, v):
                edges.append([u, v])
    return edges

def _make_k4_free(mask):
    """Remove edges to make graph K4-free."""
    guard = 0
    while not _is_k4_free(mask) and guard < 2000:
        guard += 1
        adj = _build_adj(mask)
        found = False
        for a in range(N):
            for b in range(a+1, N):
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b]
                common &= ~(1 << a)
                common &= ~(1 << b)
                # Check if common contains an edge
                c_list = []
                temp = common
                while temp:
                    lsb = temp & (-temp)
                    c = lsb.bit_length() - 1
                    temp ^= lsb
                    c_list.append(c)
                for i in range(len(c_list)):
                    for j in range(i+1, len(c_list)):
                        c, d = c_list[i], c_list[j]
                        if (adj[c] >> d) & 1:
                            # Found K4: a, b, c, d
                            edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                            e = random.choice(edges_in_k4)
                            mask = _toggle_edge(mask, e[0], e[1])
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return mask

def _initial_graph():
    """Create a good initial K4-free graph."""
    # Start with a relatively dense random graph
    mask = 0
    for u in range(N):
        for v in range(u+1, N):
            if random.random() < 0.45:
                mask |= (1 << (u * N + v))
    
    mask = _make_k4_free(mask)
    return mask

def construct():
    deadline = time.time() + 130  # Safety margin
    
    current_mask = _initial_graph()
    current_score = _max_triangle_free_size(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing with better neighborhood
    T = 1.5
    T_min = 0.005
    cooling = 0.999
    
    all_pairs = list(itertools.combinations(range(N), 2))
    
    # Track recent moves for diversity
    recent_moves = []
    
    while time.time() < deadline and T > T_min:
        # Choose move: 80% single edge, 20% double edge swap
        if random.random() < 0.8:
            u, v = random.choice(all_pairs)
            new_mask = _toggle_edge(current_mask, u, v)
        else:
            # Double edge swap: remove one edge, add another
            u1, v1 = random.choice(all_pairs)
            u2, v2 = random.choice(all_pairs)
            if (u1, v1) == (u2, v2):
                u, v = random.choice(all_pairs)
                new_mask = _toggle_edge(current_mask, u, v)
            else:
                # Remove edge (u1,v1) if present, add edge (u2,v2) if absent
                if _edge_exists(current_mask, u1, v1) and not _edge_exists(current_mask, u2, v2):
                    new_mask = _toggle_edge(current_mask, u1, v1)
                    new_mask = _toggle_edge(new_mask, u2, v2)
                elif _edge_exists(current_mask, u2, v2) and not _edge_exists(current_mask, u1, v1):
                    new_mask = _toggle_edge(current_mask, u2, v2)
                    new_mask = _toggle_edge(new_mask, u1, v1)
                else:
                    u, v = random.choice(all_pairs)
                    new_mask = _toggle_edge(current_mask, u, v)
        
        if not _is_k4_free(new_mask):
            T *= cooling
            continue
        
        # Quick greedy check before expensive exact evaluation
        greedy_score = _greedy_triangle_free(new_mask, trials=15)
        if greedy_score >= best_score:
            T *= cooling
            continue
        
        new_score = _max_triangle_free_size(new_mask)
        
        if new_score < current_score:
            current_mask = new_mask
            current_score = new_score
            if new_score < best_score:
                best_mask = new_mask
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                current_mask = new_mask
                current_score = new_score
        
        T *= cooling
    
    return _mask_to_edges(best_mask)
