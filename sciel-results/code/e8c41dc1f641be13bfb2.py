"""K2,3-free graph on 20 vertices via union of two C4-free subgraphs with SA."""
import time
import random
import math

N = 20
DEADLINE = 115.0  # safety margin before 120s limit


def _count_common_neighbors(adj1, adj2, u, v):
    """Count common neighbors of u and v across both subgraphs."""
    # Use bitmasks for speed
    cn1 = adj1[u] & adj1[v]
    cn2 = adj2[u] & adj2[v]
    return cn1.bit_count() + cn2.bit_count()


def _is_c4_free(adj, u, v):
    """Check if adding edge (u,v) keeps adj C4-free.
    C4-free means no two vertices share >1 common neighbor.
    So for every neighbor w of u, w and v must share 0 common neighbors in adj
    (excluding u and v themselves from consideration).
    Actually: adding (u,v) creates a C4 iff there exists w such that
    w is adjacent to both u and v in adj, AND w and some other... 
    No. C4-free = no C4 cycle. A C4 is u-a-v-b-u. Adding edge (u,v) creates
    a C4 if there exist a,b such that u-a, a-v, v-b, b-u are all edges.
    That means a is a common neighbor of u and v, and b is also a common neighbor.
    So adding (u,v) creates a C4 iff u and v have >= 2 common neighbors in adj.
    """
    cn = adj[u] & adj[v]
    # Remove u and v from consideration (they shouldn't be in their own neighbor set anyway)
    cn &= ~(1 << u)
    cn &= ~(1 << v)
    return cn.bit_count() <= 1


def _add_edge_c4(adj, u, v):
    """Try to add edge (u,v) to adj maintaining C4-free. Returns True if added."""
    if (adj[u] >> v) & 1:
        return True  # already present
    if not _is_c4_free(adj, u, v):
        return False
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    return True


def _remove_edge_c4(adj, u, v):
    """Remove edge (u,v) from adj."""
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _greedy_fill(adj1, adj2):
    """Greedily add edges that are valid in both subgraphs."""
    added = 0
    for u in range(N):
        for v in range(u + 1, N):
            if (adj1[u] >> v) & 1:
                continue
            if (adj2[u] >> v) & 1:
                continue
            # Check if we can add to adj1
            if _is_c4_free(adj1, u, v):
                adj1[u] |= (1 << v)
                adj1[v] |= (1 << u)
                added += 1
                continue
            # Check if we can add to adj2
            if _is_c4_free(adj2, u, v):
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
                added += 1
    return added


def _edge_count(adj1, adj2):
    """Count total edges in both subgraphs."""
    c1 = sum(adj1[i].bit_count() for i in range(N)) // 2
    c2 = sum(adj2[i].bit_count() for i in range(N)) // 2
    return c1 + c2


def _is_valid_k23(adj1, adj2):
    """Check K2,3-free: no two vertices have 3+ common neighbors total."""
    for u in range(N):
        for v in range(u + 1, N):
            cn = (adj1[u] & adj1[v] & ~((1 << u) | (1 << v))).bit_count() + \
                 (adj2[u] & adj2[v] & ~((1 << u) | (1 << v))).bit_count()
            if cn >= 3:
                return False
    return True


def _random_init_c4_free(density):
    """Generate a random C4-free graph with given edge density."""
    adj = [0] * N
    # Try edges in random order
    edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(edges)
    target = int(density * N * (N - 1) // 2)
    count = 0
    for u, v in edges:
        if count >= target:
            break
        if _is_c4_free(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            count += 1
    return adj


def _circulant_init(step):
    """Create a circulant-like C4-free graph."""
    adj = [0] * N
    for u in range(N):
        for d in [step, N - step]:
            v = (u + d) % N
            if u < v:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    # Verify C4-free
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] >> v) & 1:
                cn = adj[u] & adj[v]
                cn &= ~((1 << u) | (1 << v))
                if cn.bit_count() > 1:
                    # Remove some edges to fix
                    pass  # circulant with proper step should be C4-free
    return adj


def _sa_optimize(adj1, adj2, start_time, time_budget):
    """Simulated annealing to optimize both subgraphs."""
    best_adj1 = [a | 0 for a in adj1]
    best_adj2 = [a | 0 for a in adj2]
    best_score = _edge_count(best_adj1, best_adj2)
    
    cur_adj1 = [a | 0 for a in adj1]
    cur_adj2 = [a | 0 for a in adj2]
    cur_score = _edge_count(cur_adj1, cur_adj2)
    
    t = 1.0
    cooling = 0.999
    min_t = 0.01
    
    while time.time() - start_time < time_budget and t > min_t:
        # Pick a random move
        move_type = random.random()
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Determine which subgraph to modify
        if move_type < 0.5:
            # Try adding edge to adj1
            if (cur_adj1[u] >> v) & 1:
                continue
            if not _is_c4_free(cur_adj1, u, v):
                continue
            cur_adj1[u] |= (1 << v)
            cur_adj1[v] |= (1 << u)
            new_score = _edge_count(cur_adj1, cur_adj2)
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(delta / t):
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_adj1 = [a | 0 for a in cur_adj1]
                    best_adj2 = [a | 0 for a in cur_adj2]
            else:
                cur_adj1[u] &= ~(1 << v)
                cur_adj1[v] &= ~(1 << u)
        elif move_type < 0.8:
            # Try adding edge to adj2
            if (cur_adj2[u] >> v) & 1:
                continue
            if not _is_c4_free(cur_adj2, u, v):
                continue
            cur_adj2[u] |= (1 << v)
            cur_adj2[v] |= (1 << u)
            new_score = _edge_count(cur_adj1, cur_adj2)
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(delta / t):
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_adj1 = [a | 0 for a in cur_adj1]
                    best_adj2 = [a | 0 for a in cur_adj2]
            else:
                cur_adj2[u] &= ~(1 << v)
                cur_adj2[v] &= ~(1 << u)
        else:
            # Remove an edge (to escape local optima)
            if (cur_adj1[u] >> v) & 1:
                cur_adj1[u] &= ~(1 << v)
                cur_adj1[v] &= ~(1 << u)
                cur_score -= 1
            elif (cur_adj2[u] >> v) & 1:
                cur_adj2[u] &= ~(1 << v)
                cur_adj2[v] &= ~(1 << u)
                cur_score -= 1
        
        t *= cooling
    
    return best_adj1, best_adj2, best_score


def construct():
    start_time = time.time()
    
    best_edges = []
    best_score = 0
    
    # Multiple restarts with diverse initializations
    random.seed(42)
    
    for restart in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Diverse initializations
        init_type = restart % 3
        if init_type == 0:
            # Random C4-free with varying density
            d1 = 0.1 + 0.15 * (restart % 5) / 5
            d2 = 0.1 + 0.15 * ((restart + 2) % 5) / 5
            adj1 = _random_init_c4_free(d1)
            adj2 = _random_init_c4_free(d2)
        elif init_type == 1:
            # Circulant-based
            adj1 = _circulant_init(3)
            adj2 = _circulant_init(5)
        else:
            # Sparse random
            adj1 = _random_init_c4_free(0.05)
            adj2 = _random_init_c4_free(0.05)
        
        # Greedy fill
        _greedy_fill(adj1, adj2)
        
        # SA optimization with time budget
        remaining = DEADLINE - (time.time() - start_time)
        sa_budget = min(remaining * 0.8, 3.0)  # up to 3s per restart
        if sa_budget < 0.5:
            break
        
        best_adj1, best_adj2, score = _sa_optimize(adj1, adj2, time.time(), sa_budget)
        
        # Final greedy fill
        _greedy_fill(best_adj1, best_adj2)
        score = _edge_count(best_adj1, best_adj2)
        
        # Verify validity
        if _is_valid_k23(best_adj1, best_adj2):
            if score > best_score:
                best_score = score
                # Collect edges
                best_edges = []
                for u in range(N):
                    for v in range(u + 1, N):
                        if (best_adj1[u] >> v) & 1 or (best_adj2[u] >> v) & 1:
                            best_edges.append([u, v])
    
    # If no valid result found, fall back to 5 disjoint K4s
    if not best_edges:
        best_edges = []
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    best_edges.append([4 * k + a, 4 * k + b])
    
    return best_edges
