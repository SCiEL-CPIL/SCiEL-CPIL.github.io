"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
ALL_EDGES_SET = set(ALL_EDGES)
NUM_EDGES = len(ALL_EDGES)

# Precompute edge index for fast lookup
EDGE_IDX = {e: i for i, e in enumerate(ALL_EDGES)}

# Precompute: for each edge (u,v), the list of edges that share a vertex with it
# This helps in swap operations
EDGE_NEIGHBORS = [[] for _ in range(NUM_EDGES)]
for i, (u, v) in enumerate(ALL_EDGES):
    for j, (a, b) in enumerate(ALL_EDGES):
        if i == j:
            continue
        if a == u or a == v or b == u or b == v:
            EDGE_NEIGHBORS[i].append(j)


def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    
    When adding edge (u,v), a new K3,3 must use this edge. WLOG u in part A, v in part B.
    We need: two more vertices a1,a2 in A (adjacent to v) and two more b1,b2 in B (adjacent to u)
    such that all cross edges exist.
    
    Equivalent: for pairs (x,y) both adjacent to u, check if |N(x) ∩ N(y) ∩ N(v)| >= 2.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Build list of neighbors of u (excluding v)
    x_list = []
    nu_mask = nu & ~(1 << v)
    for x in range(N):
        if x != u and (nu_mask >> x) & 1:
            x_list.append(x)
    
    # For each pair in N(u), check common neighbors in N(v)
    for i in range(len(x_list)):
        x = x_list[i]
        ax = adj[x]
        for j in range(i+1, len(x_list)):
            y = x_list[j]
            common = ax & adj[y] & nv
            if common.bit_count() >= 2:
                return False
    
    return True


def add_edge(u, v, adj):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def remove_edge(u, v, adj):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def count_edges(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2


def get_missing_edges(adj):
    missing = []
    for u, v in ALL_EDGES:
        if not ((adj[u] >> v) & 1):
            missing.append((u, v))
    return missing


def get_current_edges(adj):
    current = []
    for u, v in ALL_EDGES:
        if (adj[u] >> v) & 1:
            current.append((u, v))
    return current


def greedy_construct(adj, deadline):
    """Greedy edge addition with random order."""
    shuffled_edges = ALL_EDGES[:]
    random.shuffle(shuffled_edges)
    for u, v in shuffled_edges:
        if time.time() > deadline:
            break
        if is_valid_add(u, v, adj):
            add_edge(u, v, adj)
    return adj


def hill_climbing(adj, deadline):
    """
    Hill climbing with add, 1-swap (remove 1 add 2), and 2-swap (remove 2 add 3).
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing edge
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                return adj
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
                improved = True
                break
        
        if time.time() > deadline:
            return adj
        
        if improved:
            continue
        
        # Phase 2: Try 1-swap (remove 1 edge, add 2 edges)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for u1, v1 in current:
            if time.time() > deadline:
                return adj
            
            remove_edge(u1, v1, adj)
            
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            added = []
            for u2, v2 in missing:
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if is_valid_add(u2, v2, adj):
                    add_edge(u2, v2, adj)
                    added.append((u2, v2))
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                improved = True
                found = True
                break
            else:
                for u2, v2 in added:
                    remove_edge(u2, v2, adj)
                add_edge(u1, v1, adj)
        
        if time.time() > deadline:
            return adj
        if found:
            continue
        
        # Phase 3: Try 2-swap (remove 2 edges, add 3 edges)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for i in range(len(current)):
            if time.time() > deadline:
                return adj
            u1, v1 = current[i]
            for j in range(i+1, len(current)):
                if time.time() > deadline:
                    return adj
                u2, v2 = current[j]
                
                remove_edge(u1, v1, adj)
                remove_edge(u2, v2, adj)
                
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                added = []
                for u3, v3 in missing:
                    if (u3, v3) in [(u1, v1), (v1, u1), (u2, v2), (v2, u2)]:
                        continue
                    if is_valid_add(u3, v3, adj):
                        add_edge(u3, v3, adj)
                        added.append((u3, v3))
                        if len(added) == 3:
                            break
                
                if len(added) == 3:
                    improved = True
                    found = True
                    break
                else:
                    for u3, v3 in added:
                        remove_edge(u3, v3, adj)
                    add_edge(u1, v1, adj)
                    add_edge(u2, v2, adj)
            
            if found:
                break
        
        if time.time() > deadline:
            return adj

    return adj


def simulated_annealing(adj, deadline, start_temp=1.0, cool_rate=0.995):
    """
    Simulated annealing with edge add/remove/swap moves.
    """
    current_adj = adj[:]
    current_count = count_edges(current_adj)
    best_adj = current_adj[:]
    best_count = current_count
    temp = start_temp
    
    while time.time() < deadline and temp > 0.01:
        # Generate a random move
        move_type = random.random()
        if move_type < 0.3:
            # Add a random missing edge
            missing = get_missing_edges(current_adj)
            if not missing:
                temp *= cool_rate
                continue
            u, v = random.choice(missing)
            if is_valid_add(u, v, current_adj):
                add_edge(u, v, current_adj)
                new_count = current_count + 1
            else:
                temp *= cool_rate
                continue
        elif move_type < 0.6:
            # Remove a random edge
            current = get_current_edges(current_adj)
            if not current:
                temp *= cool_rate
                continue
            u, v = random.choice(current)
            remove_edge(u, v, current_adj)
            new_count = current_count - 1
        else:
            # Swap: remove one edge, try to add another
            current = get_current_edges(current_adj)
            missing = get_missing_edges(current_adj)
            if not current or not missing:
                temp *= cool_rate
                continue
            u1, v1 = random.choice(current)
            u2, v2 = random.choice(missing)
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                temp *= cool_rate
                continue
            remove_edge(u1, v1, current_adj)
            if is_valid_add(u2, v2, current_adj):
                add_edge(u2, v2, current_adj)
                new_count = current_count
            else:
                add_edge(u1, v1, current_adj)
                temp *= cool_rate
                continue
        
        # Accept or reject
        delta = new_count - current_count
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
        else:
            # Undo the move
            if move_type < 0.3:
                remove_edge(u, v, current_adj)
            elif move_type < 0.6:
                add_edge(u, v, current_adj)
            else:
                # Undo swap
                remove_edge(u2, v2, current_adj)
                add_edge(u1, v1, current_adj)
        
        temp *= cool_rate
    
    return best_adj, best_count


def branch_and_bound(adj, deadline, max_depth=5):
    """
    Branch and bound on edge selection.
    """
    current_adj = adj[:]
    current_count = count_edges(current_adj)
    
    # Get missing edges sorted by some heuristic
    missing = get_missing_edges(current_adj)
    random.shuffle(missing)
    
    best_count = current_count
    best_adj = current_adj[:]
    
    def bnb(adj, depth, deadline):
        nonlocal best_count, best_adj
        if time.time() > deadline or depth <= 0:
            return
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Try adding each missing edge
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        
        for u, v in missing[:10]:  # Limit branching factor
            if time.time() > deadline:
                return
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
                bnb(adj, depth - 1, deadline)
                remove_edge(u, v, adj)
        
        # Try swapping
        current = get_current_edges(adj)
        random.shuffle(current)
        
        for u1, v1 in current[:5]:
            if time.time() > deadline:
                return
            remove_edge(u1, v1, adj)
            
            missing2 = get_missing_edges(adj)
            random.shuffle(missing2)
            
            for u2, v2 in missing2[:10]:
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if is_valid_add(u2, v2, adj):
                    add_edge(u2, v2, adj)
                    bnb(adj, depth - 1, deadline)
                    remove_edge(u2, v2, adj)
            
            add_edge(u1, v1, adj)
    
    bnb(current_adj, max_depth, deadline)
    
    return best_adj, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    best_adj = None
    best_count = 0
    
    # Try multiple random restarts with different strategies
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        
        adj = [0] * N
        
        # Strategy 1: Greedy + Hill Climbing
        if restart % 3 == 1:
            adj = greedy_construct(adj, deadline - 10)
            adj = hill_climbing(adj, deadline - 5)
        
        # Strategy 2: Greedy + Simulated Annealing
        elif restart % 3 == 2:
            adj = greedy_construct(adj, deadline - 15)
            adj, count = simulated_annealing(adj, deadline - 5)
        
        # Strategy 3: Greedy + Branch and Bound
        else:
            adj = greedy_construct(adj, deadline - 20)
            adj, count = branch_and_bound(adj, deadline - 5, max_depth=3)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Early stop if we hit a high threshold
        if best_count >= 85:
            break
        
        # If time is running low, do a final hill climb on best
        if time.time() > deadline - 15:
            break
    
    # Final hill climb on best solution
    if best_adj is not None and time.time() < deadline:
        best_adj = hill_climbing(best_adj, deadline)
        best_count = count_edges(best_adj)
    
    if best_adj is None:
        return []
        
    # Convert adj back to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

# EVOLVE-BLOCK-END
