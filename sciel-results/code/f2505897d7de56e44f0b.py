import time
import random
import math

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
ALL_EDGES_SET = set(ALL_EDGES)

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    
    When adding edge (u,v), the new K3,3 must use this edge. So u and v are in
    opposite parts. WLOG u is in part A, v is in part B.
    Then we need: two more vertices a1,a2 in A (adjacent to v and to b1,b2 in B)
    and two more vertices b1,b2 in B (adjacent to u and to a1,a2 in A).
    
    Equivalent check: for pairs (x,y) in N(u) (i.e., x,y adjacent to u),
    check if |N(x) ∩ N(y) ∩ N(v)| >= 2.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate over vertices x in N(u)
    x_list = []
    for x in range(N):
        if x != u and x != v and (nu >> x) & 1:
            x_list.append(x)
    
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

def compute_upper_bound(adj, deadline):
    """
    Compute an upper bound on the number of edges we can add to adj
    while maintaining K3,3-freeness.
    
    Uses a greedy packing argument: for each missing edge, check if it can be
    added. The upper bound is current_edges + max_addable.
    """
    current_count = count_edges(adj)
    missing = get_missing_edges(adj)
    random.shuffle(missing)
    
    addable = 0
    for u, v in missing:
        if time.time() > deadline:
            break
        if is_valid_add(u, v, adj):
            addable += 1
    
    return current_count + addable

def branch_and_bound(adj, deadline, best_count_ref):
    """
    Branch-and-bound search: try to add edges systematically, pruning branches
    that cannot beat the current best.
    """
    current_count = count_edges(adj)
    
    if current_count >= best_count_ref[0]:
        return adj
    
    # Upper bound: if even adding all missing edges can't beat best, prune
    ub = compute_upper_bound(adj, deadline)
    if ub <= best_count_ref[0]:
        return adj
    
    missing = get_missing_edges(adj)
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline:
            return adj
        
        if is_valid_add(u, v, adj):
            add_edge(u, v, adj)
            result = branch_and_bound(adj, deadline, best_count_ref)
            remove_edge(u, v, adj)
            
            if count_edges(result) > count_edges(adj):
                return result
    
    return adj

def tabu_search(adj, deadline, best_adj_ref, best_count_ref):
    """
    Tabu search with adaptive tabu tenure to escape local optima.
    """
    tabu_list = {}  # edge -> iteration when it becomes non-tabu
    tabu_tenure = 10
    best_local_adj = adj[:]
    best_local_count = count_edges(adj)
    iteration = 0
    no_improve = 0
    
    while time.time() < deadline:
        iteration += 1
        no_improve += 1
        
        if no_improve > 30:
            # Reset tabu tenure and try different strategy
            tabu_tenure = max(5, tabu_tenure // 2)
            no_improve = 0
        
        # Get all possible moves: add missing edge, or 1-swap, or 2-swap
        moves = []
        
        # Add moves
        missing = get_missing_edges(adj)
        for u, v in missing:
            if is_valid_add(u, v, adj):
                moves.append(('add', u, v))
        
        # 1-swap moves: remove one edge, add two
        current = get_current_edges(adj)
        for u1, v1 in current:
            saved = adj[:]
            remove_edge(u1, v1, adj)
            
            for u2, v2 in missing:
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if is_valid_add(u2, v2, adj):
                    add_edge(u2, v2, adj)
                    # Try to add one more
                    for u3, v3 in missing:
                        if (u3, v3) in [(u1, v1), (v1, u1), (u2, v2), (v2, u2)]:
                            continue
                        if is_valid_add(u3, v3, adj):
                            moves.append(('swap1', u1, v1, u2, v2, u3, v3))
                            break
                    remove_edge(u2, v2, adj)
            
            add_edge(u1, v1, adj)
        
        if not moves:
            break
        
        # Select best non-tabu move
        best_move = None
        best_score = -1
        
        for move in moves:
            if move[0] == 'add':
                _, u, v = move
                key = (u, v) if u < v else (v, u)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                score = 1  # Adding an edge always improves by 1
            elif move[0] == 'swap1':
                _, u1, v1, u2, v2, u3, v3 = move
                key1 = (u1, v1) if u1 < v1 else (v1, u1)
                key2 = (u2, v2) if u2 < v2 else (v2, u2)
                key3 = (u3, v3) if u3 < v3 else (v3, u3)
                if (key1 in tabu_list and tabu_list[key1] > iteration) or \
                   (key2 in tabu_list and tabu_list[key2] > iteration) or \
                   (key3 in tabu_list and tabu_list[key3] > iteration):
                    continue
                score = 1  # Net gain of 1 edge
        
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            # All moves are tabu; allow the best tabu move (aspiration criterion)
            for move in moves:
                if move[0] == 'add':
                    _, u, v = move
                    key = (u, v) if u < v else (v, u)
                    if count_edges(adj) + 1 > best_count_ref[0]:
                        best_move = move
                        break
                elif move[0] == 'swap1':
                    _, u1, v1, u2, v2, u3, v3 = move
                    if count_edges(adj) + 1 > best_count_ref[0]:
                        best_move = move
                        break
        
        if best_move is None:
            break
        
        # Apply the move
        if best_move[0] == 'add':
            _, u, v = best_move
            add_edge(u, v, adj)
            tabu_list[(u, v) if u < v else (v, u)] = iteration + tabu_tenure
        elif best_move[0] == 'swap1':
            _, u1, v1, u2, v2, u3, v3 = best_move
            remove_edge(u1, v1, adj)
            add_edge(u2, v2, adj)
            add_edge(u3, v3, adj)
            tabu_list[(u1, v1) if u1 < v1 else (v1, u1)] = iteration + tabu_tenure
            tabu_list[(u2, v2) if u2 < v2 else (v2, u2)] = iteration + tabu_tenure
            tabu_list[(u3, v3) if u3 < v3 else (v3, u3)] = iteration + tabu_tenure
        
        current_count = count_edges(adj)
        if current_count > best_local_count:
            best_local_count = current_count
            best_local_adj = adj[:]
            no_improve = 0
        
        if current_count > best_count_ref[0]:
            best_count_ref[0] = current_count
            best_adj_ref[0] = adj[:]
    
    return best_local_adj

def structured_initialization():
    """
    Create a structured initial graph based on finite geometry ideas.
    For K3,3-free graphs, we can use a bipartite-like structure with
    controlled degrees to avoid K3,3.
    """
    adj = [0] * N
    
    # Create two sets of vertices: A = {0..9}, B = {10..19}
    # Add edges between them in a pattern that avoids K3,3
    # Use a cyclic pattern: connect i in A to (i+k) mod 10 in B for k in some set
    
    # Try a few different structured patterns
    patterns = [
        # Pattern 1: Connect each A vertex to 5 B vertices in a cyclic pattern
        lambda i, j: ((j - 10 - i) % 10) in [0, 1, 2, 3, 4],
        # Pattern 2: Use quadratic residues
        lambda i, j: ((j - 10 - i) % 10) in [0, 1, 4, 5, 6],
        # Pattern 3: Sparse pattern
        lambda i, j: ((j - 10 - i) % 10) in [0, 1, 3, 4, 7],
    ]
    
    best_adj = None
    best_count = 0
    
    for pattern in patterns:
        adj = [0] * N
        for i in range(10):
            for j in range(10, 20):
                if pattern(i, j):
                    add_edge(i, j, adj)
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    return best_adj if best_adj is not None else [0] * N

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_adj = None
    best_count = 0
    best_adj_ref = [None]
    best_count_ref = [0]
    
    # Phase 1: Structured initialization
    adj = structured_initialization()
    best_adj_ref[0] = adj[:]
    best_count_ref[0] = count_edges(adj)
    
    # Phase 2: Greedy completion
    missing = get_missing_edges(adj)
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 10:
            break
        if is_valid_add(u, v, adj):
            add_edge(u, v, adj)
    
    current_count = count_edges(adj)
    if current_count > best_count_ref[0]:
        best_count_ref[0] = current_count
        best_adj_ref[0] = adj[:]
    
    # Phase 3: Tabu search
    adj = tabu_search(adj, deadline - 5, best_adj_ref, best_count_ref)
    
    # Phase 4: Multiple restarts with different seeds
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random restart with greedy
        adj = [0] * N
        shuffled_edges = ALL_EDGES[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline - 10:
                break
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
        
        # Tabu search
        adj = tabu_search(adj, deadline - 5, best_adj_ref, best_count_ref)
    
    if best_adj_ref[0] is None:
        return []
        
    # Convert adj back to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj_ref[0][u] >> v) & 1:
                edges.append([u, v])
                
    return edges
