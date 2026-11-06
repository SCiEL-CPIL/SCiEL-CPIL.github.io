import time
import random
from itertools import combinations

N = 20
C = 20
DEADLINE = 145.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def is_rainbow(col_edges):
    adj = make_adj(col_edges)
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for (x,y,z,w) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            cols = {adj[x][y], adj[y][z], adj[z][w], adj[w][x]}
                            if len(cols) != 4:
                                return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, adj_list, u, v, c):
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    for x in adj_list[u]:
        if x == v:
            continue
        ux_col = adj[u][x]
        for y in adj_list[x]:
            if y == u or y == x:
                continue
            if adj[y][v] < 0:
                continue
            xy_col = adj[x][y]
            yv_col = adj[y][v]
            if c == ux_col or c == xy_col or c == yv_col or ux_col == xy_col or ux_col == yv_col or xy_col == yv_col:
                return False
    for x in adj_list[v]:
        if x == u:
            continue
        vx_col = adj[v][x]
        for y in adj_list[x]:
            if y == v or y == x:
                continue
            if adj[y][u] < 0:
                continue
            xy_col = adj[x][y]
            yu_col = adj[y][u]
            if c == vx_col or c == xy_col or c == yu_col or vx_col == xy_col or vx_col == yu_col or xy_col == yu_col:
                return False
    return True

def _add_edge(col_edges, adj, adj_list, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c
    adj_list[u].append(v)
    adj_list[v].append(u)

def _remove_edge(col_edges, adj, adj_list, u, v):
    for i, e in enumerate(col_edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            col_edges.pop(i)
            adj[u][v] = -1
            adj[v][u] = -1
            adj_list[u].remove(v)
            adj_list[v].remove(u)
            return True
    return False

def mis_branch_and_bound():
    """
    MIS formulation:
    - Vertices: all (u, v, c) triples with u < v, c in [0, C)
    - Two vertices conflict if:
      * They share an endpoint (same u or same v)
      * They form a non-rainbow 4-cycle with two other selected edges
    We use branch-and-bound with greedy coloring upper bound.
    """
    if elapsed() > DEADLINE - 30:
        return []
    
    # Build conflict graph incrementally
    # Total vertices: N*(N-1)/2 * C = 190 * 20 = 3800
    # This is large, so we use a smarter approach
    
    # Alternative: use the greedy + local search but with better heuristics
    # Let's try a different structured approach first
    
    return []

def greedy_construct(seed_val):
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 5:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
                break
    
    return col_edges

def bipartite_greedy(seed_val):
    """
    Focus on bipartite subgraphs. A C4 in a bipartite graph K_{m,n} is simpler.
    Try to build dense bipartite graphs with rainbow C4s.
    """
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    # Try different bipartitions
    best_partition_result = []
    
    for split in range(5, 16):
        if elapsed() > DEADLINE - 10:
            break
        A = list(range(split))
        B = list(range(split, N))
        
        # Shuffle within parts
        random.shuffle(A)
        random.shuffle(B)
        
        result = []
        adj2 = [[-1]*N for _ in range(N)]
        adj_list2 = [[] for _ in range(N)]
        
        # Try all cross edges in random order
        cross_pairs = [(a, b) for a in A for b in B]
        random.shuffle(cross_pairs)
        
        for u, v in cross_pairs:
            if elapsed() > DEADLINE - 5:
                break
            if adj2[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj2, adj_list2, u, v, c):
                    _add_edge(result, adj2, adj_list2, u, v, c)
                    break
        
        if len(result) > len(best_partition_result):
            best_partition_result = result
    
    # Also try adding some within-part edges to the best bipartite result
    if best_partition_result:
        adj2 = make_adj(best_partition_result)
        adj_list2 = [[] for _ in range(N)]
        for u, v, c in best_partition_result:
            adj_list2[u].append(v)
            adj_list2[v].append(u)
        
        # Try adding within-part edges
        within_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if adj2[i][j] < 0:
                    within_pairs.append((i, j))
        random.shuffle(within_pairs)
        
        for u, v in within_pairs:
            if elapsed() > DEADLINE - 5:
                break
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj2, adj_list2, u, v, c):
                    _add_edge(best_partition_result, adj2, adj_list2, u, v, c)
                    break
    
    return best_partition_result

def local_improve(col_edges):
    adj = make_adj(col_edges)
    adj_list = [[] for _ in range(N)]
    for u, v, c in col_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Add edges greedily
    for attempt in range(20):
        if elapsed() > DEADLINE - 10:
            break
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                return col_edges
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, adj_list, u, v, c):
                    _add_edge(col_edges, adj, adj_list, u, v, c)
                    improved = True
                    break
        if not improved:
            break
    
    # Phase 2: Color-reassignment + edge swap local search
    for swap_attempt in range(300):
        if elapsed() > DEADLINE - 5:
            return col_edges
        
        missing = [(i, j) for i, j in all_pairs if adj[i][j] < 0]
        if not missing:
            return col_edges
        
        if not col_edges:
            return col_edges
        
        u_add, v_add = random.choice(missing)
        
        colours = list(range(C))
        random.shuffle(colours)
        added = False
        for c in colours:
            if _try_add(adj, adj_list, u_add, v_add, c):
                _add_edge(col_edges, adj, adj_list, u_add, v_add, c)
                added = True
                break
        
        if added:
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if elapsed() > DEADLINE - 5:
                    break
                if adj[u][v] >= 0:
                    continue
                colours2 = list(range(C))
                random.shuffle(colours2)
                for c in colours2:
                    if _try_add(adj, adj_list, u, v, c):
                        _add_edge(col_edges, adj, adj_list, u, v, c)
                        break
            continue
        
        # Try color reassignment
        idx = random.randint(0, len(col_edges) - 1)
        u_rem, v_rem, c_rem = col_edges[idx]
        
        new_colours = [c for c in range(C) if c != c_rem]
        random.shuffle(new_colours)
        
        found_reassign = False
        for new_c in new_colours:
            old_c = adj[u_rem][v_rem]
            adj[u_rem][v_rem] = new_c
            adj[v_rem][u_rem] = new_c
            col_edges[idx][2] = new_c
            
            colours3 = list(range(C))
            random.shuffle(colours3)
            can_add = False
            for c in colours3:
                if _try_add(adj, adj_list, u_add, v_add, c):
                    _add_edge(col_edges, adj, adj_list, u_add, v_add, c)
                    can_add = True
                    break
            
            if can_add:
                found_reassign = True
                random.shuffle(all_pairs)
                for u, v in all_pairs:
                    if elapsed() > DEADLINE - 5:
                        break
                    if adj[u][v] >= 0:
                        continue
                    colours4 = list(range(C))
                    random.shuffle(colours4)
                    for c in colours4:
                        if _try_add(adj, adj_list, u, v, c):
                            _add_edge(col_edges, adj, adj_list, u, v, c)
                            break
                break
            else:
                adj[u_rem][v_rem] = old_c
                adj[v_rem][u_rem] = old_c
                col_edges[idx][2] = old_c
        
        if not found_reassign:
            # Try 2-edge swap: remove one edge, add two missing edges
            if len(missing) >= 2:
                idx2 = random.randint(0, len(col_edges) - 1)
                u_rem2, v_rem2, c_rem2 = col_edges[idx2]
                
                # Pick two missing edges
                m1, m2 = random.sample(missing, 2)
                
                _remove_edge(col_edges, adj, adj_list, u_rem2, v_rem2)
                
                success = False
                for m in [m1, m2]:
                    ua, va = m
                    colours5 = list(range(C))
                    random.shuffle(colours5)
                    for c in colours5:
                        if _try_add(adj, adj_list, ua, va, c):
                            _add_edge(col_edges, adj, adj_list, ua, va, c)
                            success = True
                            break
                
                if success:
                    random.shuffle(all_pairs)
                    for u, v in all_pairs:
                        if elapsed() > DEADLINE - 5:
                            break
                        if adj[u][v] >= 0:
                            continue
                        colours6 = list(range(C))
                        random.shuffle(colours6)
                        for c in colours6:
                            if _try_add(adj, adj_list, u, v, c):
                                _add_edge(col_edges, adj, adj_list, u, v, c)
                                break
                else:
                    _add_edge(col_edges, adj, adj_list, u_rem2, v_rem2, c_rem2)
    
    return col_edges

def try_structured_coloring():
    """Try the (u+v) mod C coloring on complete graph."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u + v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring2():
    """Try (u-v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u - v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring3():
    """Try (u*v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring4():
    """Try (u*7 + v*3) mod C - a linear combination."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * 7 + v * 3) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring5():
    """Try (u^2 + v) mod C."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * u + v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def construct():
    best = []
    best_count = 0
    
    # Try structured colorings first
    for fn in [try_structured_coloring, try_structured_coloring2, try_structured_coloring3,
               try_structured_coloring4, try_structured_coloring5]:
        if elapsed() > DEADLINE - 40:
            break
        result = fn()
        count = count_edges(result)
        if count > best_count:
            best = [e[:] for e in result]
            best_count = count
    
    # Try bipartite-focused approaches
    for i in range(10):
        if elapsed() > DEADLINE - 30:
            break
        result = bipartite_greedy(1000 + i)
        count = count_edges(result)
        if count > best_count:
            best = [e[:] for e in result]
            best_count = count
    
    # Run multiple greedy attempts with local improvement
    max_attempts = 30
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 20:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        if count > best_count:
            best = [e[:] for e in result]
            best_count = count
    
    # Final polish: try to add more edges to the best solution
    if best and elapsed() < DEADLINE - 15:
        improved = local_improve([e[:] for e in best])
        improved_count = count_edges(improved)
        if improved_count > best_count:
            best = improved
            best_count = improved_count
    
    # Additional local improvement passes
    if best and elapsed() < DEADLINE - 8:
        improved2 = local_improve([e[:] for e in best])
        improved2_count = count_edges(improved2)
        if improved2_count > best_count:
            best = improved2
            best_count = improved2_count
    
    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback: try a simple valid coloring
    star = [[0, v, v % C] for v in range(1, N)]
    if is_rainbow(star):
        return star
    
    # Last resort: empty graph
    return []
