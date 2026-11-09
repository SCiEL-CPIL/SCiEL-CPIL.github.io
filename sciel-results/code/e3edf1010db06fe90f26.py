import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Q3: 8 vertices, 12 edges
    # Standard labeling: vertex i, edges between i and i XOR 1, i XOR 2, i XOR 4
    q3_edges = set()
    for i in range(8):
        for d in [1, 2, 4]:
            j = i ^ d
            if i < j:
                q3_edges.add((i, j))
    q3_edges = list(q3_edges)
    # q3_edges has 12 elements

    # For each Q3 edge (a,b), precompute the other 11 edges
    edge_other = {}
    for (a, b) in q3_edges:
        others = [(x, y) for (x, y) in q3_edges if (x, y) != (a, b) and (y, x) != (a, b)]
        edge_other[(a, b)] = others
        edge_other[(b, a)] = others

    # Precompute all 8-vertex subsets containing each edge
    all_8_with_edge = {}
    for u in range(16):
        for v in range(u+1, 16):
            others = [x for x in range(16) if x != u and x != v]
            all_8_with_edge[(u,v)] = list(itertools.combinations(others, 6))

    all_pairs = set(itertools.combinations(range(16), 2))
    all_pairs_list = list(all_pairs)

    def check_q3_in_8set(adj_mask, verts8):
        """Check if 8 vertices form a Q3 subgraph. verts8 is a tuple of 8 vertex ids."""
        # Try all 8! / 2 assignments? No, too slow.
        # Instead: for each of C(8,4)=70 bipartitions, for each of 24*24=576 perms...
        # Actually: for each bipartition (A,B) of 8 into two sets of 4:
        #   For each permutation of A (24) and B (24): check all 12 edges
        #   Total: 70 * 24 * 24 = 40320 checks, each 12 edges = 483840 edge checks
        # Too slow. 
        # Better: fix the bipartition and only permute one side: 70 * 2 * 24 = 3360 checks
        # For each check, verify 12 edges. Total: 40320 edge checks. Still a lot.
        
        # Fastest: use the fact that Q3 is 3-regular. Each vertex in the 8-set must have
        # degree exactly 3 within the set. First filter by degree.
        degs = [0] * 8
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_mask[verts8[i]] >> verts8[j]) & 1:
                    degs[i] += 1
                    degs[j] += 1
        if not all(d == 3 for d in degs):
            return False
        
        # Now check if the 3-regular graph on 8 vertices is isomorphic to Q3
        # Q3 is the unique 3-regular bipartite graph on 8 vertices with girth 6
        # Check bipartiteness: try to 2-color
        # BFS coloring
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        bipartite = True
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if u == v:
                    continue
                if (adj_mask[verts8[u]] >> verts8[v]) & 1:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        bipartite = False
                        break
            if not bipartite:
                break
        
        if not bipartite:
            return False
        
        # Now check if it's Q3: the bipartite 3-regular graph on 8 vertices with parts of size 4
        # Q3 has girth 6. Check that there are no 4-cycles.
        # A 4-cycle exists if there are two vertices in one part with two common neighbors in the other part.
        part0 = [i for i in range(8) if color[i] == 0]
        part1 = [i for i in range(8) if color[i] == 1]
        
        # Check no 4-cycles: for each pair in part0, they should have at most 1 common neighbor in part1
        for i in range(len(part0)):
            for j in range(i+1, len(part0)):
                common = 0
                for k in part1:
                    if ((adj_mask[verts8[part0[i]]] >> verts8[k]) & 1) and ((adj_mask[verts8[part0[j]]] >> verts8[k]) & 1):
                        common += 1
                if common >= 2:
                    return False  # Has a 4-cycle, not Q3
        
        # If it's 3-regular, bipartite with parts of size 4, and no 4-cycles, it's Q3
        return True

    def try_add_edge(adj_mask, edge_set, u, v):
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

        for combo in all_8_with_edge[(u, v)]:
            verts8 = (u, v) + combo
            if check_q3_in_8set(adj_mask, verts8):
                edge_set.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        return True

    def try_remove_edge(adj_mask, edge_set, u, v):
        edge_set.remove((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    best_edges = set()
    best_adj = [0] * 16
    best_count = 0

    random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        init_density = 0.25 + 0.06 * restart
        current_edges = set()
        current_adj = [0] * 16
        
        for e in all_pairs_list:
            if random.random() < init_density:
                u, v = e
                current_edges.add(e)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        # Phase 1: Greedy addition
        non_edges = [e for e in all_pairs_list if e not in current_edges]
        
        for _round in range(25):
            if time.time() > deadline - 8:
                break
            changed = False
            non_edges.sort(key=lambda e: bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1'), reverse=True)
            for e in non_edges:
                if time.time() > deadline - 8:
                    break
                if e in current_edges:
                    continue
                u, v = e
                if try_add_edge(current_adj, current_edges, u, v):
                    changed = True
            if not changed:
                break
        
        # Phase 2: Local search
        no_improve = 0
        max_no_improve = 120
        
        while time.time() < deadline - 8 and no_improve < max_no_improve:
            move_type = random.random()
            
            if move_type < 0.6:
                non_edges_now = [e for e in all_pairs_list if e not in current_edges]
                if not non_edges_now:
                    no_improve += 1
                    continue
                weights = [bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1') + 1 for e in non_edges_now]
                e = random.choices(non_edges_now, weights=weights, k=1)[0]
                u, v = e
                if try_add_edge(current_adj, current_edges, u, v):
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
                        best_adj = current_adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            elif move_type < 0.85:
                if len(current_edges) < 10:
                    no_improve += 1
                    continue
                e_rm = random.choice(list(current_edges))
                u1, v1 = e_rm
                try_remove_edge(current_adj, current_edges, u1, v1)
                non_edges_now = [e for e in all_pairs_list if e not in current_edges]
                if non_edges_now:
                    weights = [bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1') + 1 for e in non_edges_now]
                    for _ in range(2):
                        if not non_edges_now:
                            break
                        e_add = random.choices(non_edges_now, weights=weights, k=1)[0]
                        if try_add_edge(current_adj, current_edges, e_add[0], e_add[1]):
                            non_edges_now.remove(e_add)
                            weights.pop(len(weights)-1)
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                    best_adj = current_adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if not current_edges:
                    no_improve += 1
                    continue
                e_rm = random.choice(list(current_edges))
                u1, v1 = e_rm
                try_remove_edge(current_adj, current_edges, u1, v1)
                non_edges_now = [e for e in all_pairs_list if e not in current_edges]
                if non_edges_now:
                    weights = [bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1') + 1 for e in non_edges_now]
                    e_add = random.choices(non_edges_now, weights=weights, k=1)[0]
                    if try_add_edge(current_adj, current_edges, e_add[0], e_add[1]):
                        if len(current_edges) > best_count:
                            best_count = len(current_edges)
                            best_edges = set(current_edges)
                            best_adj = current_adj[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        current_edges.add((u1, v1))
                        current_adj[u1] |= (1 << v1)
                        current_adj[v1] |= (1 << u1)
                        no_improve += 1
                else:
                    current_edges.add((u1, v1))
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    no_improve += 1

    # Final greedy polish
    final_edges = set(best_edges)
    final_adj = best_adj[:]
    non_edges_final = [e for e in all_pairs_list if e not in final_edges]
    non_edges_final.sort(key=lambda e: bin(final_adj[e[0]]).count('1') + bin(final_adj[e[1]]).count('1'), reverse=True)
    
    for e in non_edges_final:
        if time.time() > deadline - 2:
            break
        u, v = e
        try_add_edge(final_adj, final_edges, u, v)

    return [list(e) for e in final_edges]
