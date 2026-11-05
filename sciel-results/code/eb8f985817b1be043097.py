import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def is_valid(adj):
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return False
        return True

    def find_violation(adj):
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k), common
        return None, None

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def cleanse(adj, max_steps=800):
        """Remove edges to make graph valid, targeting highest-degree vertices in violating triples."""
        steps = 0
        while steps < max_steps:
            viol_triple, viol_common = find_violation(adj)
            if viol_triple is None:
                break
            
            u, v, w = viol_triple
            cn_list = [x for x in range(n) if (viol_common >> x) & 1]
            if not cn_list:
                break
            
            # Pick the vertex in the triple with highest degree to remove edge from
            degs = [(adj[u].bit_count(), u), (adj[v].bit_count(), v), (adj[w].bit_count(), w)]
            degs.sort(reverse=True)
            # With some probability pick the top, otherwise random for diversity
            if random.random() < 0.7:
                tv = degs[0][1]
            else:
                tv = random.choice([u, v, w])
            
            # Pick a common neighbor that also has high degree
            cn_degs = sorted(cn_list, key=lambda x: adj[x].bit_count(), reverse=True)
            if random.random() < 0.6:
                cn = cn_degs[0]
            else:
                cn = random.choice(cn_list)
            
            adj[tv] &= ~(1 << cn)
            adj[cn] &= ~(1 << tv)
            steps += 1
        
        return find_violation(adj)[0] is None

    def greedy_add(adj, current_edges):
        """Try to add missing edges greedily."""
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() - start_time > timeout - 5:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    current_edges += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        return current_edges

    def swap_search(adj, current_edges, max_swaps=200):
        """Try structured swaps: remove one edge, add two edges sharing a vertex."""
        for _ in range(max_swaps):
            if time.time() - start_time > timeout - 5:
                break
            
            existing = [(i, j) for i, j in all_pairs if (adj[i] >> j) & 1]
            missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
            if not existing or len(missing) < 2:
                break
            
            # Pick an edge to remove
            u_rem, v_rem = random.choice(existing)
            
            # Find missing edges that share a vertex with the removed edge
            # This creates a more structured move
            candidates_u = [(u_rem, x) for x in range(n) if x != u_rem and x != v_rem and not (adj[u_rem] >> x) & 1 and x > u_rem]
            candidates_u2 = [(x, u_rem) for x in range(n) if x != u_rem and x != v_rem and not (adj[u_rem] >> x) & 1 and x < u_rem]
            candidates_v = [(v_rem, x) for x in range(n) if x != u_rem and x != v_rem and not (adj[v_rem] >> x) & 1 and x > v_rem]
            candidates_v2 = [(x, v_rem) for x in range(n) if x != u_rem and x != v_rem and not (adj[v_rem] >> x) & 1 and x < v_rem]
            
            shared_missing = candidates_u + candidates_u2 + candidates_v + candidates_v2
            
            if len(shared_missing) >= 2:
                e_add1, e_add2 = random.sample(shared_missing, 2)
            else:
                # Fallback to random missing edges
                e_add1, e_add2 = random.sample(missing, 2)
            
            # Apply swap
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            
            u1, v1 = e_add1
            u2, v2 = e_add2
            
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            if is_valid(adj):
                current_edges += 1
                return current_edges, True
            else:
                # Revert
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
        
        return current_edges, False

    best_edges = 0
    best_adj = [0] * n
    
    while time.time() - start_time < timeout - 5:
        # 1. Initialize with varying density
        p = random.uniform(0.45, 0.75)
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # 2. Cleanse
        if not cleanse(adj):
            # Try harder
            if not cleanse(adj, max_steps=2000):
                continue
        
        current_edges = count_edges(adj)
        
        # 3. Improve: greedy add then swap search
        current_edges = greedy_add(adj, current_edges)
        
        # Multiple rounds of swap search
        for _ in range(3):
            if time.time() - start_time > timeout - 5:
                break
            current_edges, swapped = swap_search(adj, current_edges, max_swaps=150)
            if not swapped:
                break
            # After a successful swap, try greedy add again
            current_edges = greedy_add(adj, current_edges)
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [a for a in adj]
        
        if time.time() - start_time > timeout - 8:
            break

    edges = []
    for i, j in all_pairs:
        if (best_adj[i] >> j) & 1:
            edges.append([i, j])
    
    return edges
