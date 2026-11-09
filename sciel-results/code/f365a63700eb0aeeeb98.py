import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def calc_score(adj, degs):
        """Compute the maximum degree sum over all triangles."""
        max_sum = 0
        for u in range(n):
            if not adj[u]: continue
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    def generate_baseline():
        """K_{8,8} plus edge (0,1) as initial solution."""
        adj = [0] * n
        degs = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        degs[0] += 1
        degs[1] += 1
        return adj, degs

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def get_edge_nonedge_lists(adj):
        """Return lists of current edges and non-edges."""
        edge_list = []
        nonedge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edge_list.append((u, v))
                else:
                    nonedge_list.append((u, v))
        return edge_list, nonedge_list

    def apply_swap(adj, degs, e_rm, e_add):
        u1, v1 = e_rm
        u2, v2 = e_add
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        degs[u1] -= 1
        degs[v1] -= 1
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        degs[u2] += 1
        degs[v2] += 1

    def revert_swap(adj, degs, e_rm, e_add):
        u1, v1 = e_rm
        u2, v2 = e_add
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        degs[u1] += 1
        degs[v1] += 1
        adj[u2] &= ~(1 << v2)
        adj[v2] &= ~(1 << u2)
        degs[u2] -= 1
        degs[v2] -= 1

    def find_worst_triangles(adj, degs, top_k=50):
        """Find triangles with highest degree sums, return list of (u,v,w,sum)."""
        triangles = []
        for u in range(n):
            if not adj[u]: continue
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        triangles.append((u, v, w, s))
                        common ^= lsb
        triangles.sort(key=lambda x: -x[3])
        return triangles[:top_k]

    def targeted_swap(adj, degs, worst_tris):
        """Pick a swap that targets edges in high-degree-sum triangles."""
        if not worst_tris:
            edge_list, nonedge_list = get_edge_nonedge_lists(adj)
            if not edge_list or not nonedge_list:
                return None, None
            return random.choice(edge_list), random.choice(nonedge_list)
        
        # Collect edges that appear in worst triangles
        edge_weights = {}
        for u, v, w, s in worst_tris:
            for e in [(u,v), (u,w), (v,w)]:
                e = tuple(sorted(e))
                edge_weights[e] = edge_weights.get(e, 0) + s
        
        edge_list, nonedge_list = get_edge_nonedge_lists(adj)
        if not edge_list or not nonedge_list:
            return None, None
        
        # Weighted random selection from edges in worst triangles
        weighted_edges = [(e, edge_weights.get(e, 1)) for e in edge_list]
        total_w = sum(w for _, w in weighted_edges)
        r = random.random() * total_w
        cum = 0
        e_rm = edge_list[0]
        for e, w in weighted_edges:
            cum += w
            if r <= cum:
                e_rm = e
                break
        
        # For non-edge, prefer those that would connect vertices in worst triangles
        tri_vertices = set()
        for u, v, w, s in worst_tris:
            tri_vertices.add(u)
            tri_vertices.add(v)
            tri_vertices.add(w)
        
        # Weight non-edges by how many triangle vertices they connect
        nonedge_weights = []
        for e in nonedge_list:
            u, v = e
            weight = 1
            if u in tri_vertices and v in tri_vertices:
                weight = 10
            elif u in tri_vertices or v in tri_vertices:
                weight = 5
            nonedge_weights.append((e, weight))
        
        total_nw = sum(w for _, w in nonedge_weights)
        r = random.random() * total_nw
        cum = 0
        e_add = nonedge_list[0]
        for e, w in nonedge_weights:
            cum += w
            if r <= cum:
                e_add = e
                break
        
        return e_rm, e_add

    def polish_search(adj, degs, score, max_iter=5000):
        """High-precision polish: only accept improving or equal moves."""
        current_adj = adj[:]
        current_degs = degs[:]
        current_score = score
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            worst_tris = find_worst_triangles(current_adj, current_degs, top_k=20)
            
            # Try targeted swaps
            for _ in range(20):
                e_rm, e_add = targeted_swap(current_adj, current_degs, worst_tris)
                if e_rm is None:
                    break
                
                apply_swap(current_adj, current_degs, e_rm, e_add)
                new_score = calc_score(current_adj, current_degs)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    revert_swap(current_adj, current_degs, e_rm, e_add)
                    if new_score == current_score:
                        # Keep equal moves to explore
                        pass
                    else:
                        revert_swap(current_adj, current_degs, e_rm, e_add)
            
            # Also try some random swaps
            for _ in range(5):
                edge_list, nonedge_list = get_edge_nonedge_lists(current_adj)
                if not edge_list or not nonedge_list:
                    break
                e_rm = random.choice(edge_list)
                e_add = random.choice(nonedge_list)
                
                apply_swap(current_adj, current_degs, e_rm, e_add)
                new_score = calc_score(current_adj, current_degs)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    revert_swap(current_adj, current_degs, e_rm, e_add)
        
        return current_adj, current_degs, current_score

    best_adj, best_degs = generate_baseline()
    best_score = calc_score(best_adj, best_degs)
    
    # Simulated annealing
    current_adj = best_adj[:]
    current_degs = best_degs[:]
    current_score = best_score
    
    initial_temp = 5.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    step = 0
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline:
        step += 1
        
        if no_improve > max_no_improve:
            # Restart with perturbation
            current_adj = best_adj[:]
            current_degs = best_degs[:]
            current_score = best_score
            no_improve = 0
            
            # Structured perturbation: target worst triangles
            worst_tris = find_worst_triangles(current_adj, current_degs, top_k=30)
            n_kicks = random.randint(8, 20)
            for _ in range(n_kicks):
                if time.time() > deadline: break
                e_rm, e_add = targeted_swap(current_adj, current_degs, worst_tris)
                if e_rm is None: break
                apply_swap(current_adj, current_degs, e_rm, e_add)
            
            current_score = calc_score(current_adj, current_degs)
            initial_temp = max(initial_temp * 0.6, 1.0)
            continue
        
        # Choose between targeted and random swaps
        if random.random() < 0.7:
            worst_tris = find_worst_triangles(current_adj, current_degs, top_k=30)
            e_rm, e_add = targeted_swap(current_adj, current_degs, worst_tris)
        else:
            edge_list, nonedge_list = get_edge_nonedge_lists(current_adj)
            if not edge_list or not nonedge_list:
                continue
            e_rm = random.choice(edge_list)
            e_add = random.choice(nonedge_list)
        
        if e_rm is None:
            continue
        
        apply_swap(current_adj, current_degs, e_rm, e_add)
        new_score = calc_score(current_adj, current_degs)
        delta = new_score - current_score
        
        temp = max(initial_temp * (cooling_rate ** step), min_temp)
        
        if delta <= 0:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
                best_degs = current_degs[:]
        else:
            if random.random() < math.exp(-delta / temp):
                current_score = new_score
                no_improve = 0
            else:
                revert_swap(current_adj, current_degs, e_rm, e_add)
                no_improve += 1
    
    # Final polish pass
    if time.time() < deadline - 10:
        best_adj, best_degs, best_score = polish_search(best_adj, best_degs, best_score, max_iter=3000)
    
    return {'n': n, 'edges': get_edges(best_adj)}
