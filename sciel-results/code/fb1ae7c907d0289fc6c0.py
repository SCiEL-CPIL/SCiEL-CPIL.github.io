import time
import random
import math
from itertools import combinations

DEADLINE = 135  # Safety margin for 150s kill

def _compute_score(n, edges):
    """Compute the largest degree sum over triangles using bitmasks."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    max_triangle_sum = 0
    for u, v in edges:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        common &= ~((1 << (v + 1)) - 1)
        while common:
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    return max_triangle_sum

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_triangles_info(n, edges):
    """Compute score and return list of (triangle, sum) sorted by sum descending."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    max_triangle_sum = 0
    triangles = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        common &= ~((1 << (v + 1)) - 1)
        while common:
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            triangles.append(((u, v, w), s))
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    triangles.sort(key=lambda x: x[1], reverse=True)
    return max_triangle_sum, triangles

def _degree_ordered_swap(n, edges_set, all_pairs, target_max_sum):
    """
    Degree-ordered edge replacement heuristic.
    For the worst triangle, try to remove an edge involving a high-degree vertex
    and add a non-edge that doesn't create worse triangles.
    Returns (new_edges_set, new_score) or (None, None) if no improvement found.
    """
    max_score, triangles = _compute_triangles_info(n, list(edges_set))
    if not triangles:
        return None, None
    
    # Focus on the worst triangles
    for tri, tri_sum in triangles[:10]:
        if tri_sum < target_max_sum:
            continue
        
        u, v, w = tri
        # Get degrees
        deg = [0] * n
        for a, b in edges_set:
            deg[a] += 1
            deg[b] += 1
        
        # Sort vertices in triangle by degree descending
        verts = sorted([u, v, w], key=lambda x: deg[x], reverse=True)
        
        # Try removing edges from highest-degree vertex first
        edge_candidates = []
        for i in range(3):
            a, b = verts[i], verts[(i + 1) % 3]
            edge_candidates.append((min(a, b), max(a, b), deg[a] + deg[b]))
        edge_candidates.sort(key=lambda x: x[2], reverse=True)
        
        for e1, _, _ in edge_candidates:
            if e1 not in edges_set:
                continue
            
            # Try adding various non-edges
            # Prioritize non-edges that connect low-degree vertices
            non_edges = [pe for pe in all_pairs if pe not in edges_set]
            if not non_edges:
                break
            
            # Sort non-edges by sum of degrees (lower is better to keep degrees balanced)
            non_edges_sorted = sorted(non_edges, key=lambda pe: deg[pe[0]] + deg[pe[1]])
            
            for e2 in non_edges_sorted[:50]:
                new_edges = set(edges_set)
                new_edges.remove(e1)
                new_edges.add(e2)
                
                new_score = _compute_score(n, list(new_edges))
                if new_score < max_score:
                    return new_edges, new_score
    
    return None, None

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    
    best_score = float('inf')
    best_edges = None
    
    # Multiple restarts with different strategies
    max_restarts = 12
    iterations_per_restart = 25000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Initialize graph
        if restart == 0:
            # K_8,8 + 1 internal edge (baseline)
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
        elif restart == 1:
            # K_8,8 + different internal edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((3, 5))
        elif restart == 2:
            # K_8,8 + 2 internal edges, remove 1 cross edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
            edges_set.add((2, 3))
            edges_set.discard((0, 8))
        elif restart == 3:
            # K_8,8 + 1 edge in other side
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((8, 9))
        else:
            # Random graph with 65 edges
            random.shuffle(all_pairs)
            edges_set = set(all_pairs[:65])
        
        edges_set = set(edges_set)
        current_score = _compute_score(n, list(edges_set))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edges_set)
        
        # Simulated Annealing
        T_start = 2.5
        T_end = 0.005
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            progress = i / iterations_per_restart
            T = T_start * math.pow(T_end / T_start, progress)
            
            # With some probability, use degree-ordered swap
            if random.random() < 0.15 and current_score > 0:
                new_edges, new_score = _degree_ordered_swap(n, edges_set, all_pairs, current_score)
                if new_edges is not None:
                    if new_score <= current_score:
                        current_score = new_score
                        edges_set = new_edges
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = list(edges_set)
                    continue
            
            # Standard SA move: remove one edge, add another
            e1 = random.choice(list(edges_set))
            
            # Pick non-edge
            attempts = 0
            e2 = None
            while attempts < 20:
                candidate = random.choice(all_pairs)
                if candidate not in edges_set:
                    e2 = candidate
                    break
                attempts += 1
            
            if e2 is None:
                continue
            
            new_edges_set = set(edges_set)
            new_edges_set.remove(e1)
            new_edges_set.add(e2)
            new_score = _compute_score(n, list(new_edges_set))
            
            # Acceptance
            if new_score <= current_score:
                current_score = new_score
                edges_set = new_edges_set
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        edges_set = new_edges_set
    
    # Final polish: intensive local search on best found
    if best_edges is not None and time.time() - start_time < DEADLINE - 5:
        edges_set = set(best_edges)
        current_score = best_score
        
        polish_iters = 30000
        T_start = 0.3
        T_end = 0.001
        
        for i in range(polish_iters):
            if time.time() - start_time > DEADLINE - 3:
                break
                
            progress = i / polish_iters
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Try degree-ordered swap first
            if random.random() < 0.25:
                new_edges, new_score = _degree_ordered_swap(n, edges_set, all_pairs, current_score)
                if new_edges is not None:
                    if new_score <= current_score:
                        current_score = new_score
                        edges_set = new_edges
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = list(edges_set)
                    continue
            
            # Standard swap
            e1 = random.choice(list(edges_set))
            
            attempts = 0
            e2 = None
            while attempts < 20:
                candidate = random.choice(all_pairs)
                if candidate not in edges_set:
                    e2 = candidate
                    break
                attempts += 1
            
            if e2 is None:
                continue
            
            new_edges_set = set(edges_set)
            new_edges_set.remove(e1)
            new_edges_set.add(e2)
            new_score = _compute_score(n, list(new_edges_set))
            
            if new_score <= current_score:
                current_score = new_score
                edges_set = new_edges_set
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        edges_set = new_edges_set
    
    # Fallback
    if best_edges is None:
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
