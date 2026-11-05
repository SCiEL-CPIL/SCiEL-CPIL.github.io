import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    m = 401
    
    def near_regular_graph():
        """Initialize as a near-regular graph to flatten degree distribution."""
        target_deg = m * 2 // n  # average degree
        edges = []
        edge_set = set()
        
        # Build a near-regular graph using greedy degree balancing
        deg = [0] * n
        # Start with a cycle
        for i in range(n):
            j = (i + 1) % n
            if i < j:
                edge_set.add((i, j))
                edges.append([i, j])
                deg[i] += 1
                deg[j] += 1
        
        # Add remaining edges greedily, preferring low-degree vertices
        remaining = m - len(edges)
        attempts = 0
        while len(edges) < m and attempts < 100000:
            attempts += 1
            # Pick two vertices with lowest degrees (not already connected)
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                continue
            # Prefer pairs where both have low degree
            edge_set.add((u, v))
            edges.append([u, v])
            deg[u] += 1
            deg[v] += 1
        
        return edges, edge_set, deg

    def build_bitmasks(edge_set):
        """Build bitmask representation of adjacency."""
        adj_masks = [0] * n
        for (u, v) in edge_set:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def calc_book_bitmask(u, v, adj_masks):
        """Calculate book size (common neighbors) using bitmasks."""
        common = adj_masks[u] & adj_masks[v]
        return bin(common).count('1')

    def calc_all_books(edge_set, adj_masks):
        """Calculate all book sizes."""
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = calc_book_bitmask(u, v, adj_masks)
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def find_worst_edge(books):
        """Find the edge with maximum book size."""
        worst_edge = None
        worst_val = -1
        for e, val in books.items():
            if val > worst_val:
                worst_val = val
                worst_edge = e
        return worst_edge, worst_val

    def find_low_triangle_candidates(edge_set, adj_masks, exclude_edge, max_candidates=50):
        """Find candidate edges to add that have low triangle count."""
        candidates = []
        # Sample random non-edges
        for _ in range(200):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set or (u, v) == exclude_edge:
                continue
            tri_count = calc_book_bitmask(u, v, adj_masks)
            candidates.append((tri_count, u, v))
        
        candidates.sort(key=lambda x: x[0])
        return candidates[:max_candidates]

    best_score = 999
    best_edges = None
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
            
        edges, edge_set, deg = near_regular_graph()
        adj_masks = build_bitmasks(edge_set)
        books = calc_all_books(edge_set, adj_masks)
        current_score = get_max_book(books)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [[u, v] for (u, v) in edge_set]
            
        if current_score <= 7:
            break
            
        # Local Search with bitmask acceleration
        stuck = 0
        for step in range(5000):
            if time.time() > deadline - 2:
                return best_edges
                
            worst_edge, worst_val = find_worst_edge(books)
            
            if worst_val <= 7:
                break
                
            # Try to swap worst_edge with a low-triangle edge
            u, v = worst_edge
            swapped = False
            
            # Get candidate edges to add
            candidates = find_low_triangle_candidates(edge_set, adj_masks, (u, v), max_candidates=30)
            
            for tri_count, ru, rv in candidates:
                # Perform swap: remove (u,v), add (ru,rv)
                edge_set.remove((u, v))
                edge_set.add((ru, rv))
                
                # Update adj_masks
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                adj_masks[ru] |= (1 << rv)
                adj_masks[rv] |= (1 << ru)
                
                # Update books for affected edges
                affected_nodes = {u, v, ru, rv}
                new_books = {}
                for e, val in books.items():
                    e_u, e_v = e
                    if e_u in affected_nodes or e_v in affected_nodes:
                        new_books[e] = calc_book_bitmask(e_u, e_v, adj_masks)
                    else:
                        new_books[e] = val
                
                # Add new edge to books
                new_books[(ru, rv)] = calc_book_bitmask(ru, rv, adj_masks)
                
                new_max = get_max_book(new_books)
                
                if new_max < current_score:
                    # Accept
                    current_score = new_max
                    books = new_books
                    # Update edges list
                    edges.remove([u, v])
                    edges.append([ru, rv])
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [[a, b] for (a, b) in edge_set]
                        
                    swapped = True
                    stuck = 0
                    break
                else:
                    # Revert
                    edge_set.remove((ru, rv))
                    edge_set.add((u, v))
                    adj_masks[ru] &= ~(1 << rv)
                    adj_masks[rv] &= ~(1 << ru)
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
            if not swapped:
                stuck += 1
            else:
                stuck = 0
                
            if stuck > 50:
                break
                
    return best_edges
