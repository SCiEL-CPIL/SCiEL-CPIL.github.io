import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing 4-cycles.
    Strategy:
    1. Construct a C4-free core with 90 edges using the polarity graph of PG(2,5).
    2. Select the best single edge to add to minimize new C4s.
    3. Use local search (edge swaps) to minimize the total number of C4s.
    """
    n = 31
    q = 5
    m = 91
    
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin under 120s limit

    # --- Step 1: Build C4-free core with 90 edges using PG(2,5) polarity ---
    # Points of PG(2,5): normalized triples (x,y,z) with first nonzero = 1
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    # Inverse of x in GF(5): x^3 mod 5 (since x^4=1)
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: B(p,q) = x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    # This graph is known to be C4-free and has 90 edges (regular of degree 6? No, degree varies? 
    # Actually, polarity graphs of PG(2,q) for odd q are regular? 
    # For q=5, number of points = 31. 
    # Let's just build it and check edge count.
    
    adj = [set() for _ in range(n)]
    edges_core = []
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 - z1*z2) % 5 == 0:
                edges_core.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    if len(edges_core) != 90:
        # Fallback: Greedy C4-free construction
        adj = [set() for _ in range(n)]
        edges_core = []
        random.seed(42)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges_core) >= 90:
                break
            # Check if adding (u,v) creates a C4
            # A C4 is created if there exists a common neighbor w such that... wait.
            # Adding (u,v) creates a C4 if there is a path u-w-v of length 2? No, that's a triangle.
            # Adding (u,v) creates a C4 if there are vertices a,b such that u-a-b-v-u is a cycle?
            # No, the new edge is (u,v). A C4 containing (u,v) must be u-v-a-b-u.
            # So we need a neighbor 'a' of v and a neighbor 'b' of u such that a and b are adjacent.
            # Wait, if the graph was C4-free, adding (u,v) creates a C4 iff there is a common neighbor? No.
            # If u and v have a common neighbor w, then u-w-v is a path. Adding (u,v) makes a triangle u-v-w-u.
            # A C4 is u-v-a-b-u. 
            # So we need a in N(v) and b in N(u) such that a-b is an edge.
            creates_c4 = False
            for a in adj[v]:
                for b in adj[u]:
                    if b in adj[a]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            if not creates_c4:
                edges_core.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
        
        if len(edges_core) < 90:
            # Fill remaining arbitrarily (might create C4s, but we need 90 for core logic)
            for u, v in all_pairs:
                if len(edges_core) >= 90:
                    break
                if v not in adj[u]:
                    edges_core.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)

    # --- Step 2: Find best single edge to add to minimize C4s ---
    # Adding edge (u,v) creates C4s = number of pairs (a,b) such that a in N(u), b in N(v), and a-b is an edge.
    # Note: a and b can be the same vertex? No, distinct vertices in cycle.
    # If a=b, then u-a-v-u is a triangle, not a C4. So a != b.
    
    def count_added_c4s(u, v, adj):
        count = 0
        for a in adj[u]:
            for b in adj[v]:
                if a != b and b in adj[a]:
                    count += 1
        return count

    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, n):
            if v in adj[u]:
                continue
            add = count_added_c4s(u, v, adj)
            if add < best_add:
                best_add = add
                best_edge = [u, v]
    
    if best_edge is None:
        # Find any non-edge
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break

    # Apply the best edge
    final_edges = edges_core + [best_edge]
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # --- Step 3: Local search - try edge swaps to reduce C4 count ---
    
    def count_c4_fast(adj):
        """Count total C4s using common neighbors."""
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                # Only count if u and v are NOT adjacent? 
                # Formula: sum_{u<v} C(common(u,v), 2) counts each C4 twice (once for each pair of opposite vertices).
                # Wait.
                # A C4 has 2 pairs of opposite vertices.
                # If u and v are opposite, they are not adjacent.
                # If u and v are adjacent, they are part of a triangle or something else?
                # The formula sum_{u<v} C(common(u,v), 2) actually counts:
                # For any pair u,v, common neighbors are w such that u-w and v-w exist.
                # Choosing 2 common neighbors w1, w2 gives a cycle u-w1-v-w2-u.
                # This is a C4 if u and v are not adjacent AND w1 and w2 are not adjacent?
                # No, the cycle is u-w1-v-w2-u.
                # If u and v are adjacent, this is not a simple C4 (it's a triangle with a chord or something).
                # Actually, if u and v are adjacent, the cycle u-w1-v-w2-u has length 4, but u-v is a chord.
                # The definition of a C4 subgraph requires the 4 edges to form the cycle.
                # So we must sum only over non-adjacent pairs u,v.
                if v not in adj[u]:
                    cn = len(adj[u] & adj[v])
                    total += cn * (cn - 1) // 2
        return total

    current_c4s = count_c4_fast(adj)
    
    import random
    random.seed(123)
    
    edges_set = set(tuple(sorted(e)) for e in final_edges)
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Helper to update non_edges
    def get_non_edges(edges_set):
        return [(i, j) for i, j in all_pairs if (i, j) not in edges_set]

    non_edges = get_non_edges(edges_set)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try random swaps
        # To speed up, we can precompute some things, but n=31 is small.
        # We'll do a limited number of random attempts per iteration.
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to remove
            e1 = random.choice(list(edges_set))
            u1, v1 = e1
            
            # Pick a random non-edge to add
            if not non_edges:
                break
            e2 = random.choice(non_edges)
            u2, v2 = e2
            
            # Simulate swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            new_c4s = count_c4_fast(adj)
            
            # Revert
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            adj[u1].add(v1)
            adj[v1].add(u1)
            
            delta = new_c4s - current_c4s
            if delta < 0:
                # Apply swap
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                non_edges = get_non_edges(edges_set)
                current_c4s = new_c4s
                improved = True
                break # Restart loop with new state

    # --- Step 4: Final greedy improvement - try all single swaps ---
    # This is O(E * NonE) * Cost(CountC4).
    # E=91, NonE=470. 91*470 = 42k swaps.
    # CountC4 is O(n^3) roughly? No, O(n^2 * avg_deg).
    # 31^2 * 6 ~ 5000 ops.
    # 42k * 5000 = 210M ops. Might be too slow in Python.
    # Let's limit this to a few seconds or just do a few passes.
    
    if time.time() < deadline - 5:
        edges_list = list(edges_set)
        non_edges_list = get_non_edges(edges_set)
        
        best_improvement = 0
        best_swap = None
        
        # Optimize: Only check swaps where u1,v1 are in a C4? 
        # Or just sample?
        # Let's try a limited exhaustive search.
        for e1 in edges_list:
            if time.time() > deadline - 5:
                break
            u1, v1 = e1
            for e2 in non_edges_list:
                u2, v2 = e2
                
                # Quick heuristic check?
                # Just simulate
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                new_c4s = count_c4_fast(adj)
                
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                adj[u1].add(v1)
                adj[v1].add(u1)
                
                delta = current_c4s - new_c4s
                if delta > best_improvement:
                    best_improvement = delta
                    best_swap = (e1, e2)
                    
        if best_swap:
            e1, e2 = best_swap
            edges_set.discard(e1)
            edges_set.add(e2)
            
            u1, v1 = e1
            u2, v2 = e2
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            current_c4s -= best_improvement

    # Convert back to list of lists
    final_edges = [list(e) for e in edges_set]
    
    # Ensure exactly m edges
    if len(final_edges) < m:
        # Should not happen, but safety
        existing = set(tuple(sorted(e)) for e in final_edges)
        for i, j in all_pairs:
            if (i, j) not in existing:
                final_edges.append([i, j])
                existing.add((i, j))
                if len(final_edges) == m:
                    break
    
    return final_edges[:m]

if __name__ == "__main__":
    start = time.time()
    e = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(e)}")
    # Count C4s for verification
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in e:
        adj[u, v] = 1
        adj[v, u] = 1
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                c = np.sum(adj[u, :] * adj[v, :])
                c4_count += c * (c - 1) // 2
    print(f"C4 Count: {c4_count}")
