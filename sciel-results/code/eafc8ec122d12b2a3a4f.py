"""
Combined strategy:
1. Structural initialization: Polarity graph of PG(2,5) (31 vertices, 93 edges, C4-free or near C4-free).
2. Refinement: Remove 2 edges to reach 91, choosing the removal that minimizes C4 count (should be 0 or very low).
3. If the polarity graph isn't C4-free or edge count differs, fallback to local search on the best structured start.
"""
import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 31
    target_edges = 91

    # --- Helper: Count 4-cycles ---
    def count_c4(neighbors):
        """
        Count 4-cycles using neighbor sets.
        Formula: sum_{u < v} C(|N(u) ∩ N(v)|, 2) // 2
        """
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(neighbors[u].intersection(neighbors[v]))
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    # --- Step 1: Generate Polarity Graph of PG(2,5) ---
    def generate_polarity_graph():
        """
        Construct the polarity graph of PG(2, 5).
        Vertices: 31 points in PG(2,5).
        Edges: P ~ Q if P^T A Q = 0 in F_5.
        We use the standard symmetric matrix A = [[1,0,0],[0,1,0],[0,0,1]] (dot product).
        However, the standard "polarity graph" for C4-free constructions often uses a specific form.
        Let's try the standard dot product first. If it's not C4-free or has wrong edge count,
        we might need a different matrix, but let's see.
        
        Actually, the graph defined by x^2+y^2+z^2=0 orthogonality in PG(2,5) is the one we want.
        Since 5 ≡ 1 mod 4, -1 is a square, so the conic is non-degenerate.
        Points on the conic are isotropic.
        In a polarity graph, we usually define adjacency for distinct points P != Q such that P^T A Q = 0.
        We do NOT include loops (P^T A P = 0 doesn't make P adjacent to itself).
        """
        q = 5
        # Generate 31 points in PG(2,5)
        points = []
        # Affine points (x, y, 1)
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        # Points at infinity (1, m, 0)
        for m in range(q):
            points.append((1, m, 0))
        # Point at infinity (0, 1, 0)
        points.append((0, 1, 0))
        
        # Verify uniqueness (they should be distinct projective points)
        # Normalize? No, just check if any are scalar multiples.
        # (x,y,1) and (1,m,0) are distinct. (1,m,0) and (0,1,0) are distinct.
        # Within affine, (x,y,1) are distinct.
        # Within infinity, (1,m,0) are distinct for m=0..4.
        # So all 31 are distinct.
        
        edges = set()
        # A = Identity matrix. P^T A Q = P[0]*Q[0] + P[1]*Q[1] + P[2]*Q[2]
        for i in range(31):
            p1 = points[i]
            for j in range(i + 1, 31):
                p2 = points[j]
                val = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
                if val == 0:
                    edges.add((i, j))
        
        return edges

    # --- Step 2: Initialize ---
    # Try the polarity graph
    pol_edges = generate_polarity_graph()
    current_edges = set(pol_edges)
    
    # Check edge count
    if len(current_edges) != target_edges:
        # If it's 93, remove 2 edges to minimize C4.
        # If it's something else, we might need to adjust.
        # But first, let's see what we have.
        if len(current_edges) > target_edges:
            # Remove edges one by one, choosing the one that minimizes C4 count.
            # Since we start from a C4-free (or low C4) graph, removing edges should keep C4 low.
            # We want to remove 2 edges.
            # Strategy: Try all pairs of edges to remove? 93*92/2 = 4278 pairs.
            # For each pair, compute C4 count. Pick the best.
            
            def build_neighbors(edges):
                neighbors = [set() for _ in range(n)]
                for u, v in edges:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                return neighbors
            
            # Get list of edges
            edge_list = list(current_edges)
            best_edges_to_remove = None
            best_c4 = float('inf')
            
            # If we have exactly 2 extra edges, we can just try all pairs.
            # If we have more, this is a bigger problem. But 93 is expected.
            if len(current_edges) - target_edges == 2:
                for i in range(len(edge_list)):
                    for j in range(i + 1, len(edge_list)):
                        # Remove edge_list[i] and edge_list[j]
                        e1 = edge_list[i]
                        e2 = edge_list[j]
                        temp_edges = set(current_edges)
                        temp_edges.discard(e1)
                        temp_edges.discard(e2)
                        
                        neighbors = build_neighbors(temp_edges)
                        c4 = count_c4(neighbors)
                        
                        if c4 < best_c4:
                            best_c4 = c4
                            best_edges_to_remove = (e1, e2)
                
                if best_edges_to_remove:
                    current_edges.discard(best_edges_to_remove[0])
                    current_edges.discard(best_edges_to_remove[1])
            else:
                # Fallback: Random removal or local search
                # Just remove random edges until we have target_edges
                while len(current_edges) > target_edges:
                    e = random.choice(list(current_edges))
                    current_edges.discard(e)
        else:
            # If we have too few edges, add random ones? 
            # This case is unlikely for PG(2,5) polarity graph.
            # Fallback to random graph if structure fails.
            pass

    # --- Step 3: Verify and Refine with Local Search ---
    # Even if we removed edges optimally from the polarity graph, we might have C4s > 0.
    # We should run a local search to ensure we are at the global minimum (likely 0 or 1).
    
    neighbors = [set() for _ in range(n)]
    for u, v in current_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    current_c4 = count_c4(neighbors)
    best_edges = set(current_edges)
    best_c4 = current_c4
    
    # Local Search: Swap edges to reduce C4 count.
    # Move: Remove (u,v), Add (w,x).
    # We want to try to get to 0 C4s.
    
    improved = True
    while improved and (time.time() - start_time < deadline - 10):
        improved = False
        # Shuffle edges to try different ones
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        # Try a limited number of swaps per pass
        for _ in range(500):
            if time.time() - start_time > deadline - 10:
                break
            if not current_edges:
                break
                
            # Pick a random edge to remove
            u, v = random.choice(edge_list)
            
            # Pick a random non-edge to add
            w = random.randint(0, n - 1)
            x = random.randint(0, n - 1)
            if w == x:
                continue
            if w > x:
                w, x = x, w
            if (w, x) in current_edges:
                continue
            if (w, x) == (u, v):
                continue
            
            # Apply swap
            current_edges.discard((u, v))
            neighbors[u].discard(v)
            neighbors[v].discard(u)
            
            current_edges.add((w, x))
            neighbors[w].add(x)
            neighbors[x].add(w)
            
            new_c4 = count_c4(neighbors)
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                improved = True
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edges = set(current_edges)
            else:
                # Revert
                current_edges.discard((w, x))
                neighbors[w].discard(x)
                neighbors[x].discard(w)
                
                current_edges.add((u, v))
                neighbors[u].add(v)
                neighbors[v].add(u)
                
                # If we found 0 C4s, we can stop early?
                # But maybe there's a better structure? 0 is the minimum possible.
                if current_c4 == 0:
                    # We can break out of the loop if we are confident 0 is optimal.
                    # Since ex(31, C4) = 90, a graph with 91 edges MUST have at least 1 C4?
                    # Wait. ex(n, H) is the max number of edges in an H-free graph.
                    # If ex(31, C4) = 90, then any graph with 91 edges has at least one C4.
                    # So 0 C4s is IMPOSSIBLE.
                    # The minimum must be >= 1.
                    # If we find 1, it is likely optimal.
                    if current_c4 == 1:
                        # We can break, as 1 is the theoretical lower bound.
                        break
        
        if current_c4 == 1:
            break

    # Return the best edges found
    result = [[u, v] for u, v in best_edges]
    return result

# Fallback in case of any issues
try:
    def construct():
        start_time = time.time()
        deadline = 120.0
        n = 31
        target_edges = 91
        
        def count_c4(neighbors):
            total = 0
            for u in range(n):
                for v in range(u + 1, n):
                    common = len(neighbors[u].intersection(neighbors[v]))
                    if common >= 2:
                        total += common * (common - 1) // 2
            return total // 2
        
        def generate_random_graph(n, m):
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_pairs)
            edges = set(all_pairs[:m])
            return edges
        
        best_edges = None
        best_c4 = float('inf')
        
        # Fallback: Random starts + local search
        num_starts = 20
        for start_idx in range(num_starts):
            if time.time() - start_time > deadline - 20:
                break
                
            edges = generate_random_graph(n, target_edges)
            neighbors = [set() for _ in range(n)]
            for u, v in edges:
                neighbors[u].add(v)
                neighbors[v].add(u)
            
            c4_count = count_c4(neighbors)
            
            if c4_count < best_c4:
                best_c4 = c4_count
                best_edges = set(edges)
            
            # Local search
            improved = True
            while improved and (time.time() - start_time < deadline - 5):
                improved = False
                edge_list = list(edges)
                random.shuffle(edge_list)
                
                max_attempts = 200
                for _ in range(max_attempts):
                    if time.time() - start_time > deadline - 5:
                        break
                    u, v = edge_list[random.randint(0, len(edge_list) - 1)]
                    
                    w = random.randint(0, n - 1)
                    x = random.randint(0, n - 1)
                    if w == x:
                        continue
                    if w > x:
                        w, x = x, w
                    if (w, x) in edges:
                        continue
                    if (w, x) == (u, v):
                        continue
                    
                    # Swap
                    edges.discard((u, v))
                    neighbors[u].discard(v)
                    neighbors[v].discard(u)
                    
                    edges.add((w, x))
                    neighbors[w].add(x)
                    neighbors[x].add(w)
                    
                    new_c4 = count_c4(neighbors)
                    
                    if new_c4 < c4_count:
                        c4_count = new_c4
                        improved = True
                        if c4_count < best_c4:
                            best_c4 = c4_count
                            best_edges = set(edges)
                    else:
                        # Revert
                        edges.discard((w, x))
                        neighbors[w].discard(x)
                        neighbors[x].discard(w)
                        
                        edges.add((u, v))
                        neighbors[u].add(v)
                        neighbors[v].add(u)
        
        if best_edges is None:
            # Fallback
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best_edges = set(all_pairs[:target_edges])
        
        return [[u, v] for u, v in best_edges]

except Exception as e:
    def construct():
        n = 31
        target_edges = 91
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return [list(p) for p in all_pairs[:target_edges]]
