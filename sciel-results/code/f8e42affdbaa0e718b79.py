import random
import time
import sys

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Precompute all triangles as bitmasks
    # A triangle is a set of 3 vertices {a,b,c} with edges (a,b),(b,c),(a,c)
    # We need to ensure no K4, i.e., no 4 vertices all pairwise connected.
    
    # Strategy: Use bitmask representation of the graph.
    # graph_mask is a list of N integers, where graph_mask[u] has bit v set if (u,v) is an edge.
    
    def is_k4_free(graph_mask):
        # For each pair of adjacent vertices (u,v), check if they share two common neighbors
        # that are also adjacent to each other (forming K4 with u,v).
        # Actually, K4 exists if there's a triangle {a,b,c} and a vertex d connected to all three.
        # More efficiently: for each edge (u,v), let common = graph_mask[u] & graph_mask[v].
        # If common contains an edge (w,x) where w,x are both in common, then {u,v,w,x} is K4.
        # Wait, that's not quite right. If w and x are both neighbors of u and v, and w-x is an edge,
        # then {u,v,w,x} has all 6 edges? No. u-w, u-x, v-w, v-x, w-x. We need u-v too (yes, it's an edge).
        # So yes, if there's an edge within the common neighborhood of (u,v), we have a K4.
        for u in range(N):
            for v in range(u+1, N):
                if (graph_mask[u] >> v) & 1:
                    common = graph_mask[u] & graph_mask[v]
                    # Check if common induces any edge
                    # For each w in common, check if graph_mask[w] & common != 0 (excluding w itself, but w isn't in common's bit representation of w)
                    # Actually, we just need to check if there's any edge within 'common'.
                    # Iterate through bits of common
                    c = common
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other vertex in common
                        if (graph_mask[w] & common) & (common ^ lsb):
                            return False
        return True

    def max_triangle_free_size(graph_mask):
        # Find the largest subset of vertices that induces a triangle-free graph.
        # Use branch and bound / backtracking.
        # A set S is triangle-free if for every vertex v in S, the neighbors of v within S form an independent set.
        # Equivalently, S contains no triangle.
        
        # We'll use a recursive search with bitmasks.
        # State: (available_vertices, current_set)
        # available_vertices: bitmask of vertices we can still add
        # current_set: bitmask of vertices already in the triangle-free set
        
        # To maintain triangle-freeness: if we add vertex v to current_set,
        # v's neighbors within current_set must form an independent set.
        # But it's easier to think: the new set must not contain any triangle.
        
        # Better approach: branch and bound on vertices.
        # Order vertices. For each vertex, decide include or exclude.
        # Prune: if current_set + all remaining can't beat best, prune.
        # Triangle-free check: maintain that current_set is triangle-free.
        
        # Even better: use the property that a set is triangle-free iff it's a union of cliques of size <= 2? No.
        # Triangle-free means no 3-clique.
        
        # Let's use a recursive function:
        # def bt(remaining, current):
        #   if not remaining: update best; return
        #   # Bound: len(current) + popcount(remaining) <= best -> prune
        #   # Try adding a vertex from remaining
        #   # For each vertex v in remaining:
        #   #   if current + {v} is triangle-free:
        #   #     new_remaining = remaining - {v} - (neighbors of v that would create a triangle with current)
        #   #     Actually, if we add v, we must remove from remaining any vertex w such that {v, w} plus some u in current forms a triangle.
        #   #     i.e., if u is in current and u is neighbor of v and u is neighbor of w, then w can't be added.
        #   #     So forbidden = union over u in current: (graph_mask[v] & graph_mask[u])
        #   #     new_remaining = remaining & ~forbidden & ~(1<<v)
        #   #     bt(new_remaining, current | (1<<v))
        #   #   Also, try skipping v:
        #   #     bt(remaining & ~(1<<v), current)
        
        # This could be slow. Let's optimize with memoization or better ordering.
        
        # Alternative: iterate over all subsets? 2^18 = 262144. That's feasible!
        # For each subset, check if it's triangle-free.
        # But we want the MAX size. We can iterate by size descending.
        
        # Actually, let's just check all subsets of size >= current_best and stop when we find one.
        # Or, iterate all subsets, track max size that is triangle-free.
        
        # 2^18 = 262144 subsets. For each, checking triangle-freeness:
        # A subset S is triangle-free iff for every edge (u,v) in S, there's no w in S adjacent to both u and v.
        # Or: for every triple (u,v,w) in S, not all three edges exist.
        
        # Let's precompute for each pair (u,v) the bitmask of common neighbors.
        # common_neighbors[u][v] = graph_mask[u] & graph_mask[v]
        
        # Then S is triangle-free iff for every edge (u,v) in S, common_neighbors[u][v] & S == 0 (excluding u and v, but u,v aren't in common_neighbors anyway since no self-loops).
        # Wait, common_neighbors[u][v] includes vertices w such that w-u and w-v are edges.
        # If w is in S, then {u,v,w} is a triangle.
        
        # So: S is triangle-free iff for every (u,v) in S with u<v and (u,v) is an edge, (common_neighbors[u][v] & S) == 0.
        
        # Let's precompute common_neighbors as a 2D list.
        common_neighbors = [[0]*N for _ in range(N)]
        for u in range(N):
            for v in range(u+1, N):
                if (graph_mask[u] >> v) & 1:
                    common_neighbors[u][v] = graph_mask[u] & graph_mask[v]
                    common_neighbors[v][u] = common_neighbors[u][v]
                else:
                    common_neighbors[u][v] = 0
                    common_neighbors[v][u] = 0
        
        # Now, for a subset S (as bitmask), check triangle-free:
        # For each u in S, for each v in S with v>u:
        #   if (u,v) is an edge, check if common_neighbors[u][v] & S has any bit set.
        
        # To speed up, we can precompute for each S... no, that's 2^18.
        
        # Let's just iterate all subsets and check.
        # Actually, we can do better: iterate subsets by size descending, stop at first valid one.
        
        # But 2^18 iterations with a check that takes up to O(N^2) each might be slow in Python.
        # 262144 * 18*18/2 ≈ 42 million operations. Might be too slow.
        
        # Let's use the recursive approach with pruning.
        
        best = 0
        best_set = 0
        
        # Precompute popcount for all masks? Or use bit_count() (Python 3.10+)
        # Let's use a helper.
        
        def popcount(x):
            return bin(x).count('1')
        
        # Order vertices by degree descending for better pruning
        degrees = [bin(graph_mask[u]).count('1') for u in range(N)]
        order = sorted(range(N), key=lambda u: -degrees[u])
        
        # Use the recursive approach
        # remaining: bitmask of vertices still available
        # current: bitmask of vertices in current triangle-free set
        
        def bt(remaining, current):
            nonlocal best, best_set
            if not remaining:
                if popcount(current) > best:
                    best = popcount(current)
                    best_set = current
                return
            
            # Bound
            if popcount(current) + popcount(remaining) <= best:
                return
            
            # Try adding vertices from remaining
            # Choose the first vertex in order that is in remaining
            r = remaining
            while r:
                lsb = r & (-r)
                v = lsb.bit_length() - 1
                r ^= lsb
                
                # Check if current + {v} is triangle-free
                # This requires that for every u in current, (u,v) is not an edge OR if it is, 
                # then no w in current is adjacent to both u and v.
                # Actually, the condition is: for every u in current where (u,v) is an edge,
                # common_neighbors[u][v] & current == 0.
                
                valid = True
                c = current
                while c:
                    lsb2 = c & (-c)
                    u = lsb2.bit_length() - 1
                    c ^= lsb2
                    if (graph_mask[u] >> v) & 1:
                        if common_neighbors[u][v] & current:
                            valid = False
                            break
                
                if valid:
                    # Compute forbidden vertices: those that would form a triangle with v and some u in current
                    forbidden = 0
                    c = current
                    while c:
                        lsb2 = c & (-c)
                        u = lsb2.bit_length() - 1
                        c ^= lsb2
                        if (graph_mask[u] >> v) & 1:
                            forbidden |= common_neighbors[u][v]
                    
                    new_remaining = remaining & ~forbidden & ~(1 << v)
                    bt(new_remaining, current | (1 << v))
                
                # Try skipping v
                bt(remaining & ~(1 << v), current)
                return  # Only branch on the first vertex to keep it a tree
        
        # Actually, the above branching strategy (always branch on the first available vertex) 
        # gives a binary tree of depth N, which is 2^N = 262144 leaves. That's fine.
        
        full_mask = (1 << N) - 1
        bt(full_mask, 0)
        return best
    
    # Initial graph: start with a random K4-free graph
    # Or start with a complete tripartite and perturb
    
    def random_k4_free_graph():
        # Start with empty graph, add random edges, ensure no K4
        graph_mask = [0] * N
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < 0.5:
                    # Check if adding (u,v) creates a K4
                    # A K4 would be formed if there exist w,x such that all of u,v,w,x are mutually connected
                    # After adding (u,v), check if there's a triangle in the common neighborhood of u and v
                    # that already exists, OR if u and v share two neighbors w,x with w-x edge.
                    common = graph_mask[u] & graph_mask[v]
                    # Check if common contains an edge
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if not has_edge:
                        graph_mask[u] |= (1 << v)
                        graph_mask[v] |= (1 << u)
                        edges.append([u, v])
        return edges, graph_mask
    
    # Local search
    def local_search(edges, graph_mask):
        # Try to improve by adding/removing edges
        # Objective: minimize max_triangle_free_size
        # Constraint: K4-free
        
        current_score = max_triangle_free_size(graph_mask)
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding an edge
            for u in range(N):
                for v in range(u+1, N):
                    if (graph_mask[u] >> v) & 1:
                        continue
                    # Check K4-free
                    common = graph_mask[u] & graph_mask[v]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (graph_mask[w] & common) & (common ^ lsb):
                            has_edge = True
                            break
                    if has_edge:
                        continue
                    
                    # Add edge
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    edges.append([u, v])
                    
                    new_score = max_triangle_free_size(graph_mask)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        graph_mask[u] &= ~(1 << v)
                        graph_mask[v] &= ~(1 << u)
                        edges.remove([u, v])
            
            # Try removing an edge
            for i in range(len(edges)-1, -1, -1):
                u, v = edges[i]
                # Remove edge
                graph_mask[u] &= ~(1 << v)
                graph_mask[v] &= ~(1 << u)
                edges.pop(i)
                
                new_score = max_triangle_free_size(graph_mask)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    graph_mask[u] |= (1 << v)
                    graph_mask[v] |= (1 << u)
                    edges.insert(i, [u, v])
            
            if not improved:
                break
        
        return edges, current_score
    
    # Start with a few random restarts
    best_edges = None
    best_score = N + 1
    
    # Start with complete tripartite
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    tripartite_edges = []
    tripartite_mask = [0] * N
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    tripartite_edges.append([a, b])
                    tripartite_mask[a] |= (1 << b)
                    tripartite_mask[b] |= (1 << a)
    
    tripartite_score = max_triangle_free_size(tripartite_mask)
    if tripartite_score < best_score:
        best_score = tripartite_score
        best_edges = tripartite_edges[:]
    
    # Local search from tripartite
    ls_edges, ls_score = local_search(tripartite_edges[:], tripartite_mask[:])
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        edges, mask = random_k4_free_graph()
        score = max_triangle_free_size(mask)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
        
        # Local search
        ls_edges, ls_score = local_search(edges[:], mask[:])
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges[:]
        
        if time.time() > deadline - 20:
            break
    
    return best_edges if best_edges else []

# Ensure the function is called correctly
if __name__ == "__main__":
    result = construct()
    print(f"Found graph with {len(result)} edges")
