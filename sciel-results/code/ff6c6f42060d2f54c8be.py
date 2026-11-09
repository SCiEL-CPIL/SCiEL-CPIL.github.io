import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_refinement(edges, deadline):
        # Initial decomposition: each edge is a path
        paths = [list(e) for e in edges]
        
        # Build adjacency for quick edge lookup
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        def get_path_set(p):
            return set(p)
            
        def get_path_edge_set(p):
            return {(min(p[i], p[i+1]), max(p[i], p[i+1])) for i in range(len(p)-1)}

        # We will iteratively try to merge paths or reorder them to reduce count.
        # Since we start with all single edges, the number of paths is |E|.
        # Merging two paths P1 and P2 into one is only possible if they share a vertex?
        # No, path decomposition requires edge-disjoint paths.
        # If we merge two paths, we are effectively finding a longer path that covers
        # the edges of both. This is only possible if the union of edges in P1 and P2
        # contains a simple path that uses all edges of P1 and P2.
        # This is equivalent to checking if the subgraph induced by P1+P2 edges has a Hamiltonian path
        # on the vertices involved, respecting the edge set.
        
        # A simpler local search:
        # 1. Try to extend a path by absorbing a neighbor path if possible.
        # 2. Try to "swap" edges between paths to enable merging.
        
        # Let's use a different strategy:
        # Start with the greedy Warnsdorff result as a good initial guess,
        # then try to improve it by local moves.
        
        # First, get a decent initial decomposition using a fast heuristic
        remaining = set(edges)
        adj_rem = defaultdict(set)
        for u, v in edges:
            adj_rem[u].add(v)
            adj_rem[v].add(u)
            
        init_paths = []
        
        def warnsdorff_step(u, path_verts, used_e):
            cands = []
            for v in adj_rem[u]:
                e = (min(u, v), max(u, v))
                if e in used_e or v in path_verts:
                    continue
                # Flexibility: degree in remaining graph excluding current path
                flex = 0
                for w in adj_rem[v]:
                    if w == u: continue
                    e2 = (min(v, w), max(v, w))
                    if e2 not in used_e and w not in path_verts:
                        flex += 1
                cands.append((flex, v))
            if not cands: return None
            cands.sort(key=lambda x: (x[0], x[1]))
            return cands[0][1]

        while remaining and time.time() < deadline - 5:
            deg_rem = defaultdict(int)
            for u, v in remaining:
                deg_rem[u] += 1
                deg_rem[v] += 1
            if not deg_rem: break
            start = max(deg_rem.keys(), key=lambda x: (deg_rem[x], -x))
            
            path = [start]
            pverts = {start}
            pedges = set()
            
            while True:
                nxt = warnsdorff_step(path[-1], pverts, pedges)
                if nxt is None: break
                path.append(nxt)
                pverts.add(nxt)
                e = (min(path[-2], nxt), max(path[-2], nxt))
                pedges.add(e)
                
            if len(path) > 1:
                init_paths.append(path)
                for e in pedges:
                    remaining.discard(e)
                    
        # If Warnsdorff failed to cover everything (shouldn't happen for connected graphs 
        # if implemented correctly, but just in case), add remaining edges as singletons
        for e in remaining:
            init_paths.append(list(e))
            
        paths = init_paths
        
        # Local Search Phase
        # We want to minimize len(paths).
        # Move 1: Merge two paths if their union of edges forms a single simple path.
        #   This happens if the two paths share exactly one vertex, and that vertex is an endpoint of both.
        #   Or more generally, if we can reorder/flip them.
        # Move 2: "2-opt" on a path: reverse a segment to allow merging with another path?
        
        # Let's focus on Move 1 first: Merging.
        # Two paths P=[u1..uk] and Q=[v1..vm] can be merged into a single path
        # if the graph formed by their edges is a path.
        # This implies the union of edges must form a simple path.
        # Condition: 
        #   1. The set of vertices V_P ∪ V_Q must have exactly 2 vertices of degree 1 in the union subgraph, and all others degree 2.
        #   2. The union must be connected (it is, if they share a vertex or are connected via edge? No, they are disjoint edge sets. They must share a vertex to be connected in the vertex sense for a path decomposition? 
        #      Wait, path decomposition paths are edge-disjoint. They CAN share vertices.
        #      If they don't share a vertex, their union is two disjoint paths, which cannot be a single path.
        #      So they MUST share at least one vertex.
        
        # If they share a vertex x:
        #   In P, x has degree 1 (if endpoint) or 2 (if internal).
        #   In Q, x has degree 1 (if endpoint) or 2 (if internal).
        #   In the union, degree of x is deg_P(x) + deg_Q(x).
        #   For the union to be a single path, x must have degree 2 in the union (if it's an internal node of the new path) or degree 1 (if it's an endpoint, which implies one path is length 0? No, paths have edges).
        #   Actually, if we merge P and Q into one path, the shared vertex x must be an endpoint of P and an endpoint of Q.
        #   Why? If x is internal in P, it has 2 neighbors in P. If we attach Q at x, x would have 3 neighbors in the union (2 from P, 1 from Q), which is a branch, not a path.
        #   Unless we "cut" P at x? But we are merging whole paths.
        #   So, a simple merge is only possible if x is an endpoint of P and an endpoint of Q.
        
        # What if they don't share a vertex? They can't be merged into a single simple path unless there is an edge connecting them, but they are edge-disjoint and cover all edges. If there's an edge between them, it's already in one of the paths or a third path.
        
        # So, the primary local move is:
        # Check all pairs of paths. If they share an endpoint, try to merge them.
        # After merging, the new path is P + Q (reversed if necessary).
        
        # Also, we can try to "extend" a path by taking an edge from another path?
        # That's more complex.
        
        # Let's implement the endpoint merging heuristic.
        # To speed it up, index paths by their endpoints.
        
        def try_merge(paths):
            # Returns True if a merge was performed
            # Group paths by endpoints
            # Since paths are lists, endpoints are paths[0] and paths[-1]
            
            # We'll just iterate pairs. N_paths is small (<= 38).
            # O(N^2) checks.
            merged = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]: continue
                p1 = paths[i]
                if len(p1) < 2: # Single vertex? Shouldn't happen in edge decomposition
                    new_paths.append(p1)
                    continue
                    
                ep1_start = p1[0]
                ep1_end = p1[-1]
                
                for j in range(i+1, len(paths)):
                    if used[j]: continue
                    p2 = paths[j]
                    if len(p2) < 2:
                        continue
                        
                    ep2_start = p2[0]
                    ep2_end = p2[-1]
                    
                    # Check if they share an endpoint
                    # Case 1: p1[0] == p2[0]
                    #   New path: p1[1:] + [p2[0]] + reversed(p2[1:])? No.
                    #   p1 = [A, ..., B]. p2 = [A, ..., C].
                    #   Merge at A: [B, ..., A, ..., C] -> reverse p1 to [B..A] then append p2[1:]
                    #   Result: [B, ..., A, v2_1, ..., C]
                    #   This is a valid simple path if p1 and p2 only share A.
                    #   We must ensure they don't share other vertices.
                    
                    # Case 2: p1[0] == p2[-1]
                    #   p1 = [A..B], p2 = [C..A]
                    #   Merge at A: [C..A..B] -> p2 + p1[1:]
                    
                    # Case 3: p1[-1] == p2[0]
                    #   p1 = [A..B], p2 = [B..C]
                    #   Merge at B: [A..B..C] -> p1 + p2[1:]
                    
                    # Case 4: p1[-1] == p2[-1]
                    #   p1 = [A..B], p2 = [C..B]
                    #   Merge at B: [C..B..A] -> p2 + reversed(p1[1:])
                    
                    shared = None
                    merge_type = None
                    
                    if ep1_start == ep2_start:
                        shared = ep1_start
                        merge_type = 1
                    elif ep1_start == ep2_end:
                        shared = ep1_start
                        merge_type = 2
                    elif ep1_end == ep2_start:
                        shared = ep1_end
                        merge_type = 3
                    elif ep1_end == ep2_end:
                        shared = ep1_end
                        merge_type = 4
                        
                    if shared is not None:
                        # Check for other shared vertices
                        set1 = set(p1)
                        set2 = set(p2)
                        common = set1 & set2
                        if common == {shared}:
                            # Valid merge
                            if merge_type == 1:
                                # p1=[S..], p2=[S..] -> reverse p1 + p2[1:]
                                new_p = p1[::-1] + p2[1:]
                            elif merge_type == 2:
                                # p1=[S..], p2=[..S] -> p2 + p1[1:]
                                new_p = p2 + p1[1:]
                            elif merge_type == 3:
                                # p1=[..E], p2=[E..] -> p1 + p2[1:]
                                new_p = p1 + p2[1:]
                            elif merge_type == 4:
                                # p1=[..E], p2=[..E] -> p2 + p1[::-1][1:]
                                new_p = p2 + p1[::-1][1:]
                                
                            # Verify new path is simple (should be by construction if sets disjoint except shared)
                            if len(new_p) == len(set(new_p)):
                                new_paths.append(new_p)
                                used[i] = True
                                used[j] = True
                                merged = True
                                break # Restart loop with new list? Or just continue?
                                # Breaking inner loop, but we modified 'used'. 
                                # It's safer to rebuild the list from scratch after a merge to avoid index issues.
                                # Let's just return True and let the caller handle it.
                                return True, new_paths, i, j
                                
            if not merged:
                return False, paths, -1, -1
            return False, paths, -1, -1

        # Iterative improvement
        # We also want to try to "re-route" paths to create endpoints that allow merging.
        # For now, just try merging repeatedly.
        
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            
            success, paths, i, j = try_merge(paths)
            if not success:
                # Try a random "swap" to break local optimum?
                # Or just stop if no merges found.
                # Let's try a simple heuristic: pick two paths, and try to see if reversing one helps.
                # Reversing doesn't change endpoints set, just swaps start/end.
                # So it doesn't help with the "share endpoint" condition unless we consider the graph structure.
                
                # If no merges, we are stuck.
                # Let's try to improve the initial decomposition by restarting Warnsdorff with different start nodes?
                # That's a different strategy.
                
                # For now, if no merge, break.
                break
                
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        
        best_paths = None
        best_count = float('inf')
        
        # Run refinement from multiple random initializations?
        # Or just run the deterministic Warnsdorff + Merge.
        
        # Let's run the decomposition a few times with slight randomization in Warnsdorff tie-breaking
        for attempt in range(10):
            if time.time() > DEADLINE - 5:
                break
            
            # Randomize the "flexibility" tie-breaking in Warnsdorff
            # We can do this by adding a small random noise to the flexibility score
            # Or just shuffle the adjacency list iteration order.
            
            # To keep it simple, we'll just run the same algorithm.
            # If we want diversity, we can randomize the start node selection.
            
            # Let's modify decompose_refinement to accept a seed for random start selection
            # But for now, let's just call it.
            
            paths = decompose_refinement(edges, DEADLINE - 5)
            count = len(paths)
            
            if count < best_count:
                best_count = count
                best_paths = paths
                
            if best_count <= 10:
                break
                
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        result.append(best_paths)
        
    return result

# EVOLVE-BLOCK-END
