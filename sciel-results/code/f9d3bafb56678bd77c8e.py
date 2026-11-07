"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    all_vertices = set(range(n))
    m = len(edges)
    
    # Greedy approach: repeatedly find a large biclique from remaining edges
    # A biclique is defined by (L, R) where all cross pairs are edges
    # We want to partition edges, so we pick bicliques whose cross-edges are disjoint
    
    def get_max_biclique(remaining_edges, adj):
        """Find a large biclique using a greedy heuristic.
        Try to find a pair (u, v) with many common neighbors, then expand."""
        if not remaining_edges:
            return None, None, 0
        
        # Strategy: pick a vertex with high degree in remaining graph,
        # then find its common neighbors with another vertex
        best_biclique = None
        best_size = 0
        
        # Get vertices that appear in remaining edges
        rem_vertices = set()
        for e in remaining_edges:
            rem_vertices.add(e[0])
            rem_vertices.add(e[1])
        
        if not rem_vertices:
            return None, None, 0
            
        # Try several seed pairs
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Sort vertices by degree in remaining graph
        deg = [len(rem_adj[i]) for i in rem_vertices]
        sorted_verts = sorted(rem_vertices, key=lambda x: -deg[x])
        
        # Try top vertices as seeds
        tried = set()
        for u in sorted_verts[:20]:
            if len(tried) > 50:
                break
            # Common neighbors of u in remaining graph
            cn_u = rem_adj[u]
            # For each common neighbor candidate, compute common neighbors
            for v in list(cn_u)[:30]:
                if (u, v) in tried or (v, u) in tried:
                    continue
                tried.add((u, v))
                # Common neighbors of both u and v
                cn = rem_adj[u] & rem_adj[v]
                # The biclique would be {u,v} on one side and cn on the other?
                # No - a biclique (L,R) requires all cross edges.
                # If we put u in L and cn in R, we need all u-c edges to be edges.
                # cn are common neighbors, so u-c and v-c are all edges.
                # So (L={u,v}, R=cn) is a valid biclique if all cross edges exist.
                # Cross edges: u-c for c in cn (yes, c in cn_u), v-c for c in cn (yes, c in rem_adj[v])
                # But we also need to check that these edges are in remaining_edges.
                
                # Actually, let's use a simpler approach:
                # Pick L = {u}, R = rem_adj[u] ∩ some set
                # Or pick a small L and find R as intersection of neighbors
                
                # Let's try: L = {u, v}, R = cn
                # Verify all cross edges are in remaining_edges
                valid = True
                for c in cn:
                    if (min(u, c), max(u, c)) not in remaining_edges:
                        valid = False
                        break
                    if (min(v, c), max(v, c)) not in remaining_edges:
                        valid = False
                        break
                if valid and cn:
                    size = len(cn) * 2  # cross edges count
                    if size > best_size:
                        best_size = size
                        best_biclique = ({u, v}, set(cn))
        
        # Also try single-vertex left side
        for u in sorted_verts[:20]:
            r_set = set(rem_adj[u])
            # We can use all neighbors of u
            if len(r_set) > best_size // 2:
                # Verify all edges exist (they do by construction)
                size = len(r_set)
                if size > best_size:
                    best_size = size
                    best_biclique = ({u}, r_set)
        
        if best_biclique is None:
            # Fallback: single edge
            e = next(iter(remaining_edges))
            return {e[0]}, {e[1]}, 1
        
        L, R = best_biclique
        return L, R, best_size
    
    # Better approach: use a smarter greedy
    def greedy_biclique_partition(remaining_edges, adj):
        """Repeatedly extract bicliques from remaining edges."""
        bicliques = []
        rem = set(remaining_edges)
        
        while rem:
            if time.time() > deadline:
                break
            
            # Find a good biclique
            best_L = None
            best_R = None
            best_edge_count = 0
            
            # Build remaining adjacency
            rem_adj = [set() for _ in range(n)]
            rem_deg = [0] * n
            for u, v in rem:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            active = [i for i in range(n) if rem_deg[i] > 0]
            
            # Try pairs of high-degree vertices
            active_sorted = sorted(active, key=lambda x: -rem_deg[x])
            
            candidates = []
            for i in range(min(len(active_sorted), 15)):
                u = active_sorted[i]
                for j in range(i+1, min(len(active_sorted), 15)):
                    v = active_sorted[j]
                    if (u, v) not in rem and (v, u) not in rem:
                        continue
                    # Common neighbors in remaining graph
                    cn = rem_adj[u] & rem_adj[v]
                    # Check that (u,v) edge is in rem
                    key_uv = (min(u,v), max(u,v))
                    if key_uv not in rem:
                        continue
                    # Cross edges: u-c and v-c for c in cn, plus (u,v)
                    # Actually (u,v) is within-side if both in L
                    # Let's put L={u,v}, R=cn
                    # Cross edges are u-c and v-c for c in cn
                    edge_count = len(cn) * 2
                    # Verify all these edges are in rem
                    valid = True
                    for c in cn:
                        if (min(u,c), max(u,c)) not in rem:
                            valid = False
                            break
                        if (min(v,c), max(v,c)) not in rem:
                            valid = False
                            break
                    if valid:
                        candidates.append((edge_count, {u, v}, set(cn)))
            
            # Also try single vertex with all its remaining neighbors
            for u in active_sorted[:15]:
                r_set = set(rem_adj[u])
                if r_set:
                    edge_count = len(r_set)
                    candidates.append((edge_count, {u}, r_set))
            
            # Pick the best candidate
            if candidates:
                candidates.sort(key=lambda x: -x[0])
                best_edge_count, best_L, best_R = candidates[0]
            
            if best_L is None or not best_R:
                # Single edge fallback
                e = next(iter(rem))
                best_L = {e[0]}
                best_R = {e[1]}
                best_edge_count = 1
            
            # Add biclique
            bicliques.append(([min(best_L)], [min(best_R)] if len(best_R)==1 else sorted(best_R)))
            
            # Wait, we need to handle multi-vertex sides properly
            # Let me restructure
            
            # Remove covered edges from rem
            for u in best_L:
                for v in best_R:
                    key = (min(u, v), max(u, v))
                    rem.discard(key)
            
            # Store properly
            bicliques[-1] = [sorted(best_L), sorted(best_R)]
        
        return bicliques
    
    # Let me redo this more carefully
    def solve():
        rem = set()
        for u, v in edges:
            rem.add((min(u, v), max(u, v)))
        
        bicliques = []
        
        while rem:
            if time.time() > deadline:
                break
            
            # Build remaining adjacency
            rem_adj = [set() for _ in range(n)]
            rem_deg = [0] * n
            for u, v in rem:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            active = [i for i in range(n) if rem_deg[i] > 0]
            active_sorted = sorted(active, key=lambda x: -rem_deg[x])
            
            best_L = None
            best_R = None
            best_score = -1
            
            # Try pairs
            for i in range(min(len(active_sorted), 20)):
                u = active_sorted[i]
                for j in range(i+1, min(len(active_sorted), 20)):
                    v = active_sorted[j]
                    key_uv = (min(u, v), max(u, v))
                    if key_uv not in rem:
                        continue
                    cn = rem_adj[u] & rem_adj[v]
                    if not cn:
                        continue
                    # Verify all cross edges are in rem
                    valid = True
                    for c in cn:
                        if (min(u, c), max(u, c)) not in rem:
                            valid = False
                            break
                        if (min(v, c), max(v, c)) not in rem:
                            valid = False
                            break
                    if not valid:
                        continue
                    score = len(cn) * 2 + 1  # +1 for the (u,v) edge if we count it
                    # Actually (u,v) is a within-side edge (both in L), so it's not a cross edge
                    # But it needs to be covered... wait, no.
                    # The biclique covers cross edges between L and R.
                    # (u,v) where both u,v in L is a within-side edge, not covered by this biclique.
                    # So we only remove cross edges.
                    score = len(cn) * 2
                    if score > best_score:
                        best_score = score
                        best_L = {u, v}
                        best_R = set(cn)
            
            # Try single vertex
            for u in active_sorted[:20]:
                r_set = set(rem_adj[u])
                if r_set:
                    score = len(r_set)
                    if score > best_score:
                        best_score = score
                        best_L = {u}
                        best_R = r_set
            
            if best_L is None or not best_R:
                # Fallback: single edge
                e = next(iter(rem))
                best_L = {e[0]}
                best_R = {e[1]}
            
            # Remove cross edges from rem
            for u in best_L:
                for v in best_R:
                    key = (min(u, v), max(u, v))
                    rem.discard(key)
            
            bicliques.append([sorted(best_L), sorted(best_R)])
        
        return bicliques
    
    result = solve()
    
    # Verify and fix if needed
    if time.time() < deadline - 5:
        # Check if all edges are covered
        covered = set()
        for L, R in result:
            for u in L:
                for v in R:
                    covered.add((min(u, v), max(u, v)))
        
        missing = edge_set - covered
        if missing:
            # Add single-edge bicliques for missing edges
            for e in missing:
                result.append([list(e)])
        
        # Verify all cross edges are actual edges
        for L, R in result:
            for u in L:
                for v in R:
                    if (min(u, v), max(u, v)) not in edge_set:
                        # This biclique is invalid, need to fix
                        # For now, just return what we have (shouldn't happen)
                        pass
    
    # Convert to proper types
    final = []
    for L, R in result:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        final.append([L_int, R_int])
    
    return final
# EVOLVE-BLOCK-END
