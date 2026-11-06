import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = {u: set() for u in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        partition = []
        
        while remaining_edges and time.time() < deadline:
            # Most-constrained-first: find the edge whose endpoints have the fewest
            # remaining common neighbors (fewest possible clique extensions)
            best_edge = None
            min_common = float('inf')
            
            for e in remaining_edges:
                u, v = e
                # Common neighbors of u and v that still have edges to both in remaining
                common_remaining = 0
                for w in adj[u]:
                    if w == v:
                        continue
                    if (min(u,w), max(u,w)) in remaining_edges and (min(v,w), max(v,w)) in remaining_edges:
                        common_remaining += 1
                
                if common_remaining < min_common:
                    min_common = common_remaining
                    best_edge = e
                if time.time() > deadline:
                    break
            
            if best_edge is None:
                break
            
            u, v = best_edge
            
            # Greedy clique growth starting from {u, v}
            clique = {u, v}
            # Candidates: vertices adjacent to all clique members in remaining_edges
            candidates = set()
            for w in adj[u]:
                if w == v:
                    continue
                if (min(u,w), max(u,w)) in remaining_edges and (min(v,w), max(v,w)) in remaining_edges:
                    candidates.add(w)
            
            while candidates and time.time() < deadline:
                # Pick candidate with highest degree within candidates
                best_cand = None
                best_deg = -1
                for w in candidates:
                    deg = 0
                    for x in candidates:
                        if x != w and (min(w,x), max(w,x)) in remaining_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_cand = w
                
                if best_cand is None:
                    break
                
                # Verify best_cand connects to all clique members
                valid = True
                for c in clique:
                    if (min(best_cand, c), max(best_cand, c)) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
                
                clique.add(best_cand)
                candidates.discard(best_cand)
                
                # Update candidates: must connect to all clique members
                new_candidates = set()
                for w in candidates:
                    ok = True
                    for c in clique:
                        if (min(w, c), max(w, c)) not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        new_candidates.add(w)
                candidates = new_candidates
            
            clique_list = sorted(clique)
            
            # Verify validity
            valid = True
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if (a, b) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid or len(clique_list) < 2:
                clique_list = [u, v]
                clique = {u, v}
            
            partition.append(clique_list)
            
            # Remove all edges within this clique
            to_remove = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    to_remove.add((a, b))
            remaining_edges -= to_remove
        
        # Fallback for remaining edges
        if remaining_edges:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 60)
    result2 = solve_instance(64, 0.55, 602, 65)
    
    return [result1, result2]
