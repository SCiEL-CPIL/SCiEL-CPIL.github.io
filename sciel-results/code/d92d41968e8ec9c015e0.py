import random
import time
from itertools import combinations

def find_c5(n, adj):
    """Find a C5 in the graph defined by adj. Returns list of 5 vertices or None."""
    # adj is list of sets
    for verts in combinations(range(n), 5):
        v0, v1, v2, v3, v4 = verts
        # Check if induced subgraph is a C5
        # Edges must form a cycle
        # We can check if the number of edges is 5 and it's connected and 2-regular
        # Or just check the 12 permutations? 
        # Faster: check if (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v0) are all edges?
        # No, the vertices are in a set, order doesn't matter for the set, but for the cycle structure it does.
        # A set of 5 vertices forms a C5 if and only if the induced subgraph is isomorphic to C5.
        # We can just check if the induced subgraph has exactly 5 edges and all degrees are 2.
        
        edges_count = 0
        degs = [0]*5
        for i in range(5):
            for j in range(i+1, 5):
                if verts[j] in adj[verts[i]]:
                    edges_count += 1
                    degs[i] += 1
                    degs[j] += 1
        if edges_count == 5 and all(d == 2 for d in degs):
            return list(verts)
    return None

def find_clique_9(n, adj):
    """Find a clique of size 9. Returns list of 9 vertices or None."""
    # Simple branch and bound or just combinations?
    # 35C9 is large. But we can prune.
    # For now, a simple check:
    # Pick a random vertex, try to extend?
    # Or just check all combinations of 9? 35C9 ~ 70 million. Too slow.
    # Use a greedy heuristic to find a large clique, if it's >= 9, return it.
    # If greedy finds < 9, we might miss a clique of 9.
    # But for the purpose of "breaking" a clique, if we find a large one, we break it.
    # If we don't find one, we assume it's not there? Risky.
    # Let's use a basic backtracking for clique 9.
    
    def expand(clique, candidates):
        if len(clique) == 9:
            return clique
        for i in range(len(candidates)):
            v = candidates[i]
            new_clique = clique + [v]
            new_candidates = [u for u in candidates[i+1:] if u in adj[v]]
            res = expand(new_clique, new_candidates)
            if res:
                return res
        return None

    # Order vertices by degree descending to prune faster
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    res = expand([], order)
    return res

def construct():
    start_time = time.time()
    timeout = 115
    
    for n in range(35, 28, -1):
        for _ in range(50): # 50 restarts per n
            if time.time() - start_time > timeout:
                break
            
            # Initialize random graph H
            adj = [set() for _ in range(n)]
            p = random.uniform(0.3, 0.7)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Repair loop
            success = False
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # 1. Check C5 in H
                c5_h = find_c5(n, adj)
                if c5_h:
                    # Remove a random edge from the C5
                    # Pick random pair from c5_h that is an edge
                    edges_in_c5 = []
                    for i in range(5):
                        u = c5_h[i]
                        v = c5_h[(i+1)%5]
                        if v in adj[u]:
                            edges_in_c5.append((u, v))
                    if edges_in_c5:
                        u, v = random.choice(edges_in_c5)
                        adj[u].remove(v)
                        adj[v].remove(u)
                    continue

                # 2. Check C5 in complement of H
                # Complement adj: u~v in comp iff v not in adj[u]
                # We can build a complement adj list or check on the fly.
                # For speed, let's build a list of sets for complement?
                # Or modify find_c5 to take a "is_edge" function.
                # Let's just build comp_adj for this check.
                comp_adj = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i+1, n):
                        if j not in adj[i]:
                            comp_adj[i].add(j)
                            comp_adj[j].add(i)
                
                c5_comp = find_c5(n, comp_adj)
                if c5_comp:
                    # Add a random edge from the C5 in comp to H
                    # This removes it from comp
                    edges_in_c5_comp = []
                    for i in range(5):
                        u = c5_comp[i]
                        v = c5_comp[(i+1)%5]
                        if v in comp_adj[u]:
                            edges_in_c5_comp.append((u, v))
                    if edges_in_c5_comp:
                        u, v = random.choice(edges_in_c5_comp)
                        adj[u].add(v)
                        adj[v].add(u)
                    continue

                # 3. Check Clique 9 in H
                clique9 = find_clique_9(n, adj)
                if clique9:
                    # Remove a random edge from the clique
                    # Any pair in the clique is an edge
                    u, v = random.sample(clique9, 2)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    continue

                # If no violations found, success
                success = True
                break
            
            if success:
                # Convert adj to edge list for G (which is complement of H)
                # Wait, the task asks for G.
                # G is the graph we return.
                # G must be C5-free and alpha < 9.
                # We constructed H such that H is C5-free, comp(H) is C5-free, and omega(H) <= 8.
                # If we return G = comp(H), then:
                # G is C5-free (since comp(H) is C5-free).
                # alpha(G) = omega(H) <= 8 < 9.
                # So G is valid.
                
                edges_g = []
                for i in range(n):
                    for j in range(i+1, n):
                        if j not in adj[i]: # Edge in G
                            edges_g.append([i, j])
                
                return {'n': n, 'edges': edges_g}
    
    # Fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
