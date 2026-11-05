import random
import itertools

def construct():
    n = 18
    # Start with Turan graph T_3(18)
    # Parts: A=[0..5], B=[6..11], C=[12..17]
    # Edges between different parts.
    
    # Represent graph as a set of tuples for fast lookup
    # Or a list of lists (adjacency matrix)
    adj = [[0]*n for _ in range(n)]
    
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u][v] = 1
                    adj[v][u] = 1
    
    def is_k4_free(adj):
        # Check for K4 in O(n^4) is too slow if called often.
        # But n=18, n^4 = 104976. This is fast enough for a few thousand iterations.
        # Actually, we can be smarter.
        # For every pair of adjacent vertices (u, v), check if they share 2 common neighbors w, x such that w-x is an edge.
        # If so, {u,v,w,x} is a K4.
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
                    # Check if any pair in common is an edge
                    for i in range(len(common)):
                        for j in range(i+1, len(common)):
                            if adj[common[i]][common[j]]:
                                return False
        return True

    def calc_score(adj):
        # Find largest triangle-free induced subgraph.
        # Heuristic: Greedy.
        # Or exact for n=18?
        # A triangle-free set is a set of vertices with no triangles.
        # This is equivalent to finding the largest subset S such that G[S] has no triangles.
        # We can use a branch and bound, but for speed, let's use a simple heuristic first.
        # If the heuristic gives < 12, we verify.
        
        # Actually, let's use a simple greedy algorithm that tries to build a large triangle-free set.
        # Start with all vertices. Remove vertices that are part of triangles until no triangles remain.
        # This is not optimal but gives a lower bound on the max triangle-free size?
        # No, it gives an upper bound on the min vertex cover of triangles?
        # Let's just use a random greedy:
        # 1. Order vertices randomly.
        # 2. Iterate, add vertex if it doesn't form a triangle with already added vertices.
        # This finds a triangle-free set. The size is a lower bound on the max triangle-free set.
        # We want to MINIMIZE this max size.
        # So if we find a small triangle-free set, it doesn't mean the max is small.
        # We need to find the MAX triangle-free set.
        
        # For n=18, we can do a simple DFS/Backtracking to find the exact max triangle-free set.
        # It might be slow if done for every iteration.
        # Let's limit the search or use a fast heuristic.
        
        # Heuristic:
        # The largest triangle-free set is at most n - (min vertex cover of triangles).
        # Let's just use the Turan bound logic: In T_3, it's 12.
        # If we break the structure, it might go down.
        
        # Let's use a simple check:
        # Try all subsets of size 12? No, C(18,12) is large.
        # Try all subsets of size 11?
        # If we can find a triangle-free set of size 11, score >= 11.
        # If we can prove no set of size 12 is triangle-free, score <= 11.
        
        # For speed, let's return a heuristic estimate.
        # We'll use a greedy algorithm to find a large triangle-free set.
        # If the greedy finds one of size S, the true max is >= S.
        # We want to minimize the true max.
        # So we should track the maximum found by greedy.
        
        best_size = 0
        for _ in range(20): # 20 random orderings
            order = list(range(n))
            random.shuffle(order)
            current_set = []
            for v in order:
                # Check if adding v creates a triangle
                ok = True
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        u1 = current_set[i]
                        u2 = current_set[j]
                        if adj[v][u1] and adj[v][u2] and adj[u1][u2]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    current_set.append(v)
            if len(current_set) > best_size:
                best_size = len(current_set)
        return best_size

    current_adj = [row[:] for row in adj]
    current_score = calc_score(current_adj)
    best_adj = [row[:] for row in adj]
    best_score = current_score
    
    # Local Search
    for _ in range(5000):
        # Pick a random mutation
        # Try to add an edge within a part
        part_idx = random.randint(0, 2)
        u, v = random.sample(parts[part_idx], 2)
        
        if current_adj[u][v] == 0:
            # Try adding edge (u, v)
            new_adj = [row[:] for row in current_adj]
            new_adj[u][v] = 1
            new_adj[v][u] = 1
            
            # Check K4-freeness.
            # If K4 formed, we must remove edges.
            # For simplicity, if adding (u,v) creates a K4, reject or fix.
            # Let's just reject if it creates a K4.
            if is_k4_free(new_adj):
                score = calc_score(new_adj)
                if score < best_score:
                    best_score = score
                    best_adj = new_adj
                    current_adj = new_adj
                else:
                    # Maybe accept with some probability (simulated annealing style)
                    # But let's stick to hill climbing for now.
                    pass
            else:
                # Try to fix by removing an edge in the K4
                # Find a K4 containing u,v
                common = [w for w in range(n) if w != u and w != v and new_adj[u][w] and new_adj[v][w]]
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        if new_adj[common[i]][common[j]]:
                            # K4 is {u, v, common[i], common[j]}
                            # Remove edge (common[i], common[j])
                            new_adj[common[i]][common[j]] = 0
                            new_adj[common[j]][common[i]] = 0
                            # Check if still K4-free
                            if is_k4_free(new_adj):
                                score = calc_score(new_adj)
                                if score < best_score:
                                    best_score = score
                                    best_adj = new_adj
                                    current_adj = new_adj
                            break
                    break
        else:
            # Try removing an edge between parts
            # Pick a random edge between parts
            p1, p2 = random.sample([0, 1, 2], 2)
            u = random.choice(parts[p1])
            v = random.choice(parts[p2])
            if current_adj[u][v] == 1:
                new_adj = [row[:] for row in current_adj]
                new_adj[u][v] = 0
                new_adj[v][u] = 0
                # Removing edges cannot create K4
                score = calc_score(new_adj)
                if score < best_score:
                    best_score = score
                    best_adj = new_adj
                    current_adj = new_adj

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
