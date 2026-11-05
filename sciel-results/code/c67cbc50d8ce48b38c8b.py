import time
import random
import sys

N = 40

def has_c6_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    adj is list of sets.
    We look for a path u-a-b-c-d-v where a in N(u), d in N(v), and a-b-c-d is a path of length 3.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We need to find a, b, c, d such that:
    # a in nu, (a,b) in E, (b,c) in E, (c,d) in E, d in nv
    # All vertices u,a,b,c,d,v distinct.
    
    # Optimization: 
    # 1. Compute set of all 'b' reachable from nu in 1 step (excluding u).
    #    Let S_b = { b | exists a in nu, b in adj[a], b != u }
    # 2. Compute set of all 'c' reachable from nv in 1 step (excluding v).
    #    Let S_c = { c | exists d in nv, c in adj[d], c != v }
    # 3. Check if there is an edge (b,c) such that b in S_b and c in S_c.
    #    If so, we have a path a-b-c-d.
    #    We must verify distinctness.
    
    # Build S_b:
    S_b = set()
    for a in nu:
        for b in adj[a]:
            if b != u:
                S_b.add(b)
                
    # Build S_c:
    S_c = set()
    for d in nv:
        for c in adj[d]:
            if c != v:
                S_c.add(c)
                
    # Now check for edge (b,c) with b in S_b, c in S_c.
    for b in S_b:
        for c in adj[b]:
            if c in S_c:
                # We have a candidate path structure a-b-c-d.
                # We need to verify that there exist a in nu and d in nv
                # such that:
                # 1. (a,b) in E  (a in nu, a in adj[b])
                # 2. (c,d) in E  (d in nv, d in adj[c])
                # 3. Distinctness:
                #    u, a, b, c, d, v all distinct.
                
                if c == u or b == v:
                    continue
                    
                # Find valid a and d
                # Candidates for a: a in nu, a in adj[b], a != c
                # Candidates for d: d in nv, d in adj[c], d != b
                
                found = False
                for a in nu:
                    if a == c:
                        continue
                    if a not in adj[b]:
                        continue
                    # a is valid candidate
                    for d in nv:
                        if d == b or d == a:
                            continue
                        if d not in adj[c]:
                            continue
                        # d is valid candidate
                        found = True
                        break
                    if found:
                        break
                if found:
                    return True
                    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    best_edges = []
    best_count = 0
    
    random.seed(42)
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Randomized Greedy Construction (from current program)
    # Run multiple restarts to find a good starting point
    
    # Optimization: Use a bitset representation for adjacency?
    # Python sets are okay for N=40.
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
            
        adj = [set() for _ in range(N)]
        edges = []
        
        # Shuffle edges
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if u in adj[v]:
                continue
            if not has_c6_fast(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    # If we already have a very good result, we can stop early
    if best_count >= 105:
        return best_edges
    
    # Phase 2: Simulated Annealing / Local Search (from inspiration 2)
    # Start from the best greedy solution and try to improve via 1-swaps
    
    if best_count > 0:
        # Initialize adjacency sets from best_edges
        adj = [set() for _ in range(N)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        current_edges = best_edges[:]
        current_count = best_count
        
        # Temperature for simulated annealing
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.995
        
        steps = 0
        max_steps = 50000
        
        while time.time() < deadline - 5 and T > T_min:
            steps += 1
            
            # Pick a random edge to remove
            if not current_edges:
                break
                
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            # Remove edge (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            current_edges.remove([u, v])
            current_count -= 1
            
            # Try to add a different edge
            # We can pick a random non-edge, or a non-edge that was previously blocked
            # For simplicity, pick a random non-edge
            # Generate a random non-edge
            attempts = 0
            added = False
            while attempts < 10 and not added:
                attempts += 1
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    continue
                if a in adj[b]:
                    continue
                # Check if adding (a, b) creates a C6
                if not has_c6_fast(adj, a, b):
                    adj[a].add(b)
                    adj[b].add(a)
                    current_edges.append([a, b])
                    current_count += 1
                    added = True
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = current_edges[:]
            
            # If we couldn't add an edge, revert the removal
            if not added:
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v])
                current_count += 1
            
            # Cool down
            T *= cooling_rate
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
