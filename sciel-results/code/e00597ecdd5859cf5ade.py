"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def build_adjacency(edges):
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_cycle(adj, target_len, start_node=None):
        """
        Find a cycle of length target_len in the graph.
        Returns list of vertices or None.
        Uses DFS with pruning. Optimized for sparse graphs.
        """
        if start_node is None:
            start_node = 0
        
        # If start_node is not in graph, try another
        if not adj[start_node]:
            for i in range(N):
                if adj[i]:
                    start_node = i
                    break
            if not adj[start_node]:
                return None

        def dfs(path, visited, remaining):
            if remaining == 0:
                if path[0] in adj[path[-1]]:
                    return list(path)
                return None
            
            # Pruning: if remaining > max possible path from current node
            # (rough heuristic: if remaining > N - len(visited), but we track visited)
            # More effective: if remaining > 0 and no neighbors available that can close
            # We just try neighbors
            
            last = path[-1]
            # Order neighbors by degree or something? Maybe just iterate
            # To speed up, we can check if any neighbor is the start node if remaining==1
            if remaining == 1:
                if path[0] in adj[last]:
                    return list(path)
                return None
            
            for nxt in adj[last]:
                if nxt == path[0]:
                    continue # Only allowed at the end
                if nxt in visited:
                    continue
                path.append(nxt)
                visited.add(nxt)
                result = dfs(path, visited, remaining - 1)
                if result:
                    return result
                path.pop()
                visited.remove(nxt)
            return None

        path = [start_node]
        visited = {start_node}
        return dfs(path, visited, target_len - 1)

    def verify_pancyclic(edges):
        """
        Check if graph is pancyclic.
        Returns dict of cycles {length: list} or None if not pancyclic.
        """
        adj = build_adjacency(edges)
        cycles = {}
        
        # Check connectivity first (simple BFS/DFS)
        visited = set([0])
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(visited) < N:
            return None

        for L in range(3, N + 1):
            # Try to find cycle of length L
            # Start from 0 is usually good for circulant
            cycle = find_cycle(adj, L, 0)
            if cycle is None:
                # Try a few other starts if 0 fails (rare for circulant)
                found = False
                for s in range(1, 5):
                    cycle = find_cycle(adj, L, s)
                    if cycle:
                        found = True
                        break
                if not found:
                    return None
            cycles[L] = cycle
            
            # Time check
            if time.time() - start_time > deadline - 10:
                return None
                
        return cycles

    def get_edges_from_S(S):
        """Generate edge set from connection set S for circulant graph C(N, S)."""
        edges = set()
        for s in S:
            for i in range(N):
                u = i
                v = (i + s) % N
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
        return edges

    # Initial S: A known good pancyclic circulant?
    # S = {1, 2, 3} gives many cycles.
    # Let's start with a dense one and anneal.
    # S = {1, 2, 3, 4, 5} -> 5*40/2? No, undirected.
    # For N=40, S={1,2,3} has edges:
    # 1: 40 edges (cycle)
    # 2: 20 edges
    # 3: 40 edges (gcd(3,40)=1)
    # Total 100 edges. Too many.
    # We want to minimize |S|.
    # Lower bound on edges for pancyclic is ~N.
    # A circulant with S={1, k} might not be pancyclic.
    # Let's start with S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} and try to remove.
    # Or better, start with a random valid one if we can find one quickly,
    # or just start with S = {1, 2, 3} and ADD if invalid, REMOVE if valid.
    
    # Strategy: Simulated Annealing on S.
    # State: Set S.
    # Cost: |S| (proxy for edge count, but actually edge count depends on gcd).
    # Actually, minimizing edge count is the goal.
    # Edge count = sum_{s in S} (N / gcd(N, s)) / 2 ? 
    # No. For each s, the number of edges generated is N / gcd(N, s) if we consider directed?
    # Undirected edges for shift s:
    # If s = N/2, 1 edge? No, N/2 edges?
    # If s=1, 40 edges.
    # If s=2, 20 edges.
    # If s=20, 20 edges.
    # So cost is sum of N/gcd(N,s) for s in S? No, that's number of directed edges?
    # For s, the edges are {(i, i+s)}. There are N such pairs.
    # But (i, i+s) is same as (i+s, i) if we consider undirected.
    # Wait, the set of edges generated by s is { (i, (i+s)%N) | i in 0..N-1 }.
    # This set has size N / gcd(N, s) * 2? No.
    # If gcd(N,s)=d, the graph decomposes into d cycles of length N/d.
    # Total edges = d * (N/d) = N.
    # So EVERY shift s contributes exactly N/2 undirected edges?
    # Let's check s=1. N=40. Edges (0,1),(1,2)...(39,0). 40 edges.
    # s=2. Edges (0,2),(2,4)...(38,0) AND (1,3),(3,5)...(39,1).
    # (0,2) is distinct from (2,0). In undirected set, (0,2) is one edge.
    # The set { (i, (i+2)%40) } has 40 elements.
    # But (0,2) and (2,0) are the same edge? No, (2,0) is generated by i=2? No, (2, 4).
    # The edge (0,2) is generated by i=0. The edge (2,0) is not generated by i=2 (which gives 2,4).
    # Is (0,2) generated by i=2? (2, 4). No.
    # So for s=2, we have edges (0,2), (2,4), ..., (38,0). These are 20 edges?
    # No, (0,2), (1,3), ..., (39,1).
    # Let's list them for N=4, s=2.
    # i=0: (0,2). i=1: (1,3). i=2: (2,0)->(0,2). i=3: (3,1)->(1,3).
    # Unique edges: (0,2), (1,3). Count = 2.
    # Formula: N / gcd(N, s) / 2 ?
    # N=4, gcd(4,2)=2. 4/2/2 = 1? No, count is 2.
    # Wait. N/gcd(N,s) is the number of edges in the directed sense?
    # The number of unique undirected edges generated by shift s is N / gcd(N, s) / 2 if N/gcd(N,s) is even?
    # Actually, it's just N / 2 for any s that doesn't map i to i?
    # No.
    # Let's just calculate it.
    # For a given s, the edges are {(i, (i+s)%N)}.
    # If s != N/2, then (i, i+s) != (i+s, i).
    # The set of edges is a 2-regular graph? No.
    # It consists of gcd(N,s) disjoint cycles, each of length N/gcd(N,s).
    # Each cycle of length L has L edges.
    # Total edges = gcd(N,s) * (N/gcd(N,s)) = N.
    # Wait.
    # N=4, s=2. gcd=2. Cycles: 0-2-0 (length 2), 1-3-1 (length 2).
    # Edges: (0,2), (1,3). Count 2.
    # Formula N=4. Count 2.
    # N=4, s=1. gcd=1. Cycle 0-1-2-3-0 (length 4).
    # Edges: (0,1),(1,2),(2,3),(3,0). Count 4.
    # So count is N / 2 if s != N/2?
    # N=4, s=1 -> 4 edges. N/2 = 2. No.
    # It seems the number of edges is N / gcd(N, 2s)? No.
    # It is N / 2 if s is not N/2?
    # N=4, s=1 -> 4 edges.
    # N=4, s=2 -> 2 edges.
    # N=6, s=1 -> 6 edges.
    # N=6, s=2 -> gcd(6,2)=2. Cycles length 3. 2 cycles. 2*3=6 edges?
    # Edges: (0,2),(2,4),(4,0) and (1,3),(3,5),(5,1).
    # (0,2) and (2,0) are same.
    # In the set {(i, (i+2)%6)}, i=0 gives (0,2). i=2 gives (2,4). i=4 gives (4,0)=(0,4).
    # i=1 gives (1,3). i=3 gives (3,5). i=5 gives (5,1)=(1,5).
    # Unique edges: (0,2), (2,4), (0,4), (1,3), (3,5), (1,5).
    # Count 6.
    # So for s=2 in N=6, count is 6.
    # For s=3 in N=6 (N/2).
    # i=0: (0,3). i=1: (1,4). i=2: (2,5). i=3: (3,0)=(0,3).
    # Unique edges: (0,3), (1,4), (2,5). Count 3.
    # So if s = N/2, count is N/2.
    # Otherwise, count is N?
    # Let's check N=4, s=1. Count 4. N=4.
    # N=4, s=2 (N/2). Count 2. N/2.
    # So cost(S) = sum_{s in S} (N/2 if s==N/2 else N).
    # This means every element in S adds roughly N/2 edges (if s=N/2) or N edges.
    # Wait, if s=1 and s=3 in N=4.
    # s=1: 4 edges. s=3: 4 edges.
    # But edge (0,1) from s=1. Edge (0,3) from s=3.
    # They are distinct.
    # So total edges is additive?
    # Yes, because the edge sets for different s are disjoint?
    # Edge (u,v) has difference d = (v-u)%N or (u-v)%N.
    # If s1 != s2, can they generate the same edge?
    # (u, u+s1) = (u, u+s2) => s1=s2.
    # (u, u+s1) = (v, v+s2) with u!=v?
    # If (u, u+s1) is the same undirected edge as (v, v+s2).
    # Then {u, u+s1} = {v, v+s2}.
    # Case 1: u=v. Then s1=s2.
    # Case 2: u = v+s2 and u+s1 = v.
    # u = v+s2 => v = u-s2.
    # u+s1 = u-s2 => s1 = -s2 => s1+s2 = N.
    # So s and N-s generate the SAME set of edges!
    # In undirected graph, C(N, S) is same as C(N, S') where S' contains min(s, N-s).
    # So we should only consider s in 1..N/2.
    # And for s in 1..N/2-1, the edge count is N/2?
    # Let's re-evaluate N=4, s=1.
    # Edges: (0,1), (1,2), (2,3), (3,0).
    # Differences: 1, 1, 1, 1 (mod 4, min diff is 1).
    # Count 4.
    # N=4, s=2.
    # Edges: (0,2), (1,3).
    # Differences: 2, 2.
    # Count 2.
    # So for s < N/2, count is N/2?
    # N=6, s=1. Count 6?
    # Edges: (0,1),(1,2),(2,3),(3,4),(4,5),(5,0).
    # Min diffs: 1,1,1,1,1,1.
    # Count 6.
    # So count is N for s < N/2?
    # Wait.
    # If s=1, we have N edges.
    # If s=2 (N=6), we have 6 edges.
    # If s=3 (N/2), we have 3 edges.
    # So cost(s) = N if s != N/2, else N/2.
    # And s and N-s are the same.
    # So we restrict S to {1, ..., N/2}.
    # If we pick s and N-s, we double count? No, we just pick s.
    # So we search S subset of {1, ..., 20}.
    # Cost = sum_{s in S} (40 if s<20 else 20).
    
    # We want to minimize this cost.
    # Start with S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}.
    # This is K40. Cost = 19*40 + 20 = 780.
    # We want to remove elements.
    
    # Initial S: Try to find a small valid one.
    # S = {1, 2, 3} ?
    # Cost = 120.
    # Is it pancyclic?
    # C(40, {1,2,3}).
    # Likely yes.
    
    current_S = set([1, 2, 3, 4, 5])
    current_edges = get_edges_from_S(current_S)
    current_cycles = verify_pancyclic(current_edges)
    
    if current_cycles is None:
        # Expand S until valid
        s = 6
        while current_cycles is None and s <= 20:
            current_S.add(s)
            current_edges = get_edges_from_S(current_S)
            current_cycles = verify_pancyclic(current_edges)
            s += 1
            if time.time() - start_time > deadline - 10:
                break
        
        if current_cycles is None:
            # Fallback
            current_S = set(range(1, 21))
            current_edges = get_edges_from_S(current_S)
            current_cycles = verify_pancyclic(current_edges)
            if current_cycles is None:
                return None

    best_S = set(current_S)
    best_edges = set(current_edges)
    best_cycles = current_cycles
    best_cost = len(best_edges)
    
    # Annealing
    temp = 5.0
    cooling = 0.95
    
    while time.time() - start_time < deadline - 5:
        # Move: Add or Remove an element from S
        if len(current_S) > 1 and random.random() < 0.7:
            # Remove
            s_to_remove = random.choice(list(current_S))
            new_S = set(current_S)
            new_S.discard(s_to_remove)
        else:
            # Add
            candidates = [s for s in range(1, 21) if s not in current_S]
            if not candidates:
                continue
            s_to_add = random.choice(candidates)
            new_S = set(current_S)
            new_S.add(s_to_add)
            
        new_edges = get_edges_from_S(new_S)
        new_cycles = verify_pancyclic(new_edges)
        
        if new_cycles is not None:
            new_cost = len(new_edges)
            delta = new_cost - best_cost
            
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_S = new_S
                current_edges = new_edges
                current_cycles = new_cycles
                
                if new_cost < best_cost:
                    best_S = new_S
                    best_edges = new_edges
                    best_cycles = new_cycles
                    best_cost = new_cost
        else:
            # Invalid, reject (or accept with low prob? No, must be valid)
            pass
            
        temp *= cooling
        
        # If we reached a very low cost, try to polish
        if best_cost <= 80:
             # Try local search on best_S
             # Try removing each element one by one
             for s in list(best_S):
                 trial_S = set(best_S)
                 trial_S.discard(s)
                 trial_edges = get_edges_from_S(trial_S)
                 trial_cycles = verify_pancyclic(trial_edges)
                 if trial_cycles is not None:
                     best_S = trial_S
                     best_edges = trial_edges
                     best_cycles = trial_cycles
                     best_cost = len(best_edges)
                     current_S = best_S
                     current_edges = best_edges
                     current_cycles = best_cycles

    # Final verification
    final_cycles = verify_pancyclic(best_edges)
    if final_cycles is None:
        return None
        
    edge_list = sorted(best_edges)
    cycles_out = []
    for L in range(3, N+1):
        cycles_out.append(final_cycles[L])
        
    return {'edges': edge_list, 'cycles': cycles_out}

# EVOLVE-BLOCK-END

# Call the construct function
construct()
