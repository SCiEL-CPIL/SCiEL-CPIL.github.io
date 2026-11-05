import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # The goal is to find a triangle-free graph with alpha < 10.
    # Maximize n.
    # Known results: R(3,10) <= 41.
    # It is known that R(3,10) > 40 is possible? Or is it 39?
    # According to Ramsey number tables, R(3,10) is known to be in [40, 41] or [39, 41]?
    # Actually, the largest known triangle-free graph with alpha=9 has n=39.
    # Some sources say R(3,10) > 40 has been proven?
    # Let's check: R(3,10) > 40 was proved by Exoo? Or is it still open?
    # Standard tables often cite R(3,10) > 39.
    # However, there is a 40-vertex graph?
    # Let's try to construct candidates for n=40 and n=39 using algebraic methods.
    
    # Strategy:
    # 1. Try n=40.
    # 2. Try n=39.
    # 3. If neither works, try a fast local search for n=39 as fallback.
    
    def is_triangle_free(adj_masks, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    if adj_masks[u] & adj_masks[v]:
                        return False
        return True

    def has_independent_set(adj_masks, n, k):
        """
        Returns True if an independent set of size k exists.
        Uses branch and bound with greedy coloring bound on the complement graph (which is the original graph's non-neighborhoods).
        Actually, finding independent set in G is finding clique in complement of G.
        Let's use a standard maximal independent set search with pruning.
        """
        full_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            # non_neighbors[i] contains vertices j != i such that (i,j) is NOT an edge
            non_neighbors[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        
        # We want to find a set of size k such that no two are adjacent.
        # This is equivalent to finding a clique of size k in the graph H where edges are non-edges of G.
        # H_adj[i] = non_neighbors[i]
        
        found = [False]
        
        def dfs(candidates, current_size, current_set):
            if found[0]:
                return
            if current_size == k:
                found[0] = True
                return
            
            # Pruning: if not enough vertices left
            if bin(candidates).count('1') < k - current_size:
                return
                
            # Greedy coloring bound on the subgraph induced by candidates in H
            # To prove no clique of size k-current_size exists, we can color the vertices of H[candidates]
            # If we can color H[candidates] with c colors, then max clique size <= c.
            # So if c < k - current_size, we can prune.
            
            # Efficient greedy coloring for small n (<=40)
            # We pick vertices in some order (e.g., degree or just index) and assign colors.
            # A simple bound: if we can't even find k-current_size independent vertices in H (i.e., clique in H), 
            # we prune.
            
            # Let's use a simpler, fast bound: 
            # If the maximum degree in H[candidates] is small? No.
            # Let's just use the basic count and maybe a greedy independent set in H (which is clique in G) 
            # No, we are looking for clique in H.
            
            # A common fast bound for clique finding: 
            # Sort candidates by degree in H[candidates] (descending).
            # If we can greedily color the graph H[candidates] with < (k - current_size) colors, prune.
            
            # Implementing greedy coloring for n=40 is cheap.
            cands_list = []
            mask = candidates
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                cands_list.append(v)
                mask ^= lsb
            
            if not cands_list:
                return

            # Greedy coloring of H[cands_list]
            # Colors are integers. color[v] = color index.
            colors = {}
            next_color = 0
            for v in cands_list:
                # Check which colors are forbidden by neighbors in H
                forbidden = set()
                for u in cands_list:
                    if u == v: continue
                    if u in colors and (non_neighbors[v] & (1 << u)):
                        forbidden.add(colors[u])
                
                c = 0
                while c in forbidden:
                    c += 1
                colors[v] = c
                if c > next_color:
                    next_color = c
            
            # If number of colors used < required clique size, prune
            if next_color + 1 < k - current_size:
                return
            
            # Branch
            # Pick a vertex. Heuristic: highest degree in H[candidates] or just first.
            # Let's pick the vertex with highest degree in H[candidates] to maximize pruning?
            # Or just the first one for simplicity in this small n.
            # Actually, branching on the vertex with highest degree in the candidate set is good.
            
            # Calculate degrees in H[candidates]
            degs = {}
            for v in cands_list:
                deg = 0
                for u in cands_list:
                    if u != v and (non_neighbors[v] & (1 << u)):
                        deg += 1
                degs[v] = deg
                
            # Pick max degree
            v_pick = max(cands_list, key=lambda x: degs[x])
            
            # Branch 1: Include v_pick
            # New candidates: neighbors of v_pick in H (i.e., non-neighbors in G) that are in candidates
            new_cands = candidates & non_neighbors[v_pick] & ~(1 << v_pick)
            dfs(new_cands, current_size + 1, current_set + [v_pick])
            
            if found[0]:
                return
            
            # Branch 2: Exclude v_pick
            dfs(candidates & ~(1 << v_pick), current_size, current_set)

        dfs(full_mask, 0, [])
        return found[0]

    def try_n(n, attempts=5, time_limit=10):
        """
        Tries to find a valid graph of size n.
        Uses randomized local search.
        Returns the graph dict if found, else None.
        """
        full_mask = (1 << n) - 1
        
        for _ in range(attempts):
            if time.time() > start_time + time_limit:
                break
                
            # Initialize: Random graph with p=0.5
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            # Phase 1: Remove triangles
            # We do this iteratively.
            for _ in range(500):
                if time.time() > start_time + time_limit:
                    break
                tri_found = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_masks[u] & (1 << v):
                            common = adj_masks[u] & adj_masks[v]
                            if common:
                                # Remove edge (u,v)
                                adj_masks[u] &= ~(1 << v)
                                adj_masks[v] &= ~(1 << u)
                                tri_found = True
                                break
                    if tri_found:
                        break
                if not tri_found:
                    break
            
            # Phase 2: Reduce Alpha
            # We want alpha < 10.
            # Check if alpha >= 10.
            for _ in range(100):
                if time.time() > start_time + time_limit:
                    break
                
                if not has_independent_set(adj_masks, n, 10):
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj_masks[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    # Find an independent set of size 10 to guide edge addition
                    # We need the actual set, not just boolean.
                    # Let's modify has_independent_set to return the set?
                    # Or just pick random non-edges and add them if they don't create triangles?
                    
                    # Strategy: Pick a random pair (u,v) that is not an edge.
                    # If adding it doesn't create a triangle, add it.
                    # This is a random walk towards denser graph.
                    improved = False
                    # Try a few random pairs
                    for _ in range(20):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u == v: continue
                        if not (adj_masks[u] & (1 << v)):
                            # Check if adding creates triangle
                            # Triangle if common neighbor exists
                            if not (adj_masks[u] & adj_masks[v]):
                                adj_masks[u] |= (1 << v)
                                adj_masks[v] |= (1 << u)
                                improved = True
                                break
                    if not improved:
                         # Stuck? Break and restart? Or just continue?
                         # If we can't add any edge without triangle, we are at a local max for this structure.
                         break

        return None

    # Try n=40 first (theoretical max)
    # Note: R(3,10) > 40 is not widely confirmed in standard tables as a *constructed* example, 
    # but R(3,10) <= 41. So n=40 is the target if it exists.
    # Most literature cites 39 as the known max.
    # Let's try 40 with a tight time budget.
    
    result = try_n(40, attempts=10, time_limit=30)
    if result:
        return result
        
    # Try n=39
    result = try_n(39, attempts=20, time_limit=60)
    if result:
        return result
        
    # Fallback: Try a specific algebraic construction if available?
    # Or just return a valid smaller graph if we run out of time.
    # A 26-vertex Andrásfai graph is safe.
    
    # Let's try one more aggressive search for 39 if we have time
    if time.time() < deadline - 5:
        result = try_n(39, attempts=10, time_limit=10)
        if result:
            return result
            
    # Safe fallback
    n = 26
    # Andrásfai graph A_3(26)?
    # Vertices 0..25.
    # Edges (i, j) if (j-i) mod 26 in {1, 3, 9, 11, 15, 21, 23, 25}? 
    # Actually, let's just use a random valid small graph or a simple structure.
    # A complete bipartite graph K_{13,13} is triangle free.
    # Alpha of K_{13,13} is 13 (take one partition). 13 >= 10. Invalid.
    # We need alpha < 10.
    # K_{9,9} has n=18, alpha=9. Valid.
    # Let's return K_{9,9}.
    n = 18
    edges = []
    for i in range(9):
        for j in range(9, 18):
            edges.append([i, j])
    return {'n': n, 'edges': edges}

# Ensure random seed is set for reproducibility if needed, but random is fine
import sys
if __name__ == "__main__":
    # Quick self-test
    pass
