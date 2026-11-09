import time
import sys
import random

sys.setrecursionlimit(10000)

N = 256
conflict = [0] * (N + 1)

def compute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))

compute_conflicts()

# Precompute degrees for initial ordering
initial_degrees = [0] * (N + 1)
for i in range(1, N + 1):
    initial_degrees[i] = bin(conflict[i]).count('1')

def popcount(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    best_size = 0
    best_mask = 0
    
    initial_mask = (1 << N) - 1
    
    # Greedy heuristic to find a good initial lower bound
    # Sort by degree descending
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: -initial_degrees[x])
    
    greedy_mask = 0
    for v in vertices:
        # Check if v conflicts with any in greedy_mask
        # We can just check (conflict[v] & greedy_mask) == 0
        if (conflict[v] & greedy_mask) == 0:
            greedy_mask |= (1 << (v - 1))
            
    best_size = popcount(greedy_mask)
    best_mask = greedy_mask
    
    # Iterative Deepening A* or similar? 
    # Let's stick to Branch and Bound but with better pruning.
    # We will use a recursive search.
    
    def search(candidates_mask, current_mask, current_size, depth):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        # Pruning: trivial upper bound
        if current_size + popcount(candidates_mask) <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Greedy coloring bound
        # This is expensive, so maybe only use it when close to the bound?
        # Or use a cheaper bound: max independent set in the induced subgraph
        # A simple upper bound is the size of the maximum independent set of the conflict graph.
        # For the conflict graph, an independent set is a set of vertices with no conflicts.
        # Wait, the problem asks for a set where NO pair has conflict.
        # So we are looking for an Independent Set in the conflict graph.
        # The size of the Maximum Independent Set (MIS) is what we want to maximize.
        # The upper bound for MIS in a graph G is the number of vertices.
        # A tighter bound can be obtained by coloring the graph. The chromatic number is an upper bound on the size of the clique cover? No.
        # The size of the Maximum Independent Set is at most the number of colors in any coloring of the graph? No, that's for Clique.
        # Actually, if we color the graph with k colors, the largest independent set is at most the size of the largest color class? No.
        # Theorem: The size of the Maximum Independent Set is at most the number of colors in a coloring? No.
        # Let's use the fact that if we partition the vertices into cliques, the MIS is at most the number of cliques.
        # Or simply: The MIS of G is the Clique of G_complement.
        # Coloring G_complement gives a lower bound on the clique number of G_complement? No.
        # Coloring G gives an upper bound on the Clique Number of G? No, Coloring G gives an upper bound on the Independence Number of G? No.
        # Actually, if G is k-colorable, then the independence number alpha(G) <= n? No.
        # Let's use the simple bound: alpha(G) <= n - delta? No.
        # Let's use the greedy coloring of the CONFLICT graph.
        # If the conflict graph is k-colorable, does that bound the independent set?
        # No, a complete graph is 1-colorable? No, complete graph K_n is n-colorable.
        # An independent set in G is a set of vertices no two of which are adjacent.
        # In a k-coloring, each color class is an independent set.
        # So the size of the MIS is at least the size of the largest color class.
        # This is a LOWER bound, not an upper bound.
        
        # We need an UPPER bound on the size of the MIS in the remaining candidates.
        # A common upper bound is the number of vertices in the largest clique in the complement?
        # Or simply: The size of the MIS is at most the number of vertices.
        # A better bound: If we find a clique of size C in the conflict graph, we can pick at most 1 vertex from it.
        # So if we partition the candidates into cliques, the sum of (1 for each clique) is an upper bound.
        # This is equivalent to finding a clique cover.
        # A simple clique cover is to just use single vertices (bound = n).
        # We can try to greedily find cliques to get a tighter bound.
        
        # Let's try a simple greedy clique cover bound.
        # Greedily pick a vertex, add its neighbors that are also connected to it (forming a clique), remove them, repeat.
        # This gives a set of cliques. The number of cliques is an upper bound on the MIS.
        
        def clique_cover_bound(mask):
            if mask == 0:
                return 0
            count = 0
            remaining = mask
            while remaining:
                # Pick a vertex with max degree in remaining
                max_deg = -1
                v_idx = -1
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    val = idx + 1
                    deg = popcount(conflict[val] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        v_idx = idx
                    temp ^= lsb
                
                if v_idx == -1:
                    break
                    
                v_bit = 1 << v_idx
                v_val = v_idx + 1
                
                # Find a clique containing v_val in remaining
                # Start with v_val, then greedily add neighbors that are connected to all current members
                clique = v_bit
                candidates_for_clique = conflict[v_val] & remaining
                # To ensure it's a clique, we need intersection of neighborhoods
                # For simplicity, let's just take v_val and its neighbors that are connected to v_val?
                # No, that's a star, not a clique.
                # A clique requires all pairs to be connected.
                # Let's just use single vertices as cliques for now if clique finding is too complex/slow.
                # Or use the fact that the neighborhood of a vertex induces a subgraph.
                
                # Let's use a simpler bound: The size of the MIS is at most the size of the maximum independent set of the conflict graph.
                # We can use the greedy coloring of the COMPLEMENT graph?
                # No, let's stick to the simple bound for now and rely on the search order.
                # The trivial bound popcount(candidates_mask) is already used.
                # Let's try to improve the search order.
                
                count += 1
                remaining &= ~v_bit
                
            return count
        
        # The clique cover bound above is just n if we don't actually form cliques.
        # Let's implement a proper greedy clique cover.
        
        def greedy_clique_cover_bound(mask):
            if mask == 0:
                return 0
            count = 0
            remaining = mask
            while remaining:
                # Find a maximal clique in remaining
                # Greedy: pick a vertex, then add neighbors connected to all existing
                # To make it fast, just pick the vertex with max degree, then its neighbors?
                # Let's just pick one vertex per "clique" for now? No, that's not a cover.
                # A valid cover is a partition into cliques.
                # If we just remove one vertex at a time, the bound is n.
                # To get a tighter bound, we need to remove multiple vertices that form a clique.
                
                # Let's find a maximal clique starting from the max-degree vertex.
                max_deg = -1
                v_idx = -1
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    val = idx + 1
                    deg = popcount(conflict[val] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        v_idx = idx
                    temp ^= lsb
                
                if v_idx == -1:
                    break
                    
                v_bit = 1 << v_idx
                v_val = v_idx + 1
                
                # Greedily build a clique
                clique = v_bit
                # Candidates must be neighbors of v_val
                candidates = conflict[v_val] & remaining
                # And neighbors of each other?
                # This is getting complex. Let's just use the trivial bound for now.
                # The search tree might be small enough.
                
                count += 1
                remaining &= ~v_bit
                
            return count

        # Since the clique cover bound is not implemented tightly, we rely on the search.
        # Let's try to order the vertices in candidates_mask by degree to pick the best branching variable.
        
        max_deg = -1
        v_idx = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            val = idx + 1
            deg = popcount(conflict[val] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                v_idx = idx
            temp_mask ^= lsb
            
        if v_idx == -1:
            return

        v_bit = 1 << v_idx
        v_val = v_idx + 1
        
        # Branch 1: Include v
        new_candidates = candidates_mask & ~(conflict[v_val] | v_bit)
        search(new_candidates, current_mask | v_bit, current_size + 1, depth + 1)
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~v_bit
        search(new_candidates, current_mask, current_size, depth + 1)

    search(initial_mask, 0, 0, 0)
    
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
