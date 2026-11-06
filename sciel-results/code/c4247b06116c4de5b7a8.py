import time
import sys
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0

    def random_graph_edges(n, p, seed):
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_tomita(adj, n):
        """
        Find a maximum clique in the graph defined by adj (list of bitmasks).
        Uses Tomita's algorithm with bitmasks.
        Returns the clique as a list of vertices.
        """
        # We will use a recursive branch and bound.
        # To speed it up, we order vertices by degree.
        
        # Calculate degrees
        degrees = [adj[i].bit_count() for i in range(n)]
        # Sort vertices by degree descending
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Remap vertices to new indices based on order to improve pruning
        # new_index[v] = index in the sorted order
        new_index = [0] * n
        for i, v in enumerate(order):
            new_index[v] = i
            
        # Rebuild adjacency in the new order
        new_adj = [0] * n
        for i in range(n):
            u = order[i]
            # Map neighbors
            mask = 0
            temp = adj[u]
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if new_index[v] > i: # Only add edges to higher indices to avoid duplicates? 
                                       # No, Tomita usually works on the subgraph induced by candidates.
                                       # Let's just keep the full adjacency in new indices.
                    mask |= (1 << new_index[v])
                temp ^= low
            new_adj[i] = mask
            
        best_clique = []
        best_size = 0
        
        # Expansion stack: (current_clique, candidates_mask)
        # candidates_mask is a bitmask of vertices in the new indexing that are candidates
        # We start with all vertices
        # However, Tomita's algorithm typically expands one vertex at a time.
        
        # We'll use a simpler recursive formulation for clarity, relying on the small n.
        # But n=64 is too big for naive recursion without good pruning.
        # Tomita's MCQ (Maximum Clique Quick) is better.
        
        # Let's implement a standard branch and bound with bitmasks.
        
        def expand(clique, cand):
            nonlocal best_clique, best_size
            
            if not cand:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return
                
            # Upper bound: len(clique) + popcount(cand)
            if len(clique) + cand.bit_count() <= best_size:
                return
                
            # Order candidates by degree in the candidate set
            # To do this efficiently, we can just iterate and pick the one with max degree in cand
            # Or pre-sort? Pre-sorting is expensive inside recursion.
            # Tomita's method uses a static ordering based on global degree, which we did.
            # We just iterate through the bits in cand.
            
            # We can improve by picking the vertex with max degree in cand first.
            # Let's find the vertex in cand with max degree in cand.
            temp = cand
            best_u = -1
            best_deg = -1
            while temp:
                low = temp & (-temp)
                u = low.bit_length() - 1
                deg = (new_adj[u] & cand).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
                temp ^= low
                
            if best_u == -1:
                return

            # Remove best_u from cand
            new_cand = cand & ~(1 << best_u)
            
            # Add best_u to clique
            clique.append(best_u)
            
            # The new candidates must be neighbors of best_u in the new_cand set
            new_cand = new_cand & new_adj[best_u]
            
            expand(clique, new_cand)
            
            clique.pop()
            
            # Remove best_u from cand for the next iteration
            # We already did new_cand = cand & ~(1 << best_u) for the recursive call.
            # For the loop, we need to iterate over the original cand's bits.
            
            # To avoid re-scanning, we can just loop through the bits of the original cand.
            # But we modified cand? No, cand is passed by value (int).
            # We need to iterate through the bits of the original cand.
            
            # Let's restructure:
            # We pick a pivot, recurse, then remove pivot and continue.
            
            # Actually, the standard Tomita loop is:
            # while cand:
            #   pick u
            #   recurse(clique + u, cand & adj[u])
            #   cand = cand & ~(1 << u)
            
            # Let's implement that loop properly.
            
        # Revised Expand
        def expand_v2(clique, cand):
            nonlocal best_clique, best_size
            
            if not cand:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return
                
            if len(clique) + cand.bit_count() <= best_size:
                return
                
            while cand:
                # Pick vertex with max degree in cand
                temp = cand
                best_u = -1
                best_deg = -1
                while temp:
                    low = temp & (-temp)
                    u = low.bit_length() - 1
                    deg = (new_adj[u] & cand).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                    temp ^= low
                
                if best_u == -1:
                    break

                cand &= ~(1 << best_u)
                
                clique.append(best_u)
                new_cand = cand & new_adj[best_u]
                
                expand_v2(clique, new_cand)
                
                clique.pop()
                
                if time.time() > DEADLINE:
                    return

        expand_v2([], (1 << n) - 1)
        
        # Map back to original indices
        orig_clique = [order[i] for i in best_clique]
        return orig_clique

    def solve_instance(n, p, seed, deadline):
        adj = random_graph_edges(n, p, seed)
        rem = [adj[i] for i in range(n)]
        
        remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
        
        if remaining_edges == 0:
            return []
            
        cliques = []
        
        while remaining_edges > 0:
            if time.time() > deadline:
                break
                
            # Find max clique in rem
            # We need to pass rem to the max clique finder
            # My max_clique_tomita takes adj. I'll make a wrapper.
            
            # To save time, if the graph is small or sparse, we can use a simpler heuristic.
            # But for now, use Tomita.
            
            # Optimization: If max degree is 1, just take an edge.
            max_deg = max(rem[i].bit_count() for i in range(n))
            if max_deg == 1:
                # Find an edge
                for i in range(n):
                    if rem[i]:
                        low = rem[i] & (-rem[i])
                        j = low.bit_length() - 1
                        cliques.append([i, j])
                        rem[i] &= ~(1 << j)
                        rem[j] &= ~(1 << i)
                        remaining_edges -= 1
                        break
                continue
            
            clique = max_clique_tomita(rem, n)
            
            if not clique:
                # Should not happen if remaining_edges > 0
                break
                
            cliques.append(clique)
            
            # Remove edges
            m = len(clique)
            for i in range(m):
                for j in range(i + 1, m):
                    a, b = clique[i], clique[j]
                    rem[a] &= ~(1 << b)
                    rem[b] &= ~(1 << a)
                    
            remaining_edges -= m * (m - 1) // 2
            
        # Handle remaining
        if remaining_edges > 0:
            for i in range(n):
                if time.time() > deadline:
                    break
                temp = rem[i]
                while temp:
                    if time.time() > deadline:
                        break
                    low = temp & (-temp)
                    j = low.bit_length() - 1
                    cliques.append([i, j])
                    rem[i] &= ~(1 << j)
                    rem[j] &= ~(1 << i)
                    temp &= temp - 1
                    rem[i] &= ~(1 << j)
                    
        return cliques

    results = []
    time_per_instance = (DEADLINE - time.time()) / 2.0
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        t_start = time.time()
        t_limit = min(t_start + time_per_instance, DEADLINE)
        
        result = solve_instance(n, p, seed, t_limit)
        results.append(result)
        
    return results
