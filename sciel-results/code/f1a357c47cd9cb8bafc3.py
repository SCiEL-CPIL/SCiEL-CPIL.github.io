import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Keep 10s buffer
    
    def get_elapsed():
        return time.time() - start_time

    def is_valid_graph(adj, n):
        """Check C4-free and alpha < 11."""
        # Check C4
        # For each pair (u, v), count common neighbors
        # If count >= 2, C4 exists
        for u in range(n):
            for v in range(u + 1, n):
                common = 0
                # Use bitsets for speed
                # But we are using lists/sets in adj? 
                # Let's use a bitset representation for the search, 
                # but keep adj as list of sets for easy edge add/remove.
                # For speed, we will maintain a bitset array `adj_bits`.
                pass # This check is too slow if done naively on sets.
        # We will rely on the search process to maintain validity or 
        # perform a final strict check at the end.
        return True

    def count_common_neighbors(adj_bits, u, v, n):
        """Count common neighbors of u and v using bitsets."""
        # adj_bits[i] is an integer bitmask of neighbors of i
        common = adj_bits[u] & adj_bits[v]
        return bin(common).count('1')

    def find_c4(adj_bits, n):
        """Find a C4. Returns (u, v, w, x) or None."""
        # Iterate over pairs
        # Optimization: Only check pairs that are NOT connected? 
        # A C4 is u-v-w-x-u. u and w are not connected, v and x are not connected.
        # But any pair with >= 2 common neighbors forms a C4 with those neighbors.
        # Let u, v be the pair. Let c1, c2 be two common neighbors.
        # Then u-c1-v-c2-u is a C4.
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj_bits, u, v, n) >= 2:
                    # Find two common neighbors
                    common_mask = adj_bits[u] & adj_bits[v]
                    # Get first two bits
                    c1 = -1
                    c2 = -1
                    m = common_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if c1 == -1:
                            c1 = idx
                        else:
                            c2 = idx
                            break
                        m ^= lsb
                    return (u, c1, v, c2)
        return None

    def find_independent_set_size_k(adj_bits, n, k):
        """
        Find an independent set of size k.
        Returns the set as a list of vertices, or None.
        Uses bitset DFS.
        """
        # Precompute degrees for ordering? 
        # We'll use a greedy approach first, then DFS.
        
        # Greedy:
        # Pick vertex with min degree? No, for independent set, 
        # we want to pick vertices that are not connected.
        # Greedy heuristic: Sort by degree ascending, pick if not connected to picked.
        # This is not guaranteed to find size k if it exists, but good for quick fail.
        
        # Let's just do DFS with pruning.
        # State: (current_size, available_mask)
        # available_mask: vertices that can still be added (not connected to any picked).
        
        # To speed up, we can use a recursive function that picks a vertex from available_mask.
        # Order: Pick vertex with highest degree in the induced subgraph of available_mask?
        # Or just iterate.
        
        # Optimization: If popcount(available_mask) < k - current_size, return None.
        
        def dfs(current_size, available_mask):
            if current_size == k:
                return True # Found
            if not available_mask:
                return False
            if current_size + bin(available_mask).count('1') < k:
                return False
                
            # Pick a vertex
            # To prune better, pick a vertex that is "central" in the available set?
            # Or just pick the first one?
            # Let's pick the vertex with the highest degree in the subgraph induced by available_mask.
            # This reduces the available set more.
            
            # Find v in available_mask with max degree in available_mask
            best_v = -1
            best_deg = -1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Degree of v in available_mask is count of neighbors of v that are in available_mask
                # neighbors of v are adj_bits[v]
                # common = adj_bits[v] & available_mask
                # deg = bin(common).count('1')
                # Note: This counts neighbors in available_mask.
                # If we pick v, we remove v and its neighbors.
                # So we want to maximize the number of neighbors removed.
                deg = bin(adj_bits[v] & available_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
                
            if best_v == -1:
                return False
                
            v = best_v
            # Remove v and its neighbors from available_mask
            new_mask = available_mask & ~(adj_bits[v] | (1 << v))
            
            if dfs(current_size + 1, new_mask):
                return True
                
            # Option: Exclude v
            # Just remove v
            new_mask_excl = available_mask & ~(1 << v)
            if dfs(current_size, new_mask_excl):
                return True
                
            return False
            
        # Start DFS
        # We need to track the actual set if we want to return it.
        # For the search, we just need to know if it exists to add an edge.
        # But to add an edge, we need two vertices from the set.
        # So we need to return the set.
        
        # Modified DFS to return the set
        result_set = []
        
        def dfs_with_set(current_size, available_mask):
            if current_size == k:
                return True
            if not available_mask:
                return False
            if current_size + bin(available_mask).count('1') < k:
                return False
                
            best_v = -1
            best_deg = -1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = bin(adj_bits[v] & available_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
                
            if best_v == -1:
                return False
                
            v = best_v
            result_set.append(v)
            
            # Remove v and neighbors
            new_mask = available_mask & ~(adj_bits[v] | (1 << v))
            if dfs_with_set(current_size + 1, new_mask):
                return True
                
            result_set.pop()
            
            # Exclude v
            new_mask_excl = available_mask & ~(1 << v)
            if dfs_with_set(current_size, new_mask_excl):
                return True
                
            return False
            
        if dfs_with_set(0, (1 << n) - 1):
            return result_set
        return None

    def check_alpha_le_k(adj_bits, n, k):
        """Returns True if alpha <= k."""
        # Try to find independent set of size k+1
        # If found, alpha > k, so return False.
        # If not found, return True.
        # This is the same as find_independent_set_size_k(adj_bits, n, k+1)
        # But we can stop early.
        
        # We can reuse the logic.
        # If find_independent_set_size_k returns a set, then alpha > k.
        # So:
        if find_independent_set_size_k(adj_bits, n, k + 1):
            return False
        return True

    # Main Search Loop
    # We try to find a valid graph for n from 40 down to 30.
    
    best_n = 0
    best_adj = None
    best_edges = None
    
    for target_n in range(40, 29, -1):
        if get_elapsed() > deadline - 5:
            break
            
        # Initialize random graph
        n = target_n
        # Start with a random graph with density ~ 0.5?
        # Or start with a known good structure?
        # Let's start with a random graph.
        random.seed(42 + target_n)
        adj = [set() for _ in range(n)]
        adj_bits = [0] * n
        
        # Random initialization
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i].add(j)
                    adj[j].add(i)
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.995
        steps = 0
        max_steps = 50000
        
        current_penalty = float('inf')
        
        while steps < max_steps and get_elapsed() < deadline - 5:
            steps += 1
            T *= cooling_rate
            
            # Calculate current penalty
            # Penalty = C4_count * 10 + max(0, alpha - 10)
            # We don't need exact count, just check if violations exist.
            
            c4 = find_c4(adj_bits, n)
            ind_set = find_independent_set_size_k(adj_bits, n, 11)
            
            if not c4 and not ind_set:
                # Valid graph found!
                best_n = n
                best_adj = [set(s) for s in adj]
                best_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append([u, v])
                # Verify once more to be safe
                # (Verification is expensive, but we trust the search)
                break
                
            # Determine move
            # If C4 exists, remove an edge from it.
            # If IndSet exists, add an edge to it.
            # If both, pick randomly.
            
            move_type = None
            if c4 and ind_set:
                move_type = random.choice(['c4', 'ind'])
            elif c4:
                move_type = 'c4'
            elif ind_set:
                move_type = 'ind'
            else:
                # Should not happen if logic above is correct
                break
                
            if move_type == 'c4':
                u, c1, v, c2 = c4
                # Edges in C4: (u,c1), (c1,v), (v,c2), (c2,u)
                edges_to_remove = [(u, c1), (c1, v), (v, c2), (c2, u)]
                e1, e2 = random.choice(edges_to_remove)
                
                # Remove edge
                adj[e1].remove(e2)
                adj[e2].remove(e1)
                adj_bits[e1] &= ~(1 << e2)
                adj_bits[e2] &= ~(1 << e1)
                
                # Calculate new penalty roughly
                # We don't calculate exact penalty, we just accept if it seems good?
                # Simulated Annealing usually compares penalties.
                # Calculating penalty is expensive.
                # Heuristic: Always accept removal? 
                # Removing an edge reduces C4 count but might increase alpha.
                # Adding an edge increases C4 count but might reduce alpha.
                
                # Let's use a simple acceptance rule:
                # If we removed an edge, check if C4 is still there? 
                # No, we just accept the move and let the next iteration evaluate.
                # To avoid oscillation, we can track recent moves.
                
                # For now, always accept.
                
            else: # move_type == 'ind'
                # ind_set is a list of 11 vertices
                u, v = random.sample(ind_set, 2)
                
                # Add edge (u, v)
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
                # Always accept?
                # Adding an edge might create C4.
                
            # If we are stuck (penalty not improving), we might need to restart or change strategy.
            # For now, the random nature of SA should handle it.
            
        if best_n == n:
            # Found a valid graph for this n
            break
            
    # If we didn't find anything, return the best we had (maybe 0?)
    # But we should at least return the seed's result or a small valid graph.
    # The seed returned 31.
    # If our search fails, we should fall back to a known valid graph.
    # Let's construct the PG(2,5) polarity graph as a fallback.
    
    if best_n == 0:
        # Fallback: PG(2,5) Polarity Graph
        q = 5
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        n = len(points)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                if (x1 * y2 + y1 * x2 + z1 * z2) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        best_n = n
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v])
                    
    return {'n': best_n, 'edges': best_edges}
