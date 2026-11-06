import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # Fallback solution from seed (n=17)
    squares = {x * x % 17 for x in range(1, 17)}
    fallback_colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    best_result = {'n': 17, 'colors': fallback_colors}
    best_score = 17  # Maximize n

    def count_k222(adj, n):
        """
        Count the number of K2,2,2 subgraphs in the graph defined by adj bitmasks.
        adj[u] is a bitmask of neighbors of u.
        """
        count = 0
        # Iterate over all pairs (u, v) as the first part A
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Common neighbors of u and v, excluding u and v
                # adj[u] & adj[v] gives common neighbors.
                # We must remove u and v from the set.
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # Iterate over all pairs (w, x) in common as the second part B
                # We need to iterate over pairs in the set 'common'
                # Extract indices
                cw = common
                while cw:
                    # Get lowest set bit index
                    # w = (cw & -cw).bit_length() - 1
                    w_bit = cw & -cw
                    w = w_bit.bit_length() - 1
                    cw ^= w_bit
                    
                    # Now iterate over x in common, x > w
                    cx = common & ~((1 << (w + 1)) - 1) # All bits higher than w
                    while cx:
                        x_bit = cx & -cx
                        x = x_bit.bit_length() - 1
                        cx ^= x_bit
                        
                        # For fixed A={u,v} and B={w,x}, find C={y,z}
                        # C must be subset of (N(w) & N(x)) & (N(u) & N(v))
                        # And C must not contain u, v, w, x
                        s = (adj[w] & adj[x]) & common
                        s &= ~(1 << u)
                        s &= ~(1 << v)
                        s &= ~(1 << w)
                        s &= ~(1 << x)
                        
                        # If |s| >= 2, we can form at least one pair (y,z)
                        # The number of ways to choose 2 from s is s_count * (s_count - 1) // 2
                        s_count = s.bit_count()
                        if s_count >= 2:
                            count += s_count * (s_count - 1) // 2
        
        return count

    def is_valid(adj, n):
        return count_k222(adj, n) == 0 and count_k222([((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)], n) == 0

    def get_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(1 if (adj[i] >> j) & 1 else 0)
        return colors

    # Try to find a solution for n=29, 28, ... down to 18
    for n in range(29, 17, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                mask = 0
                for j in range(n):
                    if i != j:
                        if random.random() < 0.5:
                            mask |= (1 << j)
                adj[i] = mask
            
            # Local Search: Hill Climbing
            # Objective: Minimize K222(G) + K222(complement)
            # We will just check validity at the end of some steps to save time,
            # or use a fast approximate score.
            # Actually, count_k222 is the score.
            
            current_score = count_k222(adj, n) + count_k222([((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)], n)
            
            # Simple Hill Climbing
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try flipping each edge
                for i in range(n):
                    for j in range(i + 1, n):
                        if time.time() > deadline:
                            break
                        # Flip edge (i, j)
                        adj[i] ^= (1 << j)
                        adj[j] ^= (1 << i)
                        
                        new_score = count_k222(adj, n) + count_k222([((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)], n)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if current_score == 0:
                                # Found valid solution!
                                colors = get_colors(adj, n)
                                best_result = {'n': n, 'colors': colors}
                                best_score = n
                                found = True
                                break
                        else:
                            # Revert
                            adj[i] ^= (1 << j)
                            adj[j] ^= (1 << i)
                    if found:
                        break
                if found:
                    break
            
            if found:
                break
        
        if found:
            break

    return best_result
