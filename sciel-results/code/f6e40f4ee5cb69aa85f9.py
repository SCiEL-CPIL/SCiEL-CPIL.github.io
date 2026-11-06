import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 285
    
    def is_clique(adj, subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def is_independent(adj, subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                if (adj[u] >> v) & 1:
                    return False
        return True

    def count_offending_fast(adj, n, size=5):
        """Count K5 and independent 5-sets using bit operations where possible."""
        count = 0
        # Precompute neighbor masks
        for comb in combinations(range(n), size):
            # Check clique
            clique = True
            indep = True
            for i in range(size):
                for j in range(i+1, size):
                    u, v = comb[i], comb[j]
                    bit = (adj[u] >> v) & 1
                    if not bit:
                        clique = False
                    else:
                        indep = False
                    if not clique and not indep:
                        break
                if not clique and not indep:
                    break
            if clique:
                count += 1
            if indep:
                count += 1
        return count

    def find_offending_set_fast(adj, n, size=5):
        """Find any K5 or independent 5-set. Returns the set or None."""
        for comb in combinations(range(n), size):
            if is_clique(adj, comb):
                return list(comb)
            if is_independent(adj, comb):
                return list(comb)
        return None

    def verify_graph(adj, n, size=5):
        """Full verification that no K5 or I5 exists."""
        for comb in combinations(range(n), size):
            if is_clique(adj, comb) or is_independent(adj, comb):
                return False
        return True

    def local_search(adj, n, max_iters, deadline):
        """Local search to eliminate offending sets by flipping edges."""
        best_adj = adj[:]
        best_count = count_offending_fast(adj, n)
        
        if best_count == 0:
            return best_adj, 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find an offending set
            offending = find_offending_set_fast(adj, n)
            if offending is None:
                # Verify fully
                if verify_graph(adj, n):
                    return adj, 0
                # If not valid, continue searching
            
            # Try flipping edges in the offending set (or random if none found)
            if offending:
                candidates = list(combinations(offending, 2))
                random.shuffle(candidates)
            else:
                candidates = [(random.randrange(n), random.randrange(n))]
                if candidates[0][0] == candidates[0][1]:
                    candidates[0] = (candidates[0][0], (candidates[0][0]+1) % n)
                if candidates[0][0] > candidates[0][1]:
                    candidates[0] = (candidates[0][1], candidates[0][0])
            
            best_adj_in_iter = adj
            best_count_in_iter = best_count
            
            for u, v in candidates:
                if u == v:
                    continue
                # Flip edge (u, v)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Count offending sets after flip
                new_count = count_offending_fast(adj, n)
                
                if new_count < best_count_in_iter:
                    best_count_in_iter = new_count
                    best_adj_in_iter = adj[:]
                    if new_count == 0:
                        # Unflip and return
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        return best_adj_in_iter, 0
                else:
                    # Unflip
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Update best
            if best_count_in_iter < best_count:
                best_count = best_count_in_iter
                best_adj = best_adj_in_iter
                adj = best_adj_in_iter[:]
        
        return best_adj, best_count

    def try_add_vertex(adj, n, deadline):
        """Try to add a new vertex to the graph while maintaining (5,5)-freeness."""
        new_n = n + 1
        # Try different connection patterns for the new vertex
        # The new vertex is index n
        best_adj = None
        best_count = float('inf')
        
        # Try all 2^n possible connection patterns? That's too many for n>20.
        # Instead, use local search to find a good connection pattern.
        # Start with a random connection pattern.
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            new_adj = adj[:] + [0]
            # Connect new vertex randomly
            for i in range(n):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << n)
                    new_adj[n] |= (1 << i)
            
            # Local search on the new vertex's connections
            # We only flip edges incident to the new vertex
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Check if current is valid
                if verify_graph(new_adj, new_n):
                    return new_adj
                
                # Find an offending set
                offending = find_offending_set_fast(new_adj, new_n)
                
                if offending is not None and n in offending:
                    # Flip an edge from new vertex to a vertex in the offending set
                    others = [x for x in offending if x != n]
                    if others:
                        target = random.choice(others)
                        # Flip edge (n, target)
                        if (new_adj[n] >> target) & 1:
                            new_adj[n] &= ~(1 << target)
                            new_adj[target] &= ~(1 << n)
                        else:
                            new_adj[n] |= (1 << target)
                            new_adj[target] |= (1 << n)
                elif offending is not None:
                    # Offending set doesn't include new vertex, flip random edge in it
                    u, v = random.sample(offending, 2)
                    if (new_adj[u] >> v) & 1:
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                else:
                    # No offending set found, flip random edge incident to new vertex
                    target = random.randrange(n)
                    if (new_adj[n] >> target) & 1:
                        new_adj[n] &= ~(1 << target)
                        new_adj[target] &= ~(1 << n)
                    else:
                        new_adj[n] |= (1 << target)
                        new_adj[target] |= (1 << n)
            
            # Check final state
            if verify_graph(new_adj, new_n):
                return new_adj
        
        return None

    # Start from Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    current_adj = adj
    
    # Try to add vertices one at a time up to 42
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 30:
            break
        
        # Try multiple times to add a vertex
        success = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            new_adj = try_add_vertex(current_adj, current_n, deadline)
            if new_adj is not None:
                current_adj = new_adj
                current_n = target_n
                success = True
                break
        
        if not success:
            break
    
    # Final verification
    if verify_graph(current_adj, current_n):
        edges = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                if (current_adj[i] >> j) & 1:
                    edges.append((i, j))
        return current_n, edges
    
    # Fallback to Paley graph on 37
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
