import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    def is_valid(n, edges):
        """
        Check if a graph has no regular induced subgraph on 6+ vertices.
        Uses bitmasks for speed.
        """
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # We only need to check subsets of size 6.
        # If a subset of size k > 6 is regular, it contains a regular subset of size 6?
        # Not necessarily. A regular graph of size 7 (e.g. 2-regular cycle C7) 
        # does NOT contain a regular induced subgraph of size 6 (C6 is 2-reg, but C7 has no C6 induced).
        # Wait, the condition is "No set of 6 or more vertices may induce a regular subgraph".
        # This means we must check ALL k >= 6.
        # However, if a set S of size k is regular, does it imply a subset of size 6 is regular?
        # No. C7 is regular (degree 2) but no 6-subset of C7 is regular (degrees in induced subgraph of 6 vertices from C7: one vertex has degree 1, others 2).
        # So we MUST check all k >= 6.
        
        # But wait, for n=20, checking all k is expensive.
        # Is there a property?
        # If G[S] is regular, then every vertex in S has degree d in G[S].
        # If we remove a vertex v from S, the degrees change.
        
        # Let's stick to checking all k >= 6.
        # Optimization: If a subset of size k is regular, its complement in the whole graph might not be, 
        # but we are checking subsets of the vertex set.
        
        # Actually, for n=20, sum_{k=6}^{20} C(20,k) is ~1.8M.
        # Checking each takes O(k) bit operations. 1.8M * 10 ~ 18M ops. Python can do this in < 1s? 
        # Maybe 2-3s.
        
        # Let's try to optimize the check.
        # We can iterate k from 6 to n.
        # For each subset, compute degrees.
        
        # A faster check for "is regular":
        # Compute degrees of all vertices in the subset.
        # If min_deg == max_deg, it is regular.
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if all vertices in subset have the same degree within the subset
                verts = list(subset)
                d0 = (adj[verts[0]] & mask).bit_count()
                regular = True
                for v in verts[1:]:
                    d = (adj[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True

    def count_violations(adj, n):
        """
        Count the number of regular induced subgraphs of size exactly 6.
        This is a proxy for the number of violations for k>=6?
        No, a violation of size 7 is not counted here.
        However, graphs with few 6-violations are more likely to be valid.
        Also, if we minimize 6-violations to 0, we still need to check 7..n.
        But 6-violations are the most numerous and easiest to find.
        
        Actually, let's count violations for k=6 only for the objective function,
        and do the full check at the end.
        """
        violations = 0
        # Precompute masks for all 6-subsets? No, n is small.
        # We can iterate all 6-subsets.
        # C(20, 6) = 38760. Very fast.
        
        # To speed up, we can use the fact that if a 6-subset is regular, 
        # it contributes to the count.
        
        # Even better: just check if ANY 6-subset is regular.
        # If we find one, we can stop? No, we want a count for gradient-like search.
        # But for local search, minimizing the number of 6-violations is good.
        
        # Let's just return the number of 6-violations.
        
        # Optimization: 
        # A 6-subset is regular if all degrees are equal.
        # Possible degrees: 0, 1, 2, 3, 4, 5.
        
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            verts = list(subset)
            d0 = (adj[verts[0]] & mask).bit_count()
            regular = True
            for v in verts[1:]:
                d = (adj[v] & mask).bit_count()
                if d != d0:
                    regular = False
                    break
            if regular:
                violations += 1
                
        return violations

    def local_search(n, max_iters, time_limit):
        """
        Start with a random graph and flip edges to reduce violations.
        """
        # Initialize random graph
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        current_violations = count_violations(adj, n)
        
        # Local search
        for i in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Check if edge exists
            is_edge = (adj[u] >> v) & 1
            
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_violations = count_violations(adj, n)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if is_edge:
                    edges.remove([u, v]) if [u, v] in edges else edges.remove([v, u])
                else:
                    edges.append([u, v])
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                # If stuck, try a random restart or just continue
                # To escape local minima, we can accept worse solutions with some probability (Simulated Annealing)
                # But for simplicity, let's just do random restarts if stuck for a while.
                
        # Final check
        if is_valid(n, edges):
            return {'n': n, 'edges': edges}
        return None

    # Try local search for n=21 first, then 20
    for n in [21, 20]:
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Use a different seed for each attempt
            random.seed(attempt * 1000 + n)
            
            result = local_search(n, max_iters=5000, time_limit=deadline)
            if result:
                return result

    # Fallback: return a known valid small graph or random 5-vertex graph
    return {'n': 5, 'edges': []}
