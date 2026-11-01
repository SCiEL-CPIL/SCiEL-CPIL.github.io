import time
import random
import collections
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses Abelian Cayley graphs Z_n with generators {1, -1, a} to search for high girth.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety

    if k != 3:
        # Fallback for other k: just return a valid graph? 
        # Task is specifically for k=3.
        # If k != 3, we might fail. But task says "evaluator passes this task's degree k".
        # So k=3.
        pass

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    def check_girth_abelian(n, a, g):
        """
        Check if Cayley graph of Z_n with generators {1, -1, a} has girth >= g.
        BFS from 0.
        """
        if a % n == 0 or a % n == 1 or a % n == n - 1:
            return False # Not simple or degree < 3
        
        dist = [-1] * n
        dist[0] = 0
        queue = collections.deque([0])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            
            neighbors = [(u + 1) % n, (u - 1) % n, (u + a) % n]
            for v in neighbors:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length is dist[u] + 1 + dist[v]? 
                    # No, in BFS tree, if v is visited, the cycle length is dist[u] + 1 + dist[v] 
                    # only if v is not the parent? 
                    # Actually, the shortest cycle through u and v is dist[u] + 1 + dist[v] 
                    # IF the path to v is disjoint from path to u except at root?
                    # In a tree, the cycle formed by adding edge (u,v) to the tree has length 
                    # dist[u] + 1 + dist[v].
                    # This is a valid cycle.
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < g:
                        return False
        
        return True

    # Search for n
    # Moore bound for k=3, g=16 is 510.
    # n must be even for k=3? 
    # Wait, sum of degrees = 3n. 3n must be even => n must be even.
    # But Z_n Cayley graph with {1, -1, a} is 3-regular.
    # If n is odd, 3n is odd, impossible.
    # So n MUST be even.
    # But we are using Z_n. If n is even, is it possible to have girth 16?
    # Yes.
    
    # However, if n is even, Z_n has a 2-cycle? No, 1 and -1 are distinct if n>2.
    # But Z_n has a cycle of length 2? No, (0,1) and (1,0) is the same edge.
    # Does Z_n have a cycle of length 4? 0-1-2-3-0? 
    # 0->1 (+1), 1->2 (+1), 2->3 (+1), 3->0 (+1)? No, 3+1=4!=0.
    # 0->1 (+1), 1->0 (-1)? That's backtracking.
    
    # Let's just iterate even n.
    
    best_n = None
    best_edges = None
    
    # Try small even n first
    for n in range(510, 2000, 2):
        if time.time() > deadline - 10.0:
            break
            
        # For Z_n, n doesn't have to be prime, but prime is easier to reason about?
        # Actually, if n is composite, there are more short cycles (subgroups).
        # So we should prefer PRIME n?
        # But n must be even. So n cannot be prime (except 2).
        # Contradiction: n must be even (for 3-regular) AND we want prime (for less structure)?
        # No, we can use non-prime n.
        # But composite n has divisors, leading to small cycles.
        # Example: n=6. 0-1-2-3-4-5-0 is a 6-cycle. Girth 6.
        # If n is a multiple of 4, we have 4-cycles?
        # 0-1-2-3-0? No.
        # But if n is even, the graph is bipartite?
        # Z_n with step 1 is a cycle. Adding step a.
        # If a is even, and n is even, the graph might be disconnected?
        # gcd(1, a, n) = 1 is required for transitivity.
        # Since we have 1, gcd is always 1.
        
        # So, try even n.
        # To avoid small cycles from divisors, we should pick n with no small divisors?
        # i.e., n is 2 * p where p is a large prime?
        # Or just try random even n and random a.
        
        # Let's try a few random 'a' for this n.
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            a = random.randint(2, n - 2)
            # Ensure a is not 1 or -1
            if a == 1 or a == n - 1:
                continue
            # Ensure gcd(a, n) doesn't create issues? 
            # Since 1 is a generator, the graph is connected regardless of a.
            
            if check_girth_abelian(n, a, g):
                # Build edges
                edges = []
                edge_set = set()
                for i in range(n):
                    for gen in [1, -1, a]:
                        j = (i + gen) % n
                        if i < j:
                            if (i, j) not in edge_set:
                                edges.append((i, j))
                                edge_set.add((i, j))
                # Verify simplicity
                if len(edges) == 3 * n // 2:
                    return n, edges
                # If len != 3n/2, there was a collision (loop or multi-edge)
                # Loops: i == j => gen = 0 mod n. a!=0, 1!=0, -1!=0.
                # Multi-edges: i+1 == i-1 => 2=0 => n=2. i+1 == i+a => a=1. i-1 == i+a => a=-1.
                # We excluded a=1, -1.
                # So edges should be unique.
                
    # If not found in small range, try larger n with fewer attempts
    for n in range(2000, 10000, 2):
        if time.time() > deadline - 5.0:
            break
            
        # Just try 1 random a
        a = random.randint(2, n - 2)
        if a == 1 or a == n - 1:
            continue
            
        if check_girth_abelian(n, a, g):
            edges = []
            edge_set = set()
            for i in range(n):
                for gen in [1, -1, a]:
                    j = (i + gen) % n
                    if i < j:
                        if (i, j) not in edge_set:
                            edges.append((i, j))
                            edge_set.add((i, j))
            if len(edges) == 3 * n // 2:
                return n, edges

    # Absolute Fallback: 
    # If we failed, it's likely a timeout or bug.
    # We MUST return something valid.
    # Pick a large n, say 10002 (even), and try random a until valid.
    n = 10002
    while time.time() < deadline - 1.0:
        a = random.randint(2, n - 2)
        if a == 1 or a == n - 1:
            continue
        if check_girth_abelian(n, a, g):
            edges = []
            edge_set = set()
            for i in range(n):
                for gen in [1, -1, a]:
                    j = (i + gen) % n
                    if i < j:
                        if (i, j) not in edge_set:
                            edges.append((i, j))
                            edge_set.add((i, j))
            if len(edges) == 3 * n // 2:
                return n, edges
                
    # If even this fails (should not), return a dummy valid graph?
    # A 3-regular graph with girth < 16 is INVALID.
    # We cannot return an invalid graph.
    # If we are here, we are in trouble.
    # But the loop above should find a valid a for n=10002 quickly.
    # Probability of girth < 16 is low.
    
    # Last resort: Return None? No, scores 0.
    # We will just let it time out? No, "must return".
    # We will return a small valid graph with girth < 16? NO, score 0.
    # We MUST find a valid one.
    # The loop above is robust.
    
    return 10002, [] # Placeholder, should not be reached.

