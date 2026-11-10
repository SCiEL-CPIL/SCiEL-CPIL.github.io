import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Strategy:
    1. For k=4, g=10, attempt to find the 320-vertex cage using randomized local search.
    2. If that fails within time, try larger n (up to a limit) with the same search.
    3. Fall back to the Lazebnik-Ustimenko-Woldar construction which is guaranteed to be valid.
    """
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin

    def elapsed():
        return time.time() - t0

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        def generate_random_4_reg_bipartite(n):
            """
            Generates a simple 4-regular bipartite graph on n vertices (n/2 in each part).
            Uses configuration model with multi-edge removal.
            """
            if n % 2 != 0:
                return None
            
            half_n = n // 2
            # Left: 0..half_n-1, Right: half_n..n-1
            adj = [set() for _ in range(n)]
            
            # Stubs for Left and Right
            # Each vertex has 4 stubs
            left_stubs = [i for i in range(half_n) for _ in range(4)]
            right_stubs = [half_n + i for i in range(half_n) for _ in range(4)]
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            # Pair them up
            # This creates a multigraph
            for l, r in zip(left_stubs, right_stubs):
                adj[l].add(r)
                adj[r].add(l)
            
            # Remove multi-edges by swapping
            # We need to ensure each vertex has exactly 4 distinct neighbors.
            # If adj[v] has size < 4, there are duplicates.
            # We can fix this by finding two vertices u, v with duplicate edges and swapping.
            
            # A simpler robust method: 
            # 1. Create list of edges (l, r)
            # 2. While any vertex has degree > 4 (impossible in simple graph, but here we have sets so degree is unique count)
            #    Wait, the set `adj` automatically removes duplicates. 
            #    So if we just add to sets, we might end up with vertices having degree < 4.
            #    We need to ensure the final graph is 4-regular.
            
            # Let's restart with a different generation method:
            # Random permutation of Right vertices for each Left vertex? No.
            
            # Method:
            # 1. Start with empty adj lists.
            # 2. For each Left vertex, pick 4 distinct Right vertices.
            # 3. This doesn't guarantee Right degrees.
            
            # Better: Use a library or robust algorithm? No, stdlib only.
            # Let's use the "swap to fix" approach on the set-based graph.
            
            # Check degrees
            def is_valid(adj):
                for v in range(n):
                    if len(adj[v]) != 4:
                        return False
                return True
            
            # If not valid, we need to fix it.
            # Actually, the zip method creates a multigraph. 
            # If we convert to sets, we lose the "multiplicity" needed to maintain degree 4 if collisions happen.
            # Example: L0 connects to R0, R0, R1, R2. Set is {R0, R1, R2}. Degree 3.
            # We need to replace one of the "lost" edges.
            
            # Alternative: Generate a random 4-regular bipartite graph using a known algorithm.
            # Algorithm:
            # 1. Create list L of 4 copies of each left vertex.
            # 2. Create list R of 4 copies of each right vertex.
            # 3. Shuffle R.
            # 4. Pair L[i] with R[i].
            # 5. This gives a multigraph.
            # 6. While there is a multi-edge (u, v) with multiplicity > 1:
            #    Find another edge (u', v') such that (u, v') and (u', v) are not edges.
            #    Swap: remove (u, v) and (u', v'), add (u, v') and (u', v).
            
            # Let's implement this.
            adj = [set() for _ in range(n)]
            # Track multiplicities? Or just use lists and sort/check?
            # Using sets is faster for lookup.
            
            # We need to store the "stubs" as a list of pairs to allow swapping.
            # Pairs: (l, r)
            pairs = list(zip(left_stubs, right_stubs))
            
            # Build initial multigraph adjacency (using lists to track multiplicity)
            adj_list = [[] for _ in range(n)]
            for l, r in pairs:
                adj_list[l].append(r)
                adj_list[r].append(l)
                
            # Function to check for multi-edges and fix
            # A multi-edge exists if len(adj_list[v]) > 4? No, degree is always 4 in multigraph.
            # Multi-edge means u appears in adj_list[v] more than once.
            
            def has_multi_edges(adj_list):
                for v in range(n):
                    if len(adj_list[v]) != len(set(adj_list[v])):
                        return True
                return False
            
            # We need to swap edges to remove multi-edges.
            # Pick a vertex v with a duplicate neighbor u.
            # Pick another vertex v' and its neighbor u'.
            # Swap (v, u) and (v', u') to (v, u') and (v', u).
            
            max_iters = 1000
            for _ in range(max_iters):
                if not has_multi_edges(adj_list):
                    break
                
                # Find a vertex with duplicate
                v = -1
                for i in range(n):
                    if len(adj_list[i]) != len(set(adj_list[i])):
                        v = i
                        break
                if v == -1:
                    break
                
                # Find a duplicate neighbor u
                # Count occurrences
                counts = {}
                for u in adj_list[v]:
                    counts[u] = counts.get(u, 0) + 1
                u = [key for key, val in counts.items() if val > 1][0]
                
                # Find a vertex v' != v that is connected to some u'
                # We want to swap (v, u) with (v', u')
                # Conditions:
                # 1. u' != u
                # 2. (v, u') is not an edge (or is a multi-edge we are breaking? No, we want simple graph)
                # 3. (v', u) is not an edge
                # 4. v' != u and u' != v (bipartite)
                
                # Try random v'
                for _ in range(100):
                    v_prime = random.randint(0, n - 1)
                    if v_prime == v:
                        continue
                    if v_prime < half_n and v < half_n: # Both left? No, bipartite
                        continue
                    if v_prime >= half_n and v >= half_n:
                        continue
                    
                    # Pick a neighbor u' of v_prime
                    # We need u' to be in the other part
                    if v_prime < half_n:
                        # v' is Left, neighbors are Right
                        u_prime_candidates = [x for x in adj_list[v_prime] if x >= half_n]
                    else:
                        u_prime_candidates = [x for x in adj_list[v_prime] if x < half_n]
                        
                    if not u_prime_candidates:
                        continue
                        
                    u_prime = random.choice(u_prime_candidates)
                    
                    # Check if swap is valid
                    # Remove (v, u) and (v', u')
                    # Add (v, u') and (v', u)
                    
                    # (v, u') must not exist in adj_list[v] (except if we are swapping the duplicate?)
                    # Actually, we want to break the duplicate (v, u).
                    # If (v, u') already exists, we create another multi-edge or keep it.
                    # We require (v, u') NOT in adj_list[v] and (v', u) NOT in adj_list[v']
                    
                    if u_prime in adj_list[v]:
                        continue
                    if u in adj_list[v_prime]:
                        continue
                    
                    # Perform swap
                    adj_list[v].remove(u)
                    adj_list[v].append(u_prime)
                    adj_list[v_prime].remove(u_prime)
                    adj_list[v_prime].append(u)
                    
                    # Update the reverse adjacencies
                    # Note: adj_list is symmetric.
                    # We removed u from v's list. We must remove v from u's list.
                    adj_list[u].remove(v)
                    adj_list[u].append(v_prime)
                    
                    # We removed u_prime from v_prime's list. We must remove v_prime from u_prime's list.
                    adj_list[u_prime].remove(v_prime)
                    adj_list[u_prime].append(v)
                    
                    break # Restart search for multi-edges
            
            if not has_multi_edges(adj_list):
                return adj_list
            return None

        def get_girth(adj_list):
            """
            Returns the length of the shortest cycle.
            """
            n_vertices = len(adj_list)
            min_girth = 100
            
            for start in range(n_vertices):
                dist = [-1] * n_vertices
                parent = [-1] * n_vertices
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    for v in adj_list[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            # Found a cycle
                            # Length = dist[u] + dist[v] + 1
                            # This is a cycle in the BFS tree?
                            # Actually, if u and v are both visited, and they are adjacent,
                            # and neither is the parent of the other, we have a cycle.
                            # The length is dist[u] + dist[v] + 1?
                            # No. The path from start to u is dist[u].
                            # The path from start to v is dist[v].
                            # The edge (u, v) closes the loop.
                            # The cycle length is dist[u] + dist[v] + 1?
                            # Only if the paths from start to u and start to v are disjoint except for start.
                            # In BFS, they might share vertices.
                            # However, the *first* time we encounter such an edge, it gives a cycle.
                            # To be safe, we can just check if dist[u] + dist[v] + 1 < min_girth.
                            # This might overestimate if paths overlap, but it's a lower bound on the girth?
                            # No, it's the length of a specific cycle.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth <= 4: # Can't get better than 4 in bipartite? No, 4 is min.
                                    return min_girth
            return min_girth

        def try_local_search(n, time_limit):
            """
            Attempts to construct a 4-regular bipartite graph on n vertices with girth >= 10.
            """
            # Generate initial graph
            adj = generate_random_4_reg_bipartite(n)
            if adj is None:
                return None
            
            current_girth = get_girth(adj)
            
            if current_girth >= 10:
                # Success
                edges = []
                for u in range(n // 2):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

            # Local search
            while time.time() - t0 < time_limit:
                # Find a short cycle
                # We can just check girth. If < 10, we need to break a cycle.
                # To be efficient, we can try random swaps and see if girth increases.
                
                # Pick two random edges (u, v) and (u', v')
                # u, u' in Left, v, v' in Right
                u = random.randint(0, n // 2 - 1)
                if not adj[u]:
                    continue
                v = random.choice(adj[u])
                
                u_prime = random.randint(0, n // 2 - 1)
                if u_prime == u:
                    continue
                if not adj[u_prime]:
                    continue
                v_prime = random.choice(adj[u_prime])
                
                # Check validity of swap (u, v), (u', v') -> (u, v'), (u', v)
                if v_prime in adj[u]:
                    continue
                if v in adj[u_prime]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[u].append(v_prime)
                adj[v].remove(u)
                adj[v].append(u_prime)
                
                adj[u_prime].remove(v_prime)
                adj[u_prime].append(v)
                adj[v_prime].remove(u_prime)
                adj[v_prime].append(u)
                
                new_girth = get_girth(adj)
                
                if new_girth > current_girth:
                    current_girth = new_girth
                    if current_girth >= 10:
                        edges = []
                        for u_node in range(n // 2):
                            for v_node in adj[u_node]:
                                if u_node < v_node:
                                    edges.append((u_node, v_node))
                        return n, edges
                else:
                    # Revert
                    adj[u].remove(v_prime)
                    adj[u].append(v)
                    adj[v].remove(u_prime)
                    adj[v].append(u)
                    
                    adj[u_prime].remove(v)
                    adj[u_prime].append(v_prime)
                    adj[v_prime].remove(u)
                    adj[v_prime].append(u_prime)

            return None

        # Try specific n values
        # 320 is the cage.
        # If we fail, try larger n where it's easier to find a high girth graph?
        # Actually, for fixed k, random graphs have girth ~ log_k n.
        # For k=4, log_4(320) ~ 4.5. So random 320-vertex graph has girth ~ 8-10.
        # It's close.
        
        n_candidates = [320, 324, 328, 332, 336, 340, 350, 400, 500]
        
        for n in n_candidates:
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Allocate time
            # Give 320 more time
            if n == 320:
                limit = min(60, DEADLINE - (time.time() - t0) - 5)
            else:
                limit = min(20, DEADLINE - (time.time() - t0) - 5)
                
            if limit < 1:
                continue
                
            result = try_local_search(n, limit)
            if result:
                return result

    # Fallback: L-U-W Construction
    q = k
    if g % 2 == 1:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = max(4, g - 4)
        if d % 2 == 1:
            d += 1
    
    if q not in [2, 3, 4, 5, 7, 11, 13]:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        q = 4
        while not (is_prime(q) or q == 4):
            q += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
