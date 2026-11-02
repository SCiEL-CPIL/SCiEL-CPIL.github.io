import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Fallback: LUW construction
    # d = ceil((g-2)/2)
    d = (g - 2 + 1) // 2
    if d < 1: d = 1
    
    # If k is not prime, LUW might not work directly or needs GF(q) handling
    # The seed code handled q=4 and primes.
    # For k=5, q=5 is prime.
    # For general k, we might need a different fallback or just use LUW if k is prime/4.
    # If k is not suitable for LUW, we rely on random search.
    
    def get_luw_graph(k, g):
        q = k
        d = (g - 2 + 1) // 2
        if d < 1: d = 1
        
        # Check if q is valid for LUW (prime or 4)
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
            
        if not (is_prime(q) or q == 4):
            return None
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try Random Local Search for smaller n
    # Target n: start from Moore bound + margin
    moore_bound = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    best_graph = None
    best_n = float('inf')
    
    # Fallback LUW
    luw_res = get_luw_graph(k, g)
    if luw_res:
        best_n, best_edges = luw_res
        best_graph = (best_n, best_edges)
    else:
        # If LUW not available, we must find something
        # Try random search with larger n
        pass

    # Search for better graph
    # We try to find a graph with n < best_n
    # Start from a reasonable size
    min_n_to_try = max(moore_bound + 10, 100)
    max_n_to_try = best_n - 10 if best_n < float('inf') else 2000
    
    # If best_n is inf, we need to find any valid graph
    if best_n == float('inf'):
        max_n_to_try = 5000
        min_n_to_try = 100

    # Search strategy:
    # Try sizes in increasing order? Or random?
    # Increasing order is better to find the smallest.
    # But we have a time limit.
    # Let's try a few specific sizes.
    
    candidate_sizes = []
    if best_n < float('inf'):
        # Try sizes between Moore bound and best_n
        step = max(10, (best_n - min_n_to_try) // 20)
        for n in range(min_n_to_try, best_n, step):
            candidate_sizes.append(n)
    else:
        for n in range(min_n_to_try, max_n_to_try, 100):
            candidate_sizes.append(n)
            
    # Shuffle to avoid getting stuck in one size? No, increasing is logical.
    
    for n in candidate_sizes:
        if time.time() > deadline - 5:
            break
            
        # Generate random bipartite k-regular graph of size n
        # n must be even for bipartite
        if n % 2 != 0:
            n += 1
            if n >= best_n: continue
            
        # Attempt to construct
        graph = try_random_bipartite(k, g, n, time.time(), deadline - 5)
        if graph is not None:
            g_n, g_edges = graph
            if g_n < best_n:
                best_n = g_n
                best_graph = (g_n, g_edges)
                # If we found one close to Moore bound, we can stop or continue
                if best_n <= moore_bound + 20:
                    break

    if best_graph is None:
        # Last resort: return a simple ring or something? No, must be k-regular.
        # If we couldn't find anything, this is bad.
        # But LUW should have worked for k=5.
        # If k is not prime, LUW fails.
        # In that case, we must have found something in search.
        # If search failed, we return a dummy? No, score 0.
        # We assume k is prime or 4 for this task.
        return (0, [])
        
    return best_graph

def try_random_bipartite(k, g, n, start_t, deadline):
    # n is total vertices, n/2 in each part
    if n % 2 != 0: return None
    half = n // 2
    if half * k < 0: return None
    
    # Generate random k-regular bipartite graph
    # Using random permutations
    # Left nodes 0..half-1, Right nodes half..n-1
    
    # Create list of right node indices repeated k times
    right_nodes = list(range(half, n)) * k
    random.shuffle(right_nodes)
    
    # Assign to left nodes
    adj = {i: set() for i in range(n)}
    for i in range(half):
        for j in range(k):
            r_node = right_nodes[i * k + j]
            adj[i].add(r_node)
            adj[r_node].add(i)
            
    # Check for multiple edges (should not happen with shuffle if we ensure distinct?)
    # Wait, the shuffle ensures the list has k copies of each right node.
    # But a left node might get the same right node twice?
    # No, we assign k distinct slots. But the values in right_nodes are not unique per left node.
    # We need to ensure that for each left node i, the k assigned right nodes are distinct.
    # The simple shuffle doesn't guarantee this.
    # We need a random regular bipartite graph generation that ensures simplicity.
    
    # Better: Use a random permutation matrix sum.
    # Or use the "Configuration Model" and check for loops/multi-edges.
    
    # Let's use a simpler method: Randomly generate, then check.
    # If invalid, retry.
    
    for _ in range(100): # 100 restarts for this n
        if time.time() > deadline: return None
        
        right_nodes = list(range(half, n)) * k
        random.shuffle(right_nodes)
        
        adj = {i: set() for i in range(n)}
        valid = True
        for i in range(half):
            assigned = []
            for j in range(k):
                r_node = right_nodes[i * k + j]
                if r_node in assigned:
                    valid = False
                    break
                assigned.append(r_node)
            if not valid: break
            for r_node in assigned:
                adj[i].add(r_node)
                adj[r_node].add(i)
                
        if not valid: continue
        
        # Check girth
        # BFS from each node to find shortest cycle
        # Since it's bipartite, cycles are even.
        # We need to check for cycles of length 4, 6, ..., g-1 (if g is odd, g-1 is even)
        max_cycle_len_to_check = g - 1
        if max_cycle_len_to_check % 2 != 0:
            max_cycle_len_to_check -= 1
            
        has_short_cycle = False
        for start in range(n):
            if has_short_cycle: break
            # BFS
            visited = {start: 0}
            parent = {start: -1}
            queue = [start]
            while queue:
                curr = queue.pop(0)
                if visited[curr] >= max_cycle_len_to_check // 2:
                    continue
                for nei in adj[curr]:
                    if nei in visited:
                        # Check if it's a cycle
                        # If nei is parent, skip
                        if parent[curr] == nei:
                            continue
                        # If we found a visited node, we have a cycle
                        # Length = visited[curr] + 1 + visited[nei]
                        # Wait, BFS from start.
                        # If nei is visited and not parent, it's a cycle.
                        # The cycle length is (visited[curr] + 1) + (visited[nei] + 1)? No.
                        # Distance from start to curr is d1.
                        # Distance from start to nei is d2.
                        # Edge (curr, nei).
                        # Cycle length is d1 + d2 + 1? No, in bipartite, d1 and d2 have same parity.
                        # Path start->curr (d1), edge curr->nei, path nei->start (d2).
                        # Total d1 + 1 + d2.
                        # Since bipartite, d1 and d2 are both even or both odd?
                        # Start is in Part A.
                        # curr is at dist d1.
                        # nei is at dist d2.
                        # Edge connects A-B.
                        # If curr is A, nei is B.
                        # d1 is even, d2 is odd?
                        # Yes.
                        # Cycle length = d1 + 1 + d2.
                        # This is odd? No, d1+1 is odd, d2 is odd?
                        # Wait.
                        # Start (A).
                        # dist 1: B.
                        # dist 2: A.
                        # If curr is A (dist even), nei is B (dist odd).
                        # Cycle length = even + 1 + odd = even. Correct.
                        
                        cycle_len = visited[curr] + 1 + visited[nei]
                        if cycle_len < g:
                            has_short_cycle = True
                            break
                    else:
                        visited[nei] = visited[curr] + 1
                        parent[nei] = curr
                        queue.append(nei)
                        
        if has_short_cycle:
            # Try to fix?
            # For now, just retry with new random graph
            continue
        else:
            # Valid!
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
    return None
