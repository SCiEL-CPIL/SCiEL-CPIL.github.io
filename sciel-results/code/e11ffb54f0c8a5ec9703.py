import time
import random
import sys

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _check_circulant_girth(n, a, b, target_girth):
    """
    Check if circulant graph C_n(1, a, b) has girth >= target_girth.
    Since it's vertex-transitive, we only need to check cycles through vertex 0.
    We perform a non-backtracking BFS/DFS from 0.
    """
    # Neighbors of 0 are 1, a, n-b (which is -b mod n)
    # Actually, edges are (i, i+1), (i, i+a), (i, i-b).
    # So neighbors of 0 are: 1, a, (0-b)%n.
    
    # We want to find the shortest cycle.
    # A cycle corresponds to a sequence of steps in {1, a, -b} (and their inverses -1, -a, b)
    # that sums to 0 mod n, with no backtracking.
    # Backtracking means we cannot immediately reverse the previous step.
    # Steps:
    # 1: +1 (inverse: -1)
    # 2: +a (inverse: -a)
    # 3: -b (inverse: +b)
    
    # Let's use a BFS on the state (current_node, previous_edge_type).
    # previous_edge_type indicates which edge we came from, to prevent backtracking.
    # Types: 0 (none), 1 (step +1), 2 (step +a), 3 (step -b).
    # Inverses:
    # If we came from +1 (edge from i-1 to i), we cannot go back to i-1 (step -1).
    # If we came from +a, we cannot go to i-a.
    # If we came from -b, we cannot go to i+b.
    
    from collections import deque
    
    # State: (node, prev_type)
    # prev_type: 0 start, 1 arrived via +1, 2 arrived via +a, 3 arrived via -b
    # Distances: dist[node][prev_type]
    
    # We only care about returning to 0.
    # Initialize queue with neighbors of 0.
    
    # Neighbors of 0:
    # 1 (via +1)
    # a (via +a)
    # n-b (via -b)
    
    q = deque()
    # (node, prev_type, dist)
    # We mark visited states (node, prev_type) to avoid loops.
    visited = [[False]*4 for _ in range(n)]
    
    # Start at 0, prev_type 0.
    # We expand from 0.
    
    # 1. From 0, go to 1 via +1.
    if 1 % n == 0: # n=1? No, n>=512
        pass
    q.append((1, 1, 1))
    visited[1][1] = True
    
    # 2. From 0, go to a via +a.
    a_mod = a % n
    if a_mod == 0:
        # Self loop? No, simple graph. a != 0.
        pass
    if not visited[a_mod][2]:
        q.append((a_mod, 2, 1))
        visited[a_mod][2] = True
        
    # 3. From 0, go to n-b via -b.
    b_mod = (-b) % n
    if not visited[b_mod][3]:
        q.append((b_mod, 3, 1))
        visited[b_mod][3] = True

    while q:
        node, prev_type, dist = q.popleft()
        
        # If we return to 0, we found a cycle of length dist + 1?
        # No, dist is the length of the path from 0.
        # If we are at node 0, the cycle length is dist.
        # But we started at 0, took 1 step to neighbor.
        # So if we pop (0, type, dist), the cycle length is dist.
        # Wait, we didn't enqueue (0, 0, 0).
        # We enqueued neighbors with dist=1.
        # If we reach 0 again, the path length is dist.
        # The cycle is 0 -> ... -> 0. Length is dist.
        
        if node == 0:
            # Check if dist >= target_girth
            # Actually, we want to know if ANY cycle is shorter.
            # If we find a return to 0 with dist < target_girth, girth is dist.
            # If we exhaust BFS without returning to 0 with dist < target_girth, 
            # and we have explored all states with dist < target_girth, then girth >= target_girth.
            # However, BFS explores by distance.
            # So the first time we hit 0, it's the shortest cycle.
            return dist >= target_girth
        
        # Expand neighbors
        # Current node is `node`.
        # Possible next steps:
        # 1. +1: next_node = (node + 1) % n. Prev type for next will be 1.
        #    Can we take +1? Only if prev_type != 3 (inverse of -1 is +1? No. Inverse of +1 is -1. Inverse of -b is +b. Inverse of +a is -a.
        #    Let's map types to steps:
        #    Type 1: Step was +1. Inverse is -1. So we cannot take step -1.
        #    Type 2: Step was +a. Inverse is -a. So we cannot take step -a.
        #    Type 3: Step was -b. Inverse is +b. So we cannot take step +b.
        
        # Available steps from `node`:
        # Step +1: next = (node+1)%n. Forbidden if prev_type == 3? No.
        #    If prev_type was 1 (came via +1), we came from node-1. We cannot go back to node-1.
        #    Going +1 goes to node+1. This is fine.
        #    Wait, backtracking means reversing the *edge*.
        #    Edge (u, v) used to arrive at v. We cannot use (v, u) next.
        #    If we arrived at `node` via +1, we came from `node-1`.
        #    So we cannot go to `node-1`.
        #    The step +1 goes to `node+1`. This is not `node-1` (unless n=2).
        #    So +1 is always allowed?
        #    Let's check:
        #    Arrived via +1: prev_node = node-1. Next step +1 goes to node+1. OK.
        #    Next step +a goes to node+a. OK.
        #    Next step -b goes to node-b. OK.
        #    BUT, what if the step we take *is* the inverse?
        #    The inverse of +1 is -1.
        #    Is -1 one of our steps? No, our steps are +1, +a, -b.
        #    So we can only backtrack if one of our steps is the inverse of the previous step.
        
        # Inverses:
        # Inv(+1) = -1. Is -1 in {+1, +a, -b}?
        #   -1 == +1 => 2=0 mod n => n=2.
        #   -1 == +a => a = n-1.
        #   -1 == -b => b = 1.
        
        # Inv(+a) = -a. Is -a in {+1, +a, -b}?
        #   -a == +1 => a = n-1.
        #   -a == +a => 2a=0 mod n => a=n/2.
        #   -a == -b => a=b.
        
        # Inv(-b) = +b. Is +b in {+1, +a, -b}?
        #   +b == +1 => b=1.
        #   +b == +a => a=b.
        #   +b == -b => 2b=0 mod n => b=n/2.
        
        # So, we just need to check if the next step is the inverse of the previous step.
        
        # Step 1: +1
        next1 = (node + 1) % n
        # Inverse of +1 is -1.
        # Did we come via -1? No, we don't have a -1 step.
        # Did we come via a step that IS -1?
        # If prev_type == 2 (came via +a), and a == n-1, then we came via -1. So we cannot take +1.
        # If prev_type == 3 (came via -b), and b == 1, then we came via -1. So we cannot take +1.
        allowed1 = True
        if prev_type == 2 and a == n - 1: allowed1 = False
        if prev_type == 3 and b == 1: allowed1 = False
        
        if allowed1 and not visited[next1][1]:
            visited[next1][1] = True
            q.append((next1, 1, dist + 1))
            
        # Step 2: +a
        next2 = (node + a) % n
        # Inverse of +a is -a.
        # Did we come via -a?
        # If prev_type == 1 (came via +1), and 1 == n-a (i.e., a=n-1), we came via -a.
        # If prev_type == 3 (came via -b), and -b == -a (i.e., a=b), we came via -a.
        allowed2 = True
        if prev_type == 1 and a == n - 1: allowed2 = False
        if prev_type == 3 and a == b: allowed2 = False
        
        if allowed2 and not visited[next2][2]:
            visited[next2][2] = True
            q.append((next2, 2, dist + 1))
            
        # Step 3: -b
        next3 = (node - b) % n
        # Inverse of -b is +b.
        # Did we come via +b?
        # If prev_type == 1 (came via +1), and 1 == b, we came via +b.
        # If prev_type == 2 (came via +a), and a == b, we came via +b.
        allowed3 = True
        if prev_type == 1 and b == 1: allowed3 = False
        if prev_type == 2 and a == b: allowed3 = False
        
        if allowed3 and not visited[next3][3]:
            visited[next3][3] = True
            q.append((next3, 3, dist + 1))
            
    # If we never returned to 0, there is no cycle? 
    # In a connected graph, there must be cycles.
    # But if the BFS exhausted all states without hitting 0, 
    # it means the shortest cycle is longer than the max distance explored?
    # No, BFS explores all reachable states.
    # If 0 is not reached, it means there is no cycle through 0? 
    # That's impossible in a connected graph.
    # Wait, `visited` prevents revisiting states.
    # If the graph is large and girth is large, we might exhaust the queue 
    # before returning to 0 if we limit depth? 
    # We don't limit depth here. We explore until queue is empty.
    # If queue is empty and we didn't return to 0, it implies no cycle?
    # Actually, if the graph is a tree, no cycle. But 3-regular graphs have cycles.
    # The BFS will eventually return to 0.
    # So if we exit the loop, we must have found 0?
    # No, we return inside the loop.
    # If we exit the loop, it means `q` became empty.
    # This can happen if we visited all states and didn't hit 0?
    # That would imply no cycle, which is false.
    # So we should always hit 0.
    
    # Just in case, return False (invalid) if we don't find it?
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find a small circulant graph
    # Moore bound for k=3, g=16 is 510.
    # N must be even.
    
    if k != 3:
        # Fallback for other k
        return _luw_construct(k, g)
        
    # Try N from 512 to 560
    for N in range(512, 562, 2):
        if time.time() > deadline:
            break
            
        # Try pairs (a, b)
        # a, b in [2, N-2]
        # To avoid duplicates and symmetry, we can restrict ranges, 
        # but for now, just iterate.
        # Optimization: skip if gcd(N, a) != 1 or gcd(N, b) != 1? 
        # Circulants are connected if gcd(N, 1, a, b) = 1. Since 1 is there, always connected.
        
        for a in range(2, N // 2 + 1):
            if time.time() > deadline:
                break
            for b in range(2, N // 2 + 1):
                if time.time() > deadline:
                    break
                
                # Quick prune:
                # If a == b, skip (symmetric, and often small girth)
                if a == b:
                    continue
                # If a + b == 1 or N-1, skip (triangles)
                if a + b == 1 or a + b == N - 1:
                    continue
                if abs(a - b) == 1:
                    continue
                
                # Check girth
                if _check_circulant_girth(N, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(N):
                        edges.append((i, (i + 1) % N))
                        edges.append((i, (i + a) % N))
                        edges.append((i, (i - b) % N))
                    # Ensure unique edges? 
                    # (i, i+1) and (i+1, i) are the same. 
                    # My loop adds (i, i+1) and (i+1, i+1+1)...
                    # It adds directed edges? No, (i, j) is an undirected edge.
                    # I should add each undirected edge once.
                    # Better:
                    edges = set()
                    for i in range(N):
                        edges.add((i, (i + 1) % N))
                        edges.add((i, (i + a) % N))
                        edges.add((i, (i - b) % N))
                    # Sort for consistency
                    edges_list = sorted(list(edges))
                    # Verify degree
                    # Each vertex has 3 neighbors.
                    return N, edges_list
                    
    # Fallback to LUW
    return _luw_construct(k, g)
