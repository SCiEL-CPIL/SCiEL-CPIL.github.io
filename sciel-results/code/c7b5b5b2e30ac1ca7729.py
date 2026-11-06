"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    q = 5
    n = 31

    # 1. Generate points of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for t in range(q):
        points.append([1, t, 0])
    points.append([0, 1, 0])
    
    # Verify we have 31 points
    if len(points) != n:
        # Fallback to a simple random graph if construction fails (should not happen)
        random.seed(42)
        edges = set()
        while len(edges) < 91:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u < v:
                edges.add((u, v))
        return [[u, v] for u, v in edges]

    # 2. Build polarity graph: edge if dot product is 0 mod q
    # This graph is known to be C4-free and has 90 edges for q=5.
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
    
    if len(edges) != 90:
        # Fallback
        random.seed(42)
        edges = set()
        while len(edges) < 91:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u < v:
                edges.add((u, v))
        return [[u, v] for u, v in edges]

    # Build adjacency matrix for fast C4 counting
    A = np.zeros((n, n), dtype=np.int64)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1

    # 3. Find the non-edge that creates the fewest 4-cycles
    # The number of new 4-cycles created by adding edge (u, v) is equal to
    # the number of length-3 paths between u and v in the current graph.
    # This is given by (A^3)[u, v].
    # Since the base graph is C4-free, (A^3)[u, v] counts exactly the new C4s.
    
    A2 = A @ A
    A3 = A @ A2
    
    min_count = float('inf')
    best_u, best_v = -1, -1
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if A[i, j] == 1:
                continue
            count = A3[i, j]
            if count < min_count:
                min_count = count
                best_u, best_v = i, j

    if best_u == -1:
        # Should not happen, but just in case
        edges.append([0, 1])
        return edges

    # Add the best edge
    edges.append([best_u, best_v])
    A[best_u, best_v] = 1
    A[best_v, best_u] = 1

    # 4. Local Search / Simulated Annealing to minimize C4 count
    # We will perform edge swaps: remove one edge, add another non-edge.
    # Goal: reduce total C4 count.
    
    def count_c4():
        # Count C4s using trace(A^4)
        # Number of closed walks of length 4 is trace(A^4).
        # Each C4 contributes 8 to trace(A^4) (4 starting vertices * 2 directions).
        # Triangles contribute 0.
        # So C4_count = trace(A^4) // 8
        A2 = A @ A
        A4 = A2 @ A2
        return np.trace(A4) // 8

    current_c4 = count_c4()
    best_c4 = current_c4
    best_edges = [e[:] for e in edges]
    best_A = A.copy()

    # Precompute list of non-edges for random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Build a set of current edges for O(1) lookup
    edge_set = set((u, v) for u, v in edges)
    
    # Start temperature for simulated annealing
    T = 10.0
    cooling_rate = 0.99
    
    # Number of iterations based on time
    # Each iteration involves:
    # 1. Pick random edge to remove
    # 2. Pick random non-edge to add
    # 3. Calculate delta C4
    # 4. Update
    
    # We can calculate delta C4 efficiently?
    # Removing edge (u,v) and adding (a,b).
    # Delta C4 = C4(new) - C4(old).
    # This is complex to calculate in O(1).
    # Given n=31, count_c4() is fast enough (matrix mult 31x31).
    # Let's estimate: 31^3 * 2 ~ 60k ops per count_c4.
    # 150s / 60k ops is plenty of time for thousands of iterations.
    
    iter_count = 0
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges) - 1)
        u, v = edges[idx]
        
        # Pick a random non-edge to add
        # We need to ensure we don't pick the same edge or an existing edge
        attempts = 0
        while attempts < 100:
            a, b = random.choice(all_pairs)
            if (a, b) not in edge_set and a != u and b != v: # Just a heuristic to avoid trivial swaps, not strictly necessary
                # Check if (a,b) is already an edge
                if A[a, b] == 0:
                    break
            attempts += 1
        else:
            continue # Couldn't find a valid non-edge quickly

        # Perform swap
        # Remove (u,v)
        A[u, v] = 0
        A[v, u] = 0
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        
        # Add (a,b)
        A[a, b] = 1
        A[b, a] = 1
        edge_set.add((a, b))
        edge_set.add((b, a))
        
        new_c4 = count_c4()
        
        delta = new_c4 - current_c4
        
        # Simulated Annealing acceptance
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            # Accept
            current_c4 = new_c4
            # Update edges list
            edges[idx] = [a, b]
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = [e[:] for e in edges]
                best_A = A.copy()
        else:
            # Reject
            # Undo changes
            A[u, v] = 1
            A[v, u] = 1
            edge_set.add((u, v))
            edge_set.add((v, u))
            
            A[a, b] = 0
            A[b, a] = 0
            edge_set.discard((a, b))
            edge_set.discard((b, a))
        
        T *= cooling_rate
        iter_count += 1

    # Return the best solution found
    return best_edges
# EVOLVE-BLOCK-END
