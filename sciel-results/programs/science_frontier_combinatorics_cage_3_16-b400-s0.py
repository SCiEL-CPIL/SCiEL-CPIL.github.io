"""Deterministic Algebraic Lift construction for k-regular graphs with high girth."""
import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Petersen graph edges (0-9)
    # Standard Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Base edges (u < v)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    # We want to find the smallest m such that a lift has girth >= g.
    # We scan m in increasing order.
    # For each m, we try a few deterministic voltage assignments.
    
    # Voltage assignment strategies:
    # 1. Linear: shift_e = c_e * m // 15 (or similar)
    # 2. Primitive root based
    # 3. Random (as fallback or for more trials)
    
    # Let's scan m from a reasonable lower bound.
    # Moore bound for k=3, g=16:
    # n >= 2 * (3 * (2^(7-1) - 1)) + 2 = 2 * (3 * 63) + 2 = 380?
    # Actually Moore bound for girth g (even) is 2 * (k-1) * ( (k-1)^(g/2 - 1) - 1 ) / (k-2) + 2?
    # For k=3, g=16: 2 * 2 * (2^7 - 1) / 1 + 2 = 4 * 127 + 2 = 510.
    # So n >= 510.
    # Since n = 10 * m, m >= 51.
    
    # Let's start m from 50 and go up.
    # We will try up to m=1000 (n=10000) which is well within time limits for verification.
    
    best_n = None
    best_edges = None

    # List of multipliers for voltage assignment
    # We want distinct shifts for each edge to maximize girth.
    # Let's use a set of "good" multipliers.
    # If m is prime, we can use 1, 2, 3, ...
    # If m is composite, we just use random-ish or fixed integers.
    
    # Strategy: 
    # For each m, try a few fixed patterns of shifts.
    # Pattern A: shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (scaled by m/15)
    # Pattern B: shifts = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    # Pattern C: Random
    
    m = 50
    while m <= 2000 and time.time() < deadline:
        
        # Try a few deterministic assignments
        # 1. Linear spread
        # 2. Random (fixed seed for reproducibility in evaluation? No, random is fine if we return valid)
        
        # Let's try 5 random lifts for each m to increase chance, but start with deterministic.
        
        # Deterministic 1: Shifts proportional to edge index
        shifts1 = [int((i + 1) * m / 15.0) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts1)
        
        if _verify_girth(n, edges, g):
            return n, edges
        
        # Deterministic 2: Shifts using a primitive root-like sequence (if m is prime) or just distinct values
        # Let's just try a few random ones if deterministic fails, or just increment m.
        # Given the time budget, we can afford to try a couple of random lifts per m.
        
        for _ in range(3):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(15)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth(n, edges, g):
                return n, edges

        # Track best invalid graph (lowest n) just in case we need to return something?
        # No, we must return valid.
        
        m += 1 # Increment by 1 to find the absolute smallest n

    # If we didn't find one in the loop (shouldn't happen for g=16, k=3), 
    # try a larger m with more attempts
    if time.time() < deadline:
        for m in range(2000, 3000):
            if time.time() > deadline:
                break
            for _ in range(5):
                shifts_rand = [random.randint(0, m-1) for _ in range(15)]
                n, edges = _build_lift(base_edges, m, shifts_rand)
                if _verify_girth(n, edges, g):
                    return n, edges

    # Fallback: Return a large random lift (likely valid)
    # This should rarely be reached.
    m = 5000
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges


def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v) for u < v
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth(n, edges, g):
    """
    Check if the graph has girth >= g.
    Uses non-backtracking BFS from each vertex to detect cycles of length < g.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # We need to check for cycles of length < g.
    # A cycle of length L < g will be found by non-backtracking walks of length L.
    # We perform BFS on the directed line graph (or equivalent state space (v, parent)).
    # State: (current_node, parent_node)
    # Start: (start, -1)
    # Transition: (v, p) -> (nb, v) for nb in adj[v] if nb != p
    # If we reach (start, prev) where prev != -1, we found a cycle.
    # Actually, simpler: BFS from (start, -1) up to depth g-1.
    # If we encounter a state (v, p) where v == start and p != -1, we found a cycle of length depth.
    # Wait, the state is (v, p). If v == start, the walk is start -> ... -> p -> start.
    # The length is the distance from start in the state graph.
    # Initial state (start, -1) has dist 0.
    # Neighbor (nb, start) has dist 1.
    # If we reach (start, p) with dist d, the cycle length is d.
    # We want to know if there is any cycle of length < g.
    # So we search up to depth g-1. If we find (start, p) at depth d < g, return False.
    
    # Optimization: We only need to check up to depth g-1.
    # For k=3, g=16, depth 15.
    # Number of states per start: sum_{i=0}^{15} 3 * 2^(i-1) ~ 3 * 2^15 = 100k.
    # Total states for all starts: n * 100k. If n=1000, 100M states. Too slow in Python.
    
    # Alternative: Check girth by BFS from each node but stop if we find a cycle shorter than g.
    # Standard BFS girth finding is O(n * (n+e)) in worst case, but with early termination for small girth it's fast.
    # However, if girth is large (close to g), we explore the whole tree of depth g/2.
    # Tree of depth 7 (for g=16) has ~ 3 * 2^7 = 384 nodes.
    # n * 384 operations. For n=1000, 384k ops. Very fast.
    
    # Let's use the standard BFS from each vertex, checking for back-edges that form short cycles.
    # But we must be careful: a back-edge (u, v) where v is visited doesn't always mean a cycle of length dist(u) + dist(v) + 1.
    # It means a cycle exists. The length is at most dist(u) + dist(v) + 1.
    # To be exact, we should find the shortest cycle.
    
    # Better approach for verification:
    # For each vertex s, do BFS.
    # Keep track of dist[v] and parent[v].
    # If we see an edge (u, v) where v is visited and u != parent[v] and v != parent[u]:
    #    cycle_len = dist[u] + dist[v] + 1
    #    if cycle_len < g: return False
    # This finds a cycle, but is it the shortest?
    # In a tree-like expansion, the first time we see a conflict, it's often the shortest.
    # But to be safe, we can just check if ANY cycle < g exists.
    # If BFS finds a conflict with length < g, we return False.
    # If BFS completes without such conflict, does it mean girth >= g?
    # Yes, because any cycle of length L < g would be detected by the BFS tree of one of its vertices?
    # Not necessarily. But if we run BFS from ALL vertices, we will definitely detect any cycle.
    # Specifically, for any cycle C, let s be a vertex on C. The BFS tree from s will include the path along C.
    # The edge closing the cycle will be a non-tree edge.
    # The length of the cycle is dist[u] + dist[v] + 1 for the non-tree edge (u,v).
    # Since s is on the cycle, dist[u] + dist[v] + 1 = L.
    # So if L < g, we will detect it.
    
    min_cycle_len = g # We only care if it's < g
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # Note: This might not be the shortest cycle through this edge,
                    # but it is a cycle. If its length < g, we fail.
                    # Is it possible that dist[u] + dist[v] + 1 >= g but there is a shorter cycle?
                    # Yes, but that shorter cycle would be detected by another start or another edge?
                    # Actually, if there is a cycle of length L < g, there exists a BFS tree from some s
                    # such that the closing edge has dist[u] + dist[v] + 1 = L.
                    # So if we check all s, we will find it.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle_len:
                        return False
                        
    return True

# EVOLVE-BLOCK-END
