import time
import sys
from itertools import combinations

# Precompute edge positions
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map: (min(u,v), max(u,v)) -> 1-based position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def greedy_initial_upper_bound():
    """
    Find a Hamilton cycle using a greedy approach to establish an upper bound.
    We try to build a cycle by adding edges in order of position.
    """
    # This is a heuristic. We don't guarantee it finds the optimal, but a good one.
    # We will use a simple backtracking that prefers early edges.
    
    # Actually, let's just use the full graph backtracking with a time limit to find *any* cycle,
    # then use its score as the upper bound.
    # But we want a *good* upper bound.
    
    # Let's try to construct a cycle where edges are as small as possible.
    # We can use a "minimum weight" heuristic.
    
    # For now, let's just return a large number if this is too complex.
    # The binary search will handle it.
    # A safe upper bound is 276.
    return 276

def solve_hamilton_cycle_bitmask(adj_masks, n, deadline):
    """
    Solve Hamilton cycle using bitmask DP/Backtracking.
    adj_masks: list of ints, where adj_masks[v] is a bitmask of neighbors.
    Returns the cycle as a list of vertices or None.
    """
    # We fix vertex 0 as the start.
    # visited_mask: bitmask of visited vertices.
    # last: the current vertex.
    # path: list of vertices in the path.
    
    # We use recursion with pruning.
    
    # Precompute: for each mask, is it valid? No, too many masks.
    
    # Pruning strategies:
    # 1. If last has no unvisited neighbors, fail (unless we are at the end).
    # 2. Connectivity of unvisited nodes.
    
    # To speed up, we can use iterative deepening or just DFS with good ordering.
    
    # Let's try a DFS with bitmask.
    
    # We need to reconstruct the path, so we pass the path list.
    
    # Optimization: If we are at the last step, check if last connects to 0.
    
    # Let's use a simple recursive DFS with bitmasks.
    
    # To avoid recursion depth issues (24 is fine), and to allow early termination.
    
    # We will sort neighbors by degree in the remaining graph to prune faster?
    # That's dynamic.
    
    # Let's just try all unvisited neighbors.
    
    # State: (visited_mask, last)
    # We can memoize "failed" states.
    failed_states = set()
    
    def dfs(visited_mask, last, path):
        if time.time() > deadline:
            return None
        
        if visited_mask == (1 << n) - 1:
            # Check if last connects to 0
            if adj_masks[last] & 1: # bit 0 is set
                return path[:]
            return None
        
        # Pruning: Check if last has any unvisited neighbors
        unvisited = ((1 << n) - 1) ^ visited_mask
        if not (adj_masks[last] & unvisited):
            return None
        
        # Pruning: Check if any unvisited node is isolated in the remaining graph
        # (excluding the connection to 'last' which is handled by the previous check for 'last')
        # Actually, we need to ensure that the unvisited subgraph is connected to 'last' and '0'.
        # This is expensive. Let's do a lighter check:
        # If an unvisited node u has no neighbors in (unvisited | {last}), it's a dead end.
        # Note: u might connect to 0, but 0 is already visited.
        # So u must connect to some other unvisited node or last.
        
        # Let's check for dead ends.
        for u in range(n):
            if not (visited_mask & (1 << u)):
                # u is unvisited
                # Neighbors of u in the remaining graph:
                # (adj_masks[u] & unvisited) | (adj_masks[u] & (1 << last))
                # If this is 0, then u cannot be reached.
                # Wait, if u is the *next* node, it connects to last.
                # If u is *not* the next node, it must connect to another unvisited node.
                
                # A node u is a "dead end" if:
                # 1. It has no neighbors in unvisited.
                # 2. It is not the only unvisited node (which would be handled by the last step check).
                # Actually, if u has no neighbors in unvisited, it MUST be visited immediately after last.
                # If there are multiple such nodes, we fail.
                
                pass # Too complex to implement quickly without bugs. Let's stick to simpler pruning.
        
        # Simple Pruning: If the number of unvisited nodes is 1, check if it connects to last and 0.
        if unvisited & (unvisited - 1) == 0:
            # Only one unvisited node
            u = unvisited.bit_length() - 1
            if (adj_masks[last] & (1 << u)) and (adj_masks[u] & 1):
                return path + [u]
            else:
                return None
        
        # Try neighbors
        # Order neighbors by degree in remaining graph?
        # Let's just iterate.
        neighbors = adj_masks[last] & unvisited
        
        while neighbors:
            lsb = neighbors & -neighbors
            u = lsb.bit_length() - 1
            neighbors ^= lsb
            
            if (visited_mask | lsb, u) in failed_states:
                continue
                
            new_path = path + [u]
            res = dfs(visited_mask | lsb, u, new_path)
            if res is not None:
                return res
        
        failed_states.add((visited_mask, last))
        return None

    return dfs(1, 0, [0])

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    n = N
    total_edges = n * (n - 1) // 2  # 276
    
    # Get a good upper bound
    # We can try to find a cycle in the full graph first (which is trivial)
    # But we want a *tight* upper bound.
    # Let's try a few random cycles or a greedy one.
    
    # For now, set upper bound to total_edges.
    # We can improve this by checking a few specific thresholds.
    
    lo, hi = 24, total_edges
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build adjacency masks for subgraph with edges <= mid
        adj_masks = [0] * n
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        
        # Check for Hamilton cycle
        cycle = solve_hamilton_cycle_bitmask(adj_masks, n, deadline)
        
        if cycle is not None:
            hi = mid - 1
            # Store best cycle
            best_cycle = cycle
            best_score = mid
        else:
            lo = mid + 1
    
    if 'best_cycle' in locals():
        return best_cycle
    
    # Fallback: Return a valid cycle from the full graph
    # The full graph is K24, so any permutation is a Hamilton cycle.
    # We just need to return a permutation.
    # To be safe, return a simple cycle 0-1-2-...-23-0.
    return list(range(n))
