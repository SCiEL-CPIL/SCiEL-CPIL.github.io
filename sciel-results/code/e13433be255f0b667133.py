import time
import random
import sys

# Increase recursion limit just in case, though we might use iterative if needed
sys.setrecursionlimit(10000)

def construct():
    """
    Find a 4-coloring of K_n with no monochromatic C6, maximizing n.
    Returns {'n': n, 'colors': [...]}
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def has_path_len_5(adj, u, v):
        """
        Check if there is a simple path of length 5 from u to v in the graph defined by adj.
        adj is a list of bitmasks (int).
        Path: u - x1 - x2 - x3 - x4 - v
        """
        # Stack-based DFS to avoid recursion overhead
        # State: (current_node, depth, visited_mask)
        # visited_mask includes the start node u
        stack = [(u, 0, 1 << u)]
        
        while stack:
            node, depth, visited = stack.pop()
            
            if depth == 5:
                if node == v:
                    return True
                continue
            
            # If we reach v before depth 5, we can't continue through v
            # because the path must be simple and v is the endpoint
            if node == v:
                continue
                
            # Get neighbors of node that are not visited
            neighbors = adj[node] & ~visited
            
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                
                # If next_node is v, we can only accept it if depth+1 == 5
                if next_node == v:
                    if depth + 1 == 5:
                        return True
                    else:
                        continue
                        
                stack.append((next_node, depth + 1, visited | lsb))
        
        return False

    def solve_backtrack(n):
        """
        Solve for K_n using backtracking with bitmask pruning.
        Returns list of colors for edges in lex order, or None.
        """
        # Precompute edge list in lex order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # adj[c] is a list of n bitmasks
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        # Symmetry breaking:
        # Fix colors of edges (0,1), (0,2), (0,3), (0,4) to 0, 1, 2, 3
        # This assumes n >= 5
        if n >= 5:
            fixed = {
                (0, 1): 0,
                (0, 2): 1,
                (0, 3): 2,
                (0, 4): 3
            }
            for (i, j), c in fixed.items():
                # Find index
                # i=0, j=k -> index is k-1
                idx = j - 1
                colors[idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Find the first unassigned edge
        start_idx = 0
        while start_idx < num_edges and colors[start_idx] != -1:
            start_idx += 1
            
        if start_idx == num_edges:
            return colors
        
        # Recursive backtracking
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == num_edges:
                return colors
            
            i, j = edges[idx]
            
            # Try all 4 colors
            for c in range(4):
                # Check if adding (i, j) to color c creates a C6
                # This happens if there is a path of length 5 from i to j in adj[c]
                if has_path_len_5(adj[c], i, j):
                    continue
                
                # Assign color c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[idx] = c
                
                # Recurse
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Undo assignment
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[idx] = -1
            
            return None
        
        return backtrack(start_idx)

    # Try to find solutions for decreasing n, starting from 17
    # We prioritize higher n
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for n in range(17, 9, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Try to solve for this n
        # We might need to restart with different random seeds or symmetry breaking?
        # For now, try once. If it fails, move to n-1.
        # To increase chances, we could randomize the order of colors tried in backtrack.
        # But symmetry breaking fixes the first few, so let's stick to deterministic first.
        
        res = solve_backtrack(n)
        
        if res is not None:
            best_n = n
            best_colors = res
            # Found a valid coloring for n, return immediately as we want max n
            # But wait, we are iterating 17, 16, ...
            # If we find 17, we return.
            # If we fail 17, we try 16.
            break
        else:
            # If it timed out or failed, we try the next smaller n
            # But if it timed out, we might not have enough time for smaller n either
            # However, smaller n is easier, so we should try.
            pass

    # Fallback if nothing found (should not happen, n=9 is trivial)
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
        
    return {'n': best_n, 'colors': best_colors}

# To allow the solver to be more robust, we can add a randomization element
# but the current deterministic approach with strong pruning should be fast enough
# for n=17 if a solution exists. If it doesn't exist, it will prune heavily.

# One potential issue: The recursion depth is n*(n-1)/2. For n=17, that's 136.
# Python's default recursion limit is 1000, so we are fine.
# However, Python function call overhead is significant.
# The `has_path_len_5` call is the bottleneck.
# Let's hope the pruning is effective.

# If the backtracker is too slow, we might need to switch to a greedy+local search hybrid
# but for now, this is the structural change: Bitmask + Path Length 5 Pruning.

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found n={res['n']}")
