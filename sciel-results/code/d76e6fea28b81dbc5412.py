import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a 4-coloring of K_n with no monochromatic C6.
    Maximizes n.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try to find the largest n
    # Known best is 17. Try 17, 16, 15...
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        # Allocate time for this n
        # If n is large, give it more time?
        # Let's just run until deadline or success.
        success, colors = solve_n(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a known valid small coloring if search fails
    # Seed solution for n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    """
    Attempts to find a valid 4-coloring for K_n.
    Returns (True, colors) if found, (False, None) otherwise.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # adj[c][v] is a bitmask of neighbors of v in color c
    adj = [[0] * n for _ in range(4)]
    colors = [0] * num_edges
    
    # Precompute edge indices for quick access?
    # edges list is fine.
    
    # Symmetry breaking:
    # We can fix the colors of the first few edges to reduce symmetry.
    # (0,1) -> 0
    # (0,2) -> 1
    # (0,3) -> 2
    # This is valid because we can permute colors and vertices.
    
    def check_c6(u, v, c):
        """
        Checks if adding edge (u, v) to color c creates a C6.
        This is equivalent to checking if there is a simple path of length 5
        from u to v in the current graph of color c.
        """
        # L_k is a set of vertices reachable by a simple path of length k from u
        # We use bitmasks for efficiency
        L = [0] * 6
        L[0] = 1 << u
        prev = L[0]
        
        for k in range(1, 6):
            curr = 0
            # For each vertex in prev, add its neighbors in color c
            # But we must exclude vertices already visited in L[0]...L[k-1]
            visited_mask = 0
            for i in range(k):
                visited_mask |= L[i]
                
            # Iterate through vertices in prev
            mask = prev
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                x = lsb.bit_length() - 1
                mask ^= lsb
                
                # Neighbors of x in color c
                neighbors = adj[c][x]
                # Remove visited vertices
                candidates = neighbors & ~visited_mask
                curr |= candidates
            
            L[k] = curr
            prev = curr
            
            if (L[k] & (1 << v)) != 0:
                return False # C6 found
            
            if L[k] == 0:
                break
                
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == num_edges:
            return True
            
        u, v = edges[idx]
        
        # Try each color
        for c in range(4):
            # Symmetry breaking for the first few edges
            if idx == 0 and c != 0: continue
            if idx == 1 and c != 1: continue
            if idx == 2 and c != 2: continue
            
            if check_c6(u, v, c):
                # Assign
                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                    
                # Unassign
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                
        return False

    return backtrack(0), (colors if backtrack(0) else None) # Note: backtrack(0) is called twice if we do this. Better to structure differently.

# Refactor solve_n to avoid double call
def solve_n(n, deadline):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    adj = [[0] * n for _ in range(4)]
    colors = [0] * num_edges
    
    def check_c6(u, v, c):
        L = [0] * 6
        L[0] = 1 << u
        prev = L[0]
        for k in range(1, 6):
            curr = 0
            visited_mask = 0
            for i in range(k):
                visited_mask |= L[i]
            mask = prev
            while mask:
                lsb = mask & -mask
                x = lsb.bit_length() - 1
                mask ^= lsb
                neighbors = adj[c][x]
                candidates = neighbors & ~visited_mask
                curr |= candidates
            L[k] = curr
            prev = curr
            if (L[k] & (1 << v)) != 0:
                return False
            if L[k] == 0:
                break
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == num_edges:
            return True
        u, v = edges[idx]
        for c in range(4):
            if idx == 0 and c != 0: continue
            if idx == 1 and c != 1: continue
            if idx == 2 and c != 2: continue
            if check_c6(u, v, c):
                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                if backtrack(idx + 1):
                    return True
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
        return False

    success = backtrack(0)
    if success:
        return True, colors
    else:
        return False, None

# EVOLVE-BLOCK-END
