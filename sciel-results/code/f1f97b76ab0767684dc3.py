import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Precompute a valid 29-vertex coloring using the Paley graph construction
    # The Paley graph on q vertices (q = 1 mod 4 prime) has edges based on quadratic residues.
    # For q=29, the Paley graph is known to be K2,2,2-free in both colors.
    
    def paley_coloring(n):
        """
        Construct a Paley-type coloring for n vertices.
        For n=29, this uses quadratic residues mod 29.
        For other n, we use a random seed or extended construction.
        """
        if n == 29:
            # Paley graph on 29 vertices
            # Edge (i,j) is color 0 if (j-i) mod 29 is a quadratic residue, else 1
            residues = set()
            for x in range(1, 29):
                residues.add((x * x) % 29)
            
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 29
                    if diff in residues:
                        colors.append(0)
                    else:
                        colors.append(1)
            return colors
        
        # For other n, use random initialization
        return [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]

    def has_k222(colors, n, color):
        """
        Check if color class 'color' contains K2,2,2.
        Uses bitmask representation for fast intersection checks.
        """
        # Build adjacency bitmasks for the given color
        # adj[i] is a bitmask where bit j is set if edge (i,j) has the given color
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        
        # For each pair (u1, u2), find common neighbors in this color
        # Then check if those common neighbors contain a K2,2
        for u1 in range(n):
            for u2 in range(u1 + 1, n):
                # Common neighbors of u1 and u2 in this color
                common = adj[u1] & adj[u2]
                # Remove u1 and u2 themselves
                common &= ~(1 << u1)
                common &= ~(1 << u2)
                
                # Count bits in common
                k = bin(common).count('1')
                if k < 4:
                    continue
                
                # Extract the list of common neighbors
                S = []
                mask = common
                while mask:
                    # Get lowest set bit
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    S.append(v)
                    mask ^= lsb
                
                # Check if S contains a K2,2
                # A K2,2 exists if there are two vertices v1, v2 in S
                # that share at least 2 common neighbors in S
                for a_idx in range(k):
                    v1 = S[a_idx]
                    for b_idx in range(a_idx + 1, k):
                        v2 = S[b_idx]
                        # Common neighbors of v1 and v2 within S
                        common_v1v2 = adj[v1] & adj[v2] & common
                        # Remove v1 and v2 themselves
                        common_v1v2 &= ~(1 << v1)
                        common_v1v2 &= ~(1 << v2)
                        if bin(common_v1v2).count('1') >= 2:
                            return True
        return False

    def is_valid(colors, n):
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)

    def count_k222(colors, n, color):
        """
        Count the number of K2,2,2 subgraphs in color class 'color'.
        Used for local search.
        """
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        
        count = 0
        for u1 in range(n):
            for u2 in range(u1 + 1, n):
                common = adj[u1] & adj[u2]
                common &= ~(1 << u1)
                common &= ~(1 << u2)
                
                k = bin(common).count('1')
                if k < 4:
                    continue
                
                S = []
                mask = common
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    S.append(v)
                    mask ^= lsb
                
                for a_idx in range(k):
                    v1 = S[a_idx]
                    for b_idx in range(a_idx + 1, k):
                        v2 = S[b_idx]
                        common_v1v2 = adj[v1] & adj[v2] & common
                        common_v1v2 &= ~(1 << v1)
                        common_v1v2 &= ~(1 << v2)
                        cnt = bin(common_v1v2).count('1')
                        if cnt >= 2:
                            count += 1
        return count

    def local_search(colors, n, max_iter=50000):
        """
        Local search to minimize the number of K2,2,2 violations.
        """
        current_violations = count_k222(colors, n, 0) + count_k222(colors, n, 1)
        if current_violations == 0:
            return colors, 0
        
        best_colors = colors[:]
        best_violations = current_violations
        temp = 1.0
        cooling = 0.9995
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, n * (n - 1) // 2 - 1)
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            colors[edge_idx] = new_val
            
            new_violations = count_k222(colors, n, 0) + count_k222(colors, n, 1)
            
            delta = new_violations - current_violations
            if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors, 0
            else:
                colors[edge_idx] = old_val
            
            temp *= cooling
        
        return best_colors, best_violations

    # Start with the known valid 29-vertex Paley coloring
    n29 = 29
    colors29 = paley_coloring(n29)
    
    # Verify it's valid
    if is_valid(colors29, n29):
        best_n = n29
        best_colors = colors29
    else:
        # Fallback: use random and local search
        best_n = 17
        colors17 = [random.randint(0, 1) for _ in range(17 * 16 // 2)]
        colors17, _ = local_search(colors17, 17, max_iter=20000)
        if is_valid(colors17, 17):
            best_n = 17
            best_colors = colors17
        else:
            return {'n': 1, 'colors': [0]}

    # Try to extend to n=30 using vertex-doubling/blow-up heuristic
    # Strategy: Add a new vertex 29 and determine its colors to existing vertices
    # such that no new K2,2,2 is created.
    
    if time.time() < deadline - 30:
        n30 = 30
        # Initialize colors for K30 by extending the 29-vertex coloring
        # The new vertex is 29. We need to assign colors for edges (29, j) for j=0..28.
        
        # Start with the 29-vertex colors
        colors30 = colors29[:]
        
        # For each possible assignment of the new vertex's edges,
        # check if it maintains validity.
        # There are 2^29 possible assignments, which is too many.
        # Instead, use a greedy approach: for each edge (29, j), try both colors
        # and pick the one that doesn't create a violation.
        
        # But we need to be careful: the choice for one edge affects others.
        # Use a backtracking or local search approach.
        
        # Simpler approach: Try random assignments for the new vertex's edges,
        # and use local search to fix violations.
        
        found_valid_30 = False
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Randomly assign colors for edges from vertex 29 to vertices 0..28
            new_edges = [random.randint(0, 1) for _ in range(29)]
            
            # Build the full 30-vertex coloring
            colors30_attempt = colors29[:] + new_edges
            
            # Check validity
            if is_valid(colors30_attempt, 30):
                best_n = 30
                best_colors = colors30_attempt
                found_valid_30 = True
                break
            
            # If not valid, try to fix with local search
            colors30_fixed, viol = local_search(colors30_attempt, 30, max_iter=20000)
            if viol == 0:
                best_n = 30
                best_colors = colors30_fixed
                found_valid_30 = True
                break
        
        # If we found a valid 30-vertex coloring, we're done
        if found_valid_30:
            return {'n': best_n, 'colors': best_colors}

    # If we couldn't find a valid 30-vertex coloring, return the best we have
    return {'n': best_n, 'colors': best_colors}
