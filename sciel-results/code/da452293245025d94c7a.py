import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    N = 512
    
    # 1. Build compatibility graph using bitmasks
    # adj[i] is a bitmask where bit j is 1 if lcm(i+1, j+1) <= 512
    adj = [0] * N
    
    for i in range(N):
        val_i = i + 1
        for j in range(i + 1, N):
            val_j = j + 1
            g = gcd(val_i, val_j)
            # lcm = val_i * val_j // g
            if (val_i * val_j) // g <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Tomita's Maximum Clique Algorithm
    # We search for the maximum clique in the compatibility graph.
    
    best_clique = []
    best_size = 0
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    # Greedy Coloring of the subgraph induced by P
    # Returns a list of (vertex, color) sorted by color
    # P is a list of vertex indices
    def greedy_color(P):
        # P is a list of vertices
        # We want to assign colors such that adjacent vertices have different colors
        # and maximize the color number to get a good upper bound
        
        # Standard greedy coloring for Max Clique (Tomita)
        # We process vertices in P.
        # We maintain available colors for each vertex.
        
        # To speed this up, we can use a simple sequential coloring.
        # For each vertex v in P, assign the smallest color not used by its neighbors in P that have been colored.
        
        # However, for Tomita, we often order P by degree or just use the given order.
        # The standard Tomita algorithm orders P by the greedy coloring result.
        
        # Let's implement a specific version:
        # 1. Initialize colors to 0.
        # 2. For each v in P, look at neighbors in P.
        #    Actually, we need to color the subgraph induced by P.
        
        # Efficient approach:
        # We maintain a list `colors` of size N (or just for P).
        # We iterate through P.
        
        # A faster heuristic:
        # Sort P by degree in the subgraph? Or just iterate.
        
        # Let's use the standard "coloring" step from Tomita.
        # It assigns colors 1, 2, ... k to vertices in P.
        # The vertices are then ordered by color.
        
        # To be efficient in Python, we can use a list of sets or bitmasks.
        
        # Let's try a simple greedy:
        # colors = {}
        # for v in P:
        #    used = set()
        #    for u in P:
        #        if u != v and (adj[v] >> u) & 1:
        #            if u in colors:
        #                used.add(colors[u])
        #    # Find smallest non-negative int not in used
        #    c = 0
        #    while c in used:
        #        c += 1
        #    colors[v] = c
        
        # This is O(|P|^2 * degree). With |P| <= 512, this is ~260k ops per color step.
        # Number of color steps = depth of recursion.
        # This might be too slow in Python if the tree is deep.
        
        # Optimization: Use bitmasks for "used colors"?
        # Colors are small integers.
        
        # Let's try to optimize the coloring loop.
        
        # We can pre-compute the neighbors in P for each v?
        # Or just iterate.
        
        # Alternative: Use a library? No, stdlib only.
        
        # Let's stick to the bitmask intersection for the search, and optimize coloring.
        
        # Actually, for N=512, a well-implemented Tomita in Python should be fast enough
        # if the pruning is effective.
        
        # Let's refine the coloring to be faster.
        # We can group vertices by their "available colors" or something?
        
        # Let's try a different ordering:
        # Order P by degree in the subgraph descending?
        
        # For now, let's implement the basic Tomita and see if it fits in time.
        # If not, we might need to add a time check and return best found.
        
        # The function returns a list of (vertex, color)
        colored = []
        color_of = {}
        
        # To speed up, we can use the fact that we only need the max color for bounds
        # and the order for branching.
        
        # Let's use a list `P` and iterate.
        # We will store the color in a dictionary `color_of`.
        
        # Optimization: Use a list `colors` of length N, initialized to -1.
        colors = [-1] * N
        
        for v in P:
            used = 0
            # Check neighbors in P
            # We can iterate through P again? O(|P|^2)
            # Or iterate through bits of adj[v] & mask(P)?
            
            # Let's create a bitmask for P
            # But we are in the middle of coloring, P is a list.
            
            # Let's just use the dictionary lookup.
            for u in P:
                if u != v and (adj[v] >> u) & 1:
                    if colors[u] != -1:
                        used |= (1 << colors[u])
            
            # Find smallest color not in used
            c = 0
            while (used >> c) & 1:
                c += 1
            colors[v] = c
            colored.append((v, c))
            
        # Sort by color descending?
        # Tomita processes vertices in reverse order of color.
        # So we sort by color descending.
        colored.sort(key=lambda x: x[1], reverse=True)
        
        return colored

    def expand(R, P, X):
        nonlocal best_clique, best_size
        
        if time_exceeded():
            return
            
        # If P is empty, we have a maximal clique
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Coloring step
        colored = greedy_color(P)
        
        # Iterate in reverse order of color
        for v, c in reversed(colored):
            # Pruning: if len(R) + c <= best_size, skip
            if len(R) + c <= best_size:
                return # Since we are iterating in descending color, all subsequent have lower/equal color
            
            # Branch on v
            R.append(v)
            
            # New P: neighbors of v in P
            # New X: neighbors of v in X
            new_P = []
            new_X = []
            
            # P is a list of vertices
            # We need to filter P to keep only neighbors of v
            # Using bitmask would be faster if P was a bitmask
            # Let's switch P and X to bitmasks for faster intersection
            
            # Wait, greedy_color expects a list.
            # Let's keep P and X as lists for the coloring, but convert to bitmask for intersection?
            # Or just filter the list.
            
            for u in P:
                if (adj[v] >> u) & 1:
                    new_P.append(u)
            for u in X:
                if (adj[v] >> u) & 1:
                    new_X.append(u)
            
            expand(R, new_P, new_X)
            
            R.pop()
            
            # Move v from P to X
            # We don't strictly need to update P and X lists for the loop if we are careful,
            # but standard Tomita removes v from P and adds to X for subsequent iterations.
            # However, since we iterate over `colored` which is a snapshot,
            # we must ensure we don't process v again or its non-neighbors.
            # Actually, the standard algorithm modifies P and X in the loop.
            # But my loop `for v, c in reversed(colored)` iterates over the original P.
            # This is incorrect for Tomita if we don't update P.
            
            # Correct Tomita structure:
            # P is a set.
            # while P is not empty:
            #   color P
            #   pick v with max color
            #   if len(R) + color[v] <= best: break
            #   expand(R + v, P & N(v), X & N(v))
            #   P = P - {v}
            #   X = X + {v}
            
            # Let's rewrite the loop to match this.
            
    # Let's re-implement expand correctly with the while loop structure.
    
    def expand_correct(R, P, X):
        nonlocal best_clique, best_size
        
        if time_exceeded():
            return
            
        # P and X are lists of vertices
        # We will use a while loop on P
        
        while P:
            if time_exceeded():
                return
            
            # Color the subgraph induced by P
            colored = greedy_color(P)
            
            # Find the vertex with the maximum color
            # colored is sorted by color descending
            # So the first element has the max color
            if not colored:
                break
                
            v, c = colored[0]
            
            # Pruning
            if len(R) + c <= best_size:
                return # All other vertices in P have color <= c, so they can't improve
            
            # Branch
            R.append(v)
            
            new_P = []
            new_X = []
            
            # Filter P and X
            for u in P:
                if (adj[v] >> u) & 1:
                    new_P.append(u)
            for u in X:
                if (adj[v] >> u) & 1:
                    new_X.append(u)
            
            expand_correct(R, new_P, new_X)
            
            R.pop()
            
            # Update P and X
            # Remove v from P
            P.remove(v)
            # Add v to X
            X.append(v)
            
            if time_exceeded():
                return

        # If P becomes empty, we have a maximal clique
        if not P and len(R) > best_size:
            best_size = len(R)
            best_clique = list(R)

    # Initial call
    # P = all vertices 0..511
    # X = empty
    P_init = list(range(N))
    X_init = []
    R_init = []
    
    expand_correct(R_init, P_init, X_init)
    
    # Construct the result set
    # The clique found is indices in 0..511, corresponding to values 1..512
    result = [v + 1 for v in best_clique]
    
    # If no clique found (shouldn't happen, at least {1} is valid), return {1}
    if not result:
        return [1]
        
    return result

if __name__ == "__main__":
    construct()
