import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # We try to find the largest n
    # Start from a high n and go down? Or start from 9 and go up?
    # If we start high, we might waste time on impossible n.
    # If we start low, we guarantee a baseline.
    # Let's try to find a solution for n, and if found, try n+1.
    # But searching for n=17 is hard.
    # Let's try n=9 (known), then 10, 11...
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to extend to larger n
    # We will use a backtracking search for each n
    
    for n in range(10, 21):
        if time.time() > deadline:
            break
            
        # Setup for search
        # neighbors[c][v] = set of neighbors
        neighbors = [set() for _ in range(4)] # Actually, we need per vertex
        # Let's use a list of sets: neighbors[c][v]
        neighbors = [[set() for _ in range(n)] for _ in range(4)]
        
        # Fixed edges for symmetry breaking
        # Vertices 0, 1, 2, 3 form a clique
        # c(0,1)=0, c(0,2)=0, c(0,3)=1
        # c(1,2)=1, c(1,3)=2, c(2,3)=2
        # This is a valid partial coloring.
        
        edges = {}
        
        def add_edge(u, v, c):
            if u > v: u, v = v, u
            edges[(u, v)] = c
            neighbors[c][u].add(v)
            neighbors[c][v].add(u)
            
        # Check if adding edge (u,v) with color c creates a C6
        def check_c6(u, v, c):
            # Path u-a-b-v in color c
            # a in N_c(u)
            for a in neighbors[c][u]:
                if a == v: continue
                for b in neighbors[c][a]:
                    if b == u or b == v: continue
                    if v in neighbors[c][b]:
                        # Check distinctness u, a, b, v
                        if u != b and a != v:
                            return False
            return True
            
        # Recursive search
        # We assign colors to edges (0, k), (1, k), ..., (k-1, k) for k from 4 to n-1
        # Vertices 0..3 are fixed.
        
        # Initialize fixed edges
        # (0,1)=0, (0,2)=0, (0,3)=1
        # (1,2)=1, (1,3)=2, (2,3)=2
        try:
            add_edge(0, 1, 0)
            add_edge(0, 2, 0)
            add_edge(0, 3, 1)
            add_edge(1, 2, 1)
            add_edge(1, 3, 2)
            add_edge(2, 3, 2)
        except:
            pass # Should not happen
            
        # We need to store the state of neighbors to backtrack
        # Copying sets is expensive. 
        # Alternative: Use a stack of changes.
        # Or, since n is small, we can just copy the `neighbors` structure.
        # n=20, 4 colors, 20 vertices. 80 sets. Copying is fast.
        
        def solve(k):
            if time.time() > deadline:
                return None
            
            if k == n:
                # Extract colors
                res = []
                for i in range(n):
                    for j in range(i+1, n):
                        res.append(edges[(i, j)])
                return res
                
            # Assign colors to (0, k), (1, k), ..., (k-1, k)
            # We can use symmetry breaking for k=4: c(0,4)=0
            # For k>4, we can try all 4 colors, but prune if possible.
            
            # To speed up, we can order the edges.
            # Let's just try all 4 colors for each edge (i, k) for i in 0..k-1.
            
            # We need to backtrack.
            # We'll use a recursive helper that takes the current vertex k.
            
            # Optimization: If we are at vertex k, we can check if the current partial graph
            # already has a C6? No, we check on insertion.
            
            # Let's try to assign colors to (0, k) first.
            
            # We need to save the state of neighbors and edges to backtrack.
            # Since Python sets are mutable, we need to undo changes.
            # We will use a stack of (u, v, c) to undo.
            
            def backtrack(i, k, undo_stack):
                if time.time() > deadline:
                    return None
                
                if i == k:
                    return solve(k + 1)
                    
                u = i
                v = k
                for c in range(4):
                    # Check if (u, v) is already assigned? 
                    # In our scheme, we assign (0, k), (1, k)... so (u, v) is new.
                    # But we must ensure we don't assign a color that creates C6.
                    if check_c6(u, v, c):
                        # Assign
                        edges[(u, v)] = c
                        neighbors[c][u].add(v)
                        neighbors[c][v].add(u)
                        undo_stack.append((u, v, c))
                        
                        res = backtrack(i + 1, k, undo_stack)
                        if res is not None:
                            return res
                            
                        # Undo
                        u_undo, v_undo, c_undo = undo_stack.pop()
                        neighbors[c_undo][u_undo].remove(v_undo)
                        neighbors[c_undo][v_undo].remove(u_undo)
                        del edges[(u_undo, v_undo)]
                        
                return None
            
            # Start backtracking for vertex k
            # We can add symmetry breaking:
            # If k == 4, force c(0,4)=0?
            # Let's just run it.
            
            undo_stack = []
            return backtrack(0, k, undo_stack)
        
        # Run solver for this n
        # We need to reset neighbors and edges for each n?
        # Yes, because n changes.
        
        # Re-initialize for this n
        neighbors = [[set() for _ in range(n)] for _ in range(4)]
        edges = {}
        
        # Redefine add_edge and check_c6 using the local variables
        # Python closures capture by reference, so we need to be careful.
        # Let's wrap in a function or class to avoid global state issues.
        
        # Actually, let's just put the search logic inside a function that takes n.
        
    # If we didn't find anything better than 9, return 9.
    # But the loop above is messy. Let's rewrite cleanly.
    
    return {'n': best_n, 'colors': best_colors}
