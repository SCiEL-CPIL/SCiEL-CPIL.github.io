"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill
    
    n = 20
    colors = 20
    
    # 1. Build adjacency and color lookup for fast access
    # adj[u][v] = c
    adj = [[None] * n for _ in range(n)]
    # edge_color[u][v] = c (for u != v)
    edge_color = [[None] * n for _ in range(n)]
    # To check 4-cycles u-a-v-b-u, we need to know if u-v edge exists and its color.
    # The 4-cycle condition: c(u,a), c(a,v), c(v,b), c(b,u) must be distinct.
    
    def is_valid_addition(u, v, c):
        # Check all 4-cycles containing edge (u, v).
        # A 4-cycle containing u-v is u-a-v-b-u.
        # This requires edges (u,a), (a,v), (v,b), (b,u) to exist.
        # We are adding (u,v) with color c.
        # So we need to check all pairs (a, b) such that:
        # 1. (u,a) exists, (a,v) exists, (v,b) exists, (b,u) exists.
        # 2. a != b, a != u, a != v, b != u, b != v.
        # 3. The colors c(u,a), c(a,v), c(v,b), c(b,u) are distinct and none is c.
        
        # Iterate over common neighbors of u and v?
        # No, a is a neighbor of u and v?
        # In a 4-cycle u-a-v-b-u:
        # a is connected to u and v.
        # b is connected to v and u.
        # So a and b are common neighbors of u and v.
        # Let Common = { w | adj[u][w] is not None and adj[v][w] is not None }
        # For any pair a, b in Common with a != b:
        # The cycle is u-a-v-b-u.
        # Edges: (u,a), (a,v), (v,b), (b,u).
        # Colors: c_ua, c_av, c_vb, c_bu.
        # We are adding (u,v) with color c. Wait, the edge (u,v) is NOT part of the 4-cycle u-a-v-b-u.
        # The edges in the cycle are (u,a), (a,v), (v,b), (b,u).
        # The edge (u,v) is a chord of this 4-cycle?
        # NO. The task says "Every 4-cycle (as a subgraph) must receive four distinct colours".
        # If we add edge (u,v), does it create new 4-cycles?
        # A 4-cycle is a subgraph isomorphic to C4.
        # If we add (u,v), new 4-cycles might be formed where (u,v) is one of the edges.
        # So we must check 4-cycles where (u,v) is an edge.
        # Cycle: u-v-a-b-u.
        # Edges: (u,v), (v,a), (a,b), (b,u).
        # This requires (v,a), (a,b), (b,u) to exist.
        # And (u,v) is the new edge.
        # So we iterate over pairs (a, b) such that:
        # 1. (v,a) exists
        # 2. (a,b) exists
        # 3. (b,u) exists
        # 4. a != u, a != v, b != u, b != v, a != b.
        # 5. Colors c(v,a), c(a,b), c(b,u) must be distinct and not equal to c.
        
        # Optimization:
        # For a fixed u, v, and candidate color c:
        # We need to ensure that for all a in N(v) \ {u}, b in N(u) \ {v}:
        # If (a,b) exists, then {c, c(v,a), c(a,b), c(b,u)} has size 4.
        
        # Let's precompute neighbors for speed.
        # This check is O(n^2) per edge addition.
        # With ~200 edges, total checks ~ 40,000 * 400 = 16M ops. Should be fast enough in Python?
        # 16M simple ops might take ~1-2 seconds. We have 115s.
        
        for a in range(n):
            if a == u or a == v: continue
            if adj[v][a] is None: continue
            c_va = adj[v][a]
            if c_va == c: return False # c(v,a) == c(u,v) -> not rainbow
            for b in range(n):
                if b == u or b == v or b == a: continue
                if adj[u][b] is None: continue
                if adj[a][b] is None: continue
                c_bu = adj[u][b]
                c_ab = adj[a][b]
                if c_bu == c or c_ab == c: return False
                if c_va == c_bu or c_va == c_ab or c_bu == c_ab: return False
        return True

    # Greedy construction with randomization
    # We want to maximize edges.
    # Strategy: Try to add edges in a random order, or try all possible edges and add if valid.
    # To get a dense graph, we should try to add edges that are "easy" to add, or just random greedy.
    # Random greedy: shuffle all possible edges, try to add them.
    # If we fail, we might get stuck.
    # Better: Multiple restarts? Or just one good greedy pass?
    # Given the time budget, a single pass over all 190 edges with random order might work.
    # But we can do better: Try to add edges in a specific order?
    # Let's try random greedy first.
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = []
    best_count = 0
    
    # We can try a few restarts if time permits.
    # But let's start with one robust greedy.
    # To improve density, we can prioritize edges that don't conflict with many existing edges?
    # Or just random.
    
    # Let's try 10 restarts with random edge order.
    # Each restart takes ~ 190 * O(n^2) = 190 * 400 = 76,000 ops.
    # 10 restarts = 760,000 ops. Very fast.
    # We can even do 100 restarts.
    
    for restart in range(50):
        if time.time() > DEADLINE - 5:
            break
            
        random.shuffle(all_edges)
        adj = [[None] * n for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if time.time() > DEADLINE - 5:
                break
                
            # Try to add edge (u, v) with a random color
            # We should try colors that are least likely to cause conflicts.
            # Or just pick a random color.
            # If we pick a color that is already used in a 4-cycle with (u,v), it fails.
            # Let's try all 20 colors and pick the first valid one?
            # Or random color?
            # Trying all colors is 20 * O(n^2) = 8000 ops per edge.
            # 190 edges * 8000 = 1.5M ops per restart.
            # 50 restarts = 75M ops. Might be too slow in Python.
            
            # Let's try random color first, and if it fails, try a few others?
            # Or just random color.
            
            # Better: Pick a random color from 0..19.
            c = random.randint(0, 19)
            
            if is_valid_addition(u, v, c):
                adj[u][v] = c
                adj[v][u] = c
                current_edges.append([u, v, c])
                
            # If it fails, we could try other colors, but that's expensive.
            # Let's just skip if random color fails.
            # This might result in fewer edges.
            # To improve, we can try 3 random colors.
            # Or, we can pick the color that minimizes conflicts?
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
        # If we found a very dense graph (e.g., > 100 edges), we might stop early?
        # No, let's use the time to try more restarts.
        
    # Return the best found
    # The task asks for [[u, v, c], ...]
    return best_edges

# EVOLVE-BLOCK-END
