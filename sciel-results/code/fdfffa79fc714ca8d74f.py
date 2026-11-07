"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        # Check if any simple path s-u-v-t has deg(s) == deg(t)
        # s, u, v, t must be distinct
        # Path is s-u, u-v, v-t
        
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # For each edge (u, v), it can be the middle edge of a path s-u-v-t
        # s must be in adj[u] \ {v}
        # t must be in adj[v] \ {u}
        # We need to ensure deg(s) != deg(t) for all such pairs
        
        for u in range(n):
            for v in adj[u]:
                # s is a neighbor of u, not v
                # t is a neighbor of v, not u
                # If there exists s in N(u)\{v} and t in N(v)\{u} such that deg(s) == deg(t), it's invalid.
                
                # Optimization: Group neighbors by degree
                # Degrees of s: {deg(s) for s in N(u) if s != v}
                # Degrees of t: {deg(t) for t in N(v) if t != u}
                # If the intersection of these two sets is non-empty, invalid.
                
                deg_s_set = set()
                for s in adj[u]:
                    if s != v:
                        deg_s_set.add(deg[s])
                        
                # If deg_s_set is empty, no path s-u-v-t exists for this u,v
                if not deg_s_set:
                    continue
                    
                for t in adj[v]:
                    if t != u:
                        if deg[t] in deg_s_set:
                            return False
        return True

    # Initialize with a random graph (start sparse to be likely valid, then grow)
    # Or start with a star which is known valid?
    # Star: 14 edges.
    # Let's try starting with a random subset and growing.
    # Actually, let's start with a known valid small graph and add edges greedily/randomly.
    
    # Strategy:
    # 1. Start with a star (center 0). This is valid.
    #    Why? Path s-0-v-t. s is a leaf (deg 1). v is a leaf (deg 1).
    #    Wait, in a star K_1,14, leaves have degree 1.
    #    Path s-0-v-t: s is leaf, 0 is center, v is leaf, t is leaf?
    #    No, v is a neighbor of 0. t is a neighbor of v.
    #    In a star, v is a leaf, so N(v) = {0}. So t must be 0?
    #    But s != t and s,u,v,t distinct.
    #    If v is a leaf, N(v) = {0}. So t=0.
    #    Path s-0-v-0? No, s-u-v-t is a path of 3 edges.
    #    Vertices s, u, v, t distinct.
    #    In star, if u=0, v=leaf. N(v)={0}. So t must be 0.
    #    But u=0 and t=0 implies u=t, which violates distinctness.
    #    So no simple path of length 3 exists in a star?
    #    Let's check: s-0-v-? v only connects to 0. So the path ends.
    #    Yes, star has no simple path of length 3. So it is valid.
    #    Edge count: 14.
    
    current_edges = [(0, i) for i in range(1, n)]
    current_valid = True
    
    # Try to add edges
    # Randomize the order of remaining edges to escape local optima
    remaining_edges = [e for e in all_edges if e not in current_edges]
    
    best_edges = current_edges[:]
    best_count = len(best_edges)
    
    # Greedy Add with Randomization
    # We will try to add edges. If adding an edge makes it invalid, we skip it (or maybe try to remove another to fix?).
    # Simpler: Random restarts with Greedy Add.
    
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize the set of candidate edges to add
        # Start from star or random valid graph?
        # Let's start from star every time but shuffle the order of attempts.
        # Actually, starting from star is deterministic.
        # Let's try starting from a random valid graph?
        # Hard to generate random valid graph easily.
        
        # Alternative: Start with star, shuffle remaining edges, add if valid.
        # Then maybe remove some and add others?
        
        # Let's try: Start with star. Shuffle remaining edges.
        # For each edge in shuffled remaining:
        #   Try to add. If valid, keep it.
        
        edges_to_try = remaining_edges[:]
        random.shuffle(edges_to_try)
        
        cur = current_edges[:]
        
        for e in edges_to_try:
            if time.time() > deadline:
                break
            new_edges = cur + [e]
            if is_valid(new_edges):
                cur = new_edges
        
        if len(cur) > best_count:
            best_count = len(cur)
            best_edges = cur[:]
            
        # Local Search: Try to swap edges
        # If we are stuck, try removing a random edge and adding two others?
        # Or just simple hill climbing on swaps.
        
        # Let's do a few iterations of "remove one, add one" or "add one, remove one"
        # Actually, just "add one" is monotonic if we start from valid.
        # But we might get stuck in a local maximum where no single edge addition is valid.
        
        # Try 100 random swaps (remove e1, add e2)
        for _ in range(100):
            if time.time() > deadline:
                break
            if not cur:
                break
            e1 = random.choice(cur)
            # Find an edge not in cur
            # This is slow if we pick randomly from all_edges
            # Pick a random edge from remaining or just try to add a random missing edge
            # Let's pick a random missing edge
            # To speed up, keep a set of current edges
            cur_set = set(tuple(e) for e in cur)
            # Find a missing edge
            missing = [e for e in all_edges if tuple(e) not in cur_set]
            if not missing:
                break
            e2 = random.choice(missing)
            
            new_edges = [e for e in cur if e != e1]
            new_edges.append(e2)
            
            if is_valid(new_edges):
                # Accept if it doesn't reduce size (it doesn't, swap)
                # But maybe we can improve? No, size is same.
                # So swap is only useful if it allows future additions.
                # This is simulated annealing territory.
                # Let's just accept if valid, to explore.
                cur = new_edges
                
                # After swap, try to add more edges
                # Re-shuffle remaining and add
                cur_set = set(tuple(e) for e in cur)
                missing = [e for e in all_edges if tuple(e) not in cur_set]
                random.shuffle(missing)
                for e in missing:
                    if time.time() > deadline:
                        break
                    test_edges = cur + [e]
                    if is_valid(test_edges):
                        cur = test_edges
                        cur_set.add(tuple(e))
                        
        if len(cur) > best_count:
            best_count = len(cur)
            best_edges = cur[:]

    # Return best found
    # Ensure types are correct: list of lists of ints
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
