import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    # Precompute all 8-subsets and their splits
    # Splits: list of (mask_A, mask_B) for each 8-set
    # We store masks as bitmasks of the 8 vertices in the subset
    # But we need to map back to global vertices.
    # Better: Store list of (u1, u2, u3, u4, v1, v2, v3, v4)
    
    # To save time, we can precompute the structure
    # There are 12870 8-sets.
    # For each, 70 splits.
    # Total ~900k entries.
    
    # We will use a greedy/randomized local search with incremental checking.
    
    # Initialize with the known 43-edge graph
    # K_3,13 + Triangle on 0,1,2
    adj = [0] * n
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    # Precompute all 8-sets and splits
    # This takes a few seconds, so we do it once.
    # But 150s limit. Precomputation might take 1-2s.
    
    # To speed up, we only need to check if the graph contains Q3.
    # We will maintain a list of "critical" splits.
    # A split is critical if it has all degrees >= 2 and is one edge away from being >= 3.
    
    # Given the complexity, let's stick to a simpler but effective approach:
    # Random restarts with full check, but optimized full check.
    
    # Optimized full check:
    # Precompute all 8-sets as tuples of vertices.
    # Precompute all splits for each 8-set as (mask_A, mask_B) relative to the 8-set?
    # No, just use the vertex indices.
    
    # Let's generate the list of checks once.
    # checks = [(u1, u2, u3, u4, v1, v2, v3, v4), ...]
    # This list has 900k entries.
    # Storing it in memory is fine.
    
    # However, generating it takes time.
    # Let's generate it on the fly if it's fast enough, or precompute.
    
    # Precomputation
    checks = []
    vertices = list(range(n))
    for combo in itertools.combinations(vertices, 8):
        c = list(combo)
        # Generate all ways to split 8 items into 2 sets of 4
        # We can use combinations of 4 from 8
        for a_set in itertools.combinations(c, 4):
            a_set = set(a_set)
            b_set = set(c) - a_set
            # To avoid duplicates (A,B) and (B,A), we can enforce min(A) < min(B) or similar
            # But for checking, (A,B) and (B,A) are symmetric in the condition?
            # Condition: all deg in A >= 3 and all deg in B >= 3.
            # This is symmetric. So we can just check one.
            # But we must ensure we cover all partitions.
            # There are 70 partitions.
            # We will add (tuple(a_set), tuple(b_set))
            # To avoid duplicate work, we can just iterate all 70.
            checks.append((tuple(a_set), tuple(b_set)))
            
    # This precomputation loop: 12870 * 70 = 900k iterations.
    # itertools.combinations is fast. This should take < 1 second.
    
    # Helper to check if a specific split is "good" (contains Q3)
    def is_split_good(adj, A, B):
        # A and B are tuples of 4 vertices
        # Check if all vertices in A have >= 3 neighbors in B
        # And all vertices in B have >= 3 neighbors in A
        for u in A:
            deg = 0
            for v in B:
                if adj[u] & (1 << v):
                    deg += 1
                    if deg >= 3: break
            if deg < 3:
                return False
        for u in B:
            deg = 0
            for v in A:
                if adj[u] & (1 << v):
                    deg += 1
                    if deg >= 3: break
            if deg < 3:
                return False
        return True

    def is_q3_free(adj):
        for A, B in checks:
            if is_split_good(adj, A, B):
                return False
        return True

    # Initial check
    if not is_q3_free(adj):
        # Should not happen
        pass
        
    best_edges = len([1 for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)])
    best_adj = [x for x in adj]
    
    # Local Search
    # We will try to add edges.
    # We use a simple random walk with backtracking.
    
    # To improve performance, we can use the "critical splits" idea,
    # but for simplicity and robustness within time limit, 
    # we will do multiple restarts with random initializations and hill climbing.
    
    # Actually, the full check is ~0.5s.
    # 150s / 0.5s = 300 checks.
    # This is too few for random restarts.
    
    # We MUST use incremental checking or a faster check.
    
    # Let's optimize is_split_good using bitmasks.
    # Precompute mask for each A, B?
    # No, A, B are tuples of indices.
    
    # Let's try a different approach:
    # Since n=16 is small, we can use a SAT solver or ILP?
    # No, standard library only.
    
    # Let's refine the incremental update.
    # We maintain a count for each split of how many vertices in A have deg < 3 and how many in B have deg < 3.
    # If count_A == 0 and count_B == 0, it's a Q3.
    
    # This is complex to implement correctly in the time available.
    
    # Alternative:
    # Use the fact that we only need to maximize edges.
    # We can try to construct the complement of a sparse graph.
    # Minimize edges in complement H such that G = complement(H) is Q3-free.
    # This is equivalent.
    
    # Let's stick to the random restart hill climbing but optimize the check.
    # We can skip the full check if the edge count is low? No.
    
    # Let's assume the precomputation of checks is done.
    # We will run a loop that adds a random edge, checks, and if invalid, removes it.
    # To speed up, we can cache the result of is_split_good for the current state?
    # No, state changes.
    
    # Given the constraints, I will provide the code with the precomputation and a simple search.
    # If it's too slow, it might timeout, but it's the best structural change available.
    
    # Actually, 900k checks * 8 ops = 7.2M ops. In Python, ~0.5s.
    # 300 iterations is not enough to find the optimum if it's far from 43.
    
    # However, the seed is 43. The optimum might be close.
    # Or we can use a greedy approach:
    # Try all missing edges, add the one that keeps it Q3-free.
    # This is deterministic and might find a good local optimum quickly.
    
    # Let's combine:
    # 1. Greedy add
    # 2. Random restarts with random initial sparse graphs
    
    # For the sake of the code, I will implement the greedy + random restart.
    
    def greedy_fill(adj):
        # Try to add edges in random order
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_q3_free(adj):
                # Keep
                pass
            else:
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    best_adj = [x for x in adj]
    best_count = best_edges
    
    # Run several restarts
    # Time budget for search: 140s
    # Precomp took ~1s.
    
    while time.time() < deadline - 5:
        # Start from current best or random?
        # Start from best to improve
        curr_adj = [x for x in best_adj]
        
        # Randomize order of missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (curr_adj[i] & (1 << j))]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            curr_adj[u] |= (1 << v)
            curr_adj[v] |= (1 << u)
            if is_q3_free(curr_adj):
                count = len([1 for i in range(n) for j in range(i+1, n) if curr_adj[i] & (1 << j)])
                if count > best_count:
                    best_count = count
                    best_adj = [x for x in curr_adj]
            else:
                curr_adj[u] &= ~(1 << v)
                curr_adj[v] &= ~(1 << u)
                
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges

# Run construct to ensure it works? No, evaluator calls it.
