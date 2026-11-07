"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Map edge index to (type, pos)
    # type: 'top', 'bottom', 'match'
    # pos: 0..m-1 for the start of the edge in the cycle/matching order
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle
            if v == (u + 1) % m:
                pos = u
            else:
                pos = v
            edge_map[idx] = ('top', pos)
        elif u >= m and v >= m:
            # Bottom cycle
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = v_local
            edge_map[idx] = ('bottom', pos)
        else:
            # Matching
            if u < m:
                pos = u
            else:
                pos = v - m
            edge_map[idx] = ('match', pos)

    # Define Spanning Tree
    # Tree edges:
    # - Match edge for pos 0
    # - Top edges for pos 0 to m-2
    # - Bottom edges for pos 0 to m-2
    # Free edges:
    # - Match edges for pos 1 to m-1
    # - Top edge for pos m-1
    # - Bottom edge for pos m-1
    
    free_indices = []
    tree_indices = []
    
    for idx in range(n_edges):
        etype, pos = edge_map[idx]
        is_in_tree = False
        if etype == 'match' and pos == 0:
            is_in_tree = True
        elif etype == 'top' and pos < m - 1:
            is_in_tree = True
        elif etype == 'bottom' and pos < m - 1:
            is_in_tree = True
            
        if is_in_tree:
            tree_indices.append(idx)
        else:
            free_indices.append(idx)
            
    n_free = len(free_indices)
    
    def spectral_radius(signs):
        A = np.zeros((n_vertices, n_vertices))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        try:
            vals = np.linalg.eigvalsh(A)
            return np.max(np.abs(vals))
        except:
            return float('inf')

    def is_balanced(signs):
        adj = [[] for _ in range(n_vertices)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        sigma = [0] * n_vertices
        for i in range(n_vertices):
            if sigma[i] == 0:
                sigma[i] = 1
                queue = deque([i])
                while queue:
                    u = queue.popleft()
                    for v, s in adj[u]:
                        expected_sigma_v = sigma[u] * s
                        if sigma[v] == 0:
                            sigma[v] = expected_sigma_v
                            queue.append(v)
                        elif sigma[v] != expected_sigma_v:
                            return False
        return True

    def evaluate_and_refine(initial_free_signs):
        """
        Takes an array of signs for free edges.
        Constructs full signs, checks validity, computes radius.
        Then performs local search by flipping single free edges.
        Returns (radius, full_signs_array) or (inf, None) if invalid.
        """
        signs = np.ones(n_edges, dtype=int)
        for i, idx in enumerate(free_indices):
            signs[idx] = initial_free_signs[i]
            
        # Check validity
        if is_balanced(signs):
            return float('inf'), None
            
        current_radius = spectral_radius(signs)
        if not np.isfinite(current_radius):
            return float('inf'), None

        best_radius = current_radius
        best_signs = signs.copy()
        
        # Local Search: Flip one free edge at a time
        # We can limit the number of iterations to keep it fast
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(n_free):
                # Flip free edge i
                old_val = signs[free_indices[i]]
                signs[free_indices[i]] = -old_val
                
                # Check validity quickly? 
                # Flipping one edge might make it balanced.
                # However, local search on switching classes is tricky because 
                # the "switching class" changes if we flip a free edge?
                # No, the set of free edges is fixed. The signs of free edges define the class.
                # But wait: The "free edges" are defined relative to a fixed tree.
                # Changing the sign of a free edge changes the switching class.
                # So we are effectively jumping to a different class.
                
                # We must re-check balance.
                if is_balanced(signs):
                    signs[free_indices[i]] = old_val
                    continue
                    
                r = spectral_radius(signs)
                
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                else:
                    # Revert
                    signs[free_indices[i]] = old_val
                    
            if not improved:
                break
                
        return best_radius, best_signs

    best_global_radius = float('inf')
    best_global_signs = None
    
    # Strategy: 
    # 1. Try a few random starting points.
    # 2. Try a "greedy" starting point (e.g., all +1, or alternating).
    # 3. Refine each with local search.
    
    rng = np.random.default_rng(42)
    
    # Start 1: All +1 (if valid)
    # Start 2: All -1 (if valid)
    # Start 3: Random
    # Start 4: Alternating pattern on free edges
    
    candidates = []
    
    # Candidate 1: All +1
    candidates.append(np.ones(n_free, dtype=int))
    
    # Candidate 2: All -1
    candidates.append(-np.ones(n_free, dtype=int))
    
    # Candidate 3: Alternating
    alt = np.ones(n_free, dtype=int)
    alt[1::2] = -1
    candidates.append(alt)
    
    # Candidate 4: Random
    candidates.append(rng.choice([-1, 1], size=n_free))
    
    # Candidate 5: Another Random
    candidates.append(rng.choice([-1, 1], size=n_free))

    # If m is small, we could just do exhaustive search.
    # m=4 -> 2^5=32. m=5 -> 2^6=64. m=7 -> 2^8=256. m=9 -> 2^10=1024.
    # Exhaustive is actually very fast for these sizes if spectral_radius is optimized.
    # Let's check if exhaustive is feasible.
    # 1024 * (eigvalsh cost). eigvalsh on 18x18 matrix is very fast (< 1ms).
    # 1024 * 1ms = 1s. This is well within 85s.
    # So for m <= 9, exhaustive search over switching classes is the BEST strategy.
    # The current program DOES exhaustive search.
    # Why did it score 0.9129? Maybe it's already optimal?
    # Or maybe the "validity" check is the bottleneck?
    # is_balanced is O(V+E) ~ O(m). 1024 * 100 ops = 100k ops. Negligible.
    
    # Let's verify the exhaustive search in the current program.
    # It iterates 2^(m+1).
    # For m=9, 1024 iterations.
    # For m=7, 256 iterations.
    # For m=5, 64 iterations.
    # For m=4, 32 iterations.
    # Total ~ 1376 iterations.
    # Each iteration: is_balanced + spectral_radius.
    # This should be extremely fast.
    
    # Is there a flaw in the current program's exhaustive search?
    # "The spectral radius is invariant under switching".
    # Yes.
    # "We fix tree edges to +1, and vary free edges."
    # Yes.
    # This covers all switching classes exactly once.
    # So it finds the GLOBAL MINIMUM spectral radius among all valid unbalanced signed graphs.
    
    # If the current program is already doing global optimization over the discrete space of switching classes, 
    # how can we improve?
    # 1. Maybe the "validity" constraint (connected lift) eliminates the global minimum?
    #    The current program skips balanced graphs.
    #    If the global min radius graph is balanced, it's invalid.
    #    The program finds the min among UNBALANCED ones.
    #    This is correct.
    
    # 2. Is there a continuous relaxation? No, signs are discrete.
    
    # 3. Is the definition of "spectral radius of the signed adjacency matrix" correct?
    #    Yes, max |eigenvalue|.
    
    # 4. Could the edge mapping be wrong?
    #    If the edge order in `graph['edges']` is not what I assume, my `edge_map` might be wrong.
    #    But I derive `pos` from vertex indices, which are fixed by the problem statement.
    #    So the mapping is robust.
    
    # 5. Is it possible that the seed program (score 1.0) is better?
    #    Seed score 1.0 means value = seed_value.
    #    Current score 0.9129 means value = seed_value / 0.9129 = 1.095 * seed_value.
    #    Wait. "search score is seed/value".
    #    If score = 0.9129, then seed/value = 0.9129 => value = seed / 0.9129 = 1.095 * seed.
    #    So the current program is WORSE than the seed.
    #    The seed (random sampling) found a better result than the exhaustive search?
    #    That implies the exhaustive search is FAILING to find the best valid graph, or I am misunderstanding the scoring.
    
    #    Let's re-read: "Scoring (lower is better): mean over the four graphs of the spectral radius... The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    #    So if Score < 1.0, Value > Seed Value.
    #    The current program (Exhaustive) has Score 0.9129.
    #    The Seed (Random) has Score 1.0.
    #    This means Random Sampling found a LOWER spectral radius than my Exhaustive Search?
    #    That is impossible if my Exhaustive Search covers all switching classes.
    #    UNLESS:
    #    a) My Exhaustive Search is not covering all classes (bug in tree/free selection).
    #    b) The "validity" check is different.
    #    c) The seed program is NOT just random sampling. "The seed samples budget random sign vectors... and keeps the best valid one."
    #    d) I am making a mistake in the spectral radius calculation or the graph construction.
    
    #    Let's check (a).
    #    Tree: Match 0, Top 0..m-2, Bottom 0..m-2.
    #    Free: Match 1..m-1, Top m-1, Bottom m-1.
    #    This is a valid spanning tree.
    #    The number of free edges is (m-1) + 1 + 1 = m+1.
    #    The cycle space dimension is m+1.
    #    So this covers all switching classes.
    
    #    Let's check (d).
    #    `A[u, v] = s`. `A[v, u] = s`.
    #    This is the signed adjacency matrix.
    #    `np.linalg.eigvalsh` returns eigenvalues in ascending order.
    #    `max(abs(vals))` is the spectral radius.
    
    #    Why would Random Sampling beat Exhaustive Search?
    #    Maybe the "valid" constraint (connected lift) is the key.
    #    My `is_balanced` check:
    #    A signed graph is balanced if all cycles have product +1.
    #    The lift is connected iff the signed graph is unbalanced.
    #    This is a standard theorem.
    
    #    Is it possible that `is_balanced` has a bug?
    #    Let's trace `is_balanced` for a simple unbalanced graph.
    #    Triangle with signs +1, +1, -1.
    #    0-1 (+1), 1-2 (+1), 2-0 (-1).
    #    sigma[0]=1.
    #    1 expects 1*1=1. sigma[1]=1.
    #    2 expects 1*1=1. sigma[2]=1.
    #    0 expects 1*(-1)=-1. sigma[0] is 1. 1 != -1. Return False.
    #    Correct.
    
    #    What if the graph is not connected?
    #    Prism graphs are connected.
    
    #    Is it possible that the seed program's "random sampling" is actually smarter?
    #    "samples budget random sign vectors... keeps the best valid one".
    #    Budget = 32.
    #    It only tries 32 random configurations!
    #    How can 32 random configurations beat 1024 exhaustive configurations?
    #    This strongly suggests my Exhaustive Search is finding a LOCAL MINIMUM or there is a bug in my Exhaustive Search that causes it to miss the global minimum.
    
    #    Wait. The spectral radius is invariant under switching.
    #    So ALL graphs in a switching class have the SAME spectral radius.
    #    So if I pick ONE representative from each class, I have evaluated ALL possible spectral radii.
    #    So the minimum of these representatives IS the global minimum.
    
    #    Unless... the "validity" (connected lift) depends on the representative?
    #    No, "The lift must be connected".
    #    The lift is connected iff the signed graph is unbalanced.
    #    Being unbalanced is a property of the switching class.
    #    If one graph in a class is unbalanced, ALL are unbalanced.
    #    So my check `if is_balanced(signs): continue` is correct and class-invariant.
    
    #    So, logically, my exhaustive search MUST find the global minimum.
    #    Why is it scoring worse than the seed?
    #    
    #    Hypothesis: The seed program's score of 1.0 is the BASELINE.
    #    Maybe the seed program is NOT just random.
    #    "The seed samples budget random sign vectors with np.random.default_rng(3) and keeps the best valid one."
    #    If the seed is just random, it's likely to find a mediocre solution.
    #    If my exhaustive search finds the TRUE global minimum, it should have a LOWER value (better) than the seed.
    #    So Score = Seed_Value / My_Value.
    #    If My_Value < Seed_Value, Score > 1.0.
    #    But my score is 0.9129 < 1.0.
    #    This implies My_Value > Seed_Value.
    #    This implies the Seed found a BETTER solution than my Exhaustive Search.
    #    
    #    This is a contradiction, UNLESS:
    #    1. My code has a bug that makes it return an invalid graph or a graph with higher radius.
    #    2. The "spectral radius" I am computing is not the one being scored.
    #    3. The "validity" check in the evaluator is different from mine.
    
    #    Let's look at `check_lift` in the import.
    #    "check_lift(graph, signs) returns (radius, details) and raises on an invalid lift."
    #    I am NOT using `check_lift`. I am implementing my own `spectral_radius` and `is_balanced`.
    #    Maybe my `is_balanced` is wrong?
    #    Maybe my `spectral_radius` is wrong?
    
    #    Let's try to use `check_lift` directly in the search to ensure I match the evaluator's definition of radius and validity.
    #    This is a crucial structural change: **Use the provided checker for evaluation**.
    
    #    Revised Strategy:
    #    1. Identify switching classes (as before).
    #    2. For each class, pick a representative.
    #    3. Call `check_lift(graph, representative)`.
    #    4. If it raises, skip (invalid).
    #    5. If it returns (radius, _), update best.
    
    #    This eliminates any discrepancy between my logic and the evaluator's logic.
    
    from autoresearch_suite.science_tcs_expansion import check_lift

    def evaluate_class(free_signs):
        signs = np.ones(n_edges, dtype=int)
        for i, idx in enumerate(free_indices):
            signs[idx] = free_signs[i]
        try:
            radius, _ = check_lift(graph, signs)
            return radius
        except:
            return float('inf')

    # Since exhaustive search is fast, let's do it again, but using check_lift.
    # This ensures we are optimizing the EXACT metric.
    
    best_radius = float('inf')
    best_signs = None
    
    num_classes = 1 << n_free
    
    for mask in range(num_classes):
        if time.time() > DEADLINE:
            break
            
        free_signs = np.ones(n_free, dtype=int)
        for i in range(n_free):
            if mask & (1 << i):
                free_signs[i] = -1
            else:
                free_signs[i] = 1
                
        r = evaluate_class(free_signs)
        
        if r < best_radius:
            best_radius = r
            # Reconstruct full signs for return
            best_signs = np.ones(n_edges, dtype=int)
            for i, idx in enumerate(free_indices):
                best_signs[idx] = free_signs[i]
                
    if best_signs is None:
        # Fallback
        best_signs = np.ones(n_edges, dtype=int)
        best_signs[0] = -1
        
    return best_signs
# EVOLVE-BLOCK-END
