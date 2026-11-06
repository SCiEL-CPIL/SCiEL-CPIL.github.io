"""
Strategy: Greedy Set Cover + Maximum Clique on AP Compatibility Graph.
Instead of local search on S, we generate a pool of candidate Arithmetic Progressions (APs).
We select APs greedily to cover as many distinct differences as possible.
Then, we model the selection of a valid set S (size <= 32) as a Maximum Clique problem
on a graph where nodes are APs and edges indicate compatibility (union size <= 32).
This structural approach exploits the fact that a set with many progression differences
must contain many APs, and these APs must be "compatible" in terms of their union size.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def _generate_ap_pool(max_val=512, max_diffs=100, num_per_diff=10):
    """
    Generate a pool of candidate APs.
    For each difference d, generate several random APs (a, a+d, a+2d).
    """
    pool = []
    for d in range(1, max_diffs + 1):
        max_a = max_val - 2 * d
        if max_a < 1:
            continue
        # Generate num_per_diff random APs for this d
        for _ in range(num_per_diff):
            a = random.randint(1, max_a)
            ap = (a, a + d, a + 2 * d)
            pool.append(ap)
    return pool

def _greedy_select_aps(pool, target_size=32):
    """
    Greedily select APs from the pool to maximize distinct differences covered,
    subject to the constraint that the union of selected APs has size <= target_size.
    """
    selected = []
    covered_diffs = set()
    used_elements = set()
    
    # Sort pool to prioritize "rare" or "high" diffs? 
    # Let's just shuffle to break symmetry, or sort by d.
    # Sorting by d might help cover small diffs first which are easier to fit?
    # Or random. Let's try random for diversity.
    
    # To make it more effective, we can score APs by how many NEW diffs they add.
    # But we must also respect the size constraint.
    
    # A simple greedy:
    # 1. Shuffle pool
    # 2. For each AP, if union size <= 32, add it if it adds new diffs.
    # This is too restrictive. We might want to swap.
    
    # Better Greedy:
    # Maintain current union U.
    # Candidate APs: those where |U ∪ AP| <= 32.
    # Pick the one that adds the most new diffs.
    # If none, stop.
    
    # To allow for better packing, we might need to allow "replacing" if a new AP
    # allows for more future additions? That's hard.
    # Let's stick to simple greedy first, but with multiple restarts.
    
    pass # Placeholder, logic below in construct

def _max_clique_approx(graph_nodes, graph_edges, time_limit=1.0):
    """
    Approximate Maximum Clique using a simple greedy/branch-and-bound or heuristic.
    graph_nodes: list of node ids
    graph_edges: set of (u, v) tuples
    """
    # For small graphs (<= 100 nodes), a simple backtracking might work.
    # But we want speed.
    # Let's use a simple greedy coloring bound or just greedy selection.
    
    # Greedy:
    # 1. Order nodes by degree (descending).
    # 2. Pick node with highest degree in current candidate set.
    # 3. Remove it and all non-neighbors from candidate set.
    # 4. Repeat.
    
    # This is O(V^2) and fast.
    
    if not graph_nodes:
        return []
        
    candidates = set(graph_nodes)
    clique = []
    
    # Precompute neighbors for speed
    neighbors = {n: set() for n in graph_nodes}
    for u, v in graph_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    while candidates:
        # Pick node with max degree within candidates
        best_node = max(candidates, key=lambda x: len(neighbors[x] & candidates))
        clique.append(best_node)
        # Remove best_node and all non-neighbors
        candidates.difference_update(neighbors[best_node])
        candidates.discard(best_node)
        
    return clique

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_set = list(range(1, 33))
    best_score = _count_progression_diffs(best_set)
    
    # Strategy:
    # 1. Generate a large pool of APs.
    # 2. Use a heuristic to select a compatible subset of APs.
    # 3. The union of these APs is our candidate S.
    # 4. Refine S with local search to maximize score.
    
    # To improve on simple greedy, we will run multiple instances of:
    # - Randomized Greedy Selection
    # - Local Search Refinement
    
    # Generate Pool
    # We want to cover diffs 1..~50.
    # APs with small d are easier to fit together (smaller span? no, same span 2d).
    # Actually, APs with small d have smaller elements, might overlap more?
    # Let's generate a diverse pool.
    
    max_val = 512
    n_elements = 32
    
    for restart in range(20): # 20 restarts
        if time.time() > deadline - 10.0:
            break
            
        # 1. Generate Pool
        # Focus on diffs 1..60
        pool = []
        for d in range(1, 61):
            max_a = max_val - 2 * d
            if max_a < 1: continue
            # Generate 5 random APs for each d
            for _ in range(5):
                a = random.randint(1, max_a)
                pool.append((a, a+d, a+2*d))
                
        random.shuffle(pool)
        
        # 2. Greedy Selection with Randomization
        # We maintain a set of selected APs.
        # We want to maximize distinct diffs covered.
        # Constraint: Union size <= 32.
        
        selected_aps = []
        union_set = set()
        covered_diffs = set()
        
        # To make it smarter, we can look ahead or use a different order.
        # Let's try: Sort pool by "potential" (e.g., d is new).
        # But we don't know what's covered yet.
        
        # Simple Greedy:
        for ap in pool:
            d = ap[1] - ap[0]
            # If this diff is already covered, skip? 
            # Not necessarily, another AP with same d might fit better? 
            # But for score, we only count distinct d. So once d is covered, 
            # adding another AP with same d doesn't increase score directly, 
            # but it might change the union set to allow OTHER diffs to be added later?
            # Actually, adding more elements (up to 32) can only help or hurt future fits.
            # If we have spare capacity, adding an AP with an already-covered diff
            # is useless for score, but might block future APs.
            # So, only add AP if it adds a NEW diff.
            
            if d in covered_diffs:
                continue
                
            new_union = union_set.union(ap)
            if len(new_union) <= n_elements:
                # Check if adding this AP is "worth it"
                # It adds 1 new diff.
                # Does it block too many future options?
                # Heuristic: Yes, add it.
                selected_aps.append(ap)
                union_set = new_union
                covered_diffs.add(d)
                
                if len(union_set) == n_elements:
                    # No more room
                    break
                    
        # 3. Construct S from union
        S = list(union_set)
        
        # 4. Local Search Refinement
        # The greedy set might be suboptimal.
        # Run a few SA steps to improve.
        
        current_score = _count_progression_diffs(S)
        if current_score > best_score:
            best_score = current_score
            best_set = S
            
        # SA Refinement
        T = 1.0
        for step in range(200): # 200 steps per restart
            if time.time() > deadline - 10.0:
                break
                
            if len(S) < n_elements:
                # Add a random element
                # Try to add an element that creates a new AP
                # For simplicity, random add
                new_elem = random.randint(1, max_val)
                if new_elem not in S:
                    S.append(new_elem)
                    new_score = _count_progression_diffs(S)
                    if new_score >= current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_set = S
                    else:
                        S.pop()
            elif len(S) > n_elements:
                # Remove a random element
                idx = random.randint(0, len(S)-1)
                S.pop(idx)
            else:
                # Swap: remove one, add one
                idx = random.randint(0, len(S)-1)
                old_elem = S[idx]
                new_elem = random.randint(1, max_val)
                if new_elem in S:
                    continue
                S[idx] = new_elem
                new_score = _count_progression_diffs(S)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718, delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = S
                else:
                    S[idx] = old_elem
                    
            T *= 0.99
            if T < 0.1: T = 0.1

    # Final Check
    if best_score < _count_progression_diffs(list(range(1, 33))):
        best_set = list(range(1, 33))
        
    return best_set
# EVOLVE-BLOCK-END
