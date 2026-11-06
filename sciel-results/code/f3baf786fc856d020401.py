"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    # 1. Generate all possible triples (indices 0..559)
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points.
    # We need to choose 6 distinct points, then partition them into 3 pairs.
    # Number of ways to choose 6 points: C(16, 6) = 8008
    # Number of ways to partition 6 points into 3 pairs: 15
    # Total Ks: 8008 * 15 = 120120
    
    # Hardcoding the 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # For each triple, we store a list of lists of the other 7 triple indices that complete a K with it.
    # triple_neighbors[idx] = [ [7 other indices], [7 other indices], ... ]
    # This allows us to check if any of these groups is fully present in our current set.
    triple_neighbors = [[] for _ in range(num_triples)]
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            t_indices = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        t_indices.append(triple_to_idx[t])
            
            # For each triple in this K, the other 7 form a neighbor group
            for ti in t_indices:
                triple_neighbors[ti].append([x for x in t_indices if x != ti])
    
    # Greedy Construction with local improvement
    best_triples = set()
    best_count = 0
    
    # We will run greedy passes.
    # To speed up the check, we maintain a count of how many of the "other 7" are present for each neighbor group?
    # That might be complex to update.
    # Alternative: Since the set is small (8), checking is fast.
    # Let's try to optimize the check loop.
    
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        current_triples = set()
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            # Check if adding idx creates a K(2,2,2)
            # A K is created if there is a neighbor group of idx where all 7 are in current_triples
            is_conflict = False
            for neighbors in triple_neighbors[idx]:
                # neighbors is a list of 7 indices
                # Check if all are in current_triples
                # Using a loop is faster than set intersection for small lists
                all_present = True
                for nb in neighbors:
                    if nb not in current_triples:
                        all_present = False
                        break
                if all_present:
                    is_conflict = True
                    break
            
            if not is_conflict:
                current_triples.add(idx)
                
        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_triples = set(current_triples)
            
    # Local improvement: try to add more triples
    # We can iterate through all triples not in best_triples and see if they can be added.
    # If adding one blocks nothing, add it.
    # We can repeat this process.
    
    # To make this efficient, we need a fast check.
    # Let's just do a few passes of "try to add any remaining triple".
    # This is O(N * D) per pass, where N is num triples, D is degree (num Ks containing triple).
    # N=560, D~?
    # Total Ks = 120120. Total incidences = 120120 * 8 = 960960.
    # Avg degree = 960960 / 560 = 1716.
    # So checking one triple is ~1716 * 7 ops.
    # Checking all 560 is ~1M ops.
    # This is fast enough for several passes.
    
    for _ in range(10):
        if time.time() > deadline:
            break
        changed = False
        # Shuffle to avoid bias
        remaining = [i for i in range(num_triples) if i not in best_triples]
        random.shuffle(remaining)
        for idx in remaining:
            if idx in best_triples: continue
            is_conflict = False
            for neighbors in triple_neighbors[idx]:
                all_present = True
                for nb in neighbors:
                    if nb not in best_triples:
                        all_present = False
                        break
                if all_present:
                    is_conflict = True
                    break
            if not is_conflict:
                best_triples.add(idx)
                changed = True
        if not changed:
            break
            
    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_triples]
    
    return result
# EVOLVE-BLOCK-END
