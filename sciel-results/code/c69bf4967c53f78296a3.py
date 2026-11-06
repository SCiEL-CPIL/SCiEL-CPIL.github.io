import time
import random
from itertools import combinations

def _base_colouring():
    """Return the 17-colour base: colour(i,j) = (i+j) mod 17 for 0<=i<j<16."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _edge_index(i, j):
    """Map vertex pair (i,j), i<j, to edge index in the 120-edge list."""
    # Edges are in lexicographic order: (0,1),(0,2),...,(0,15),(1,2),...,(14,15)
    # For a given i, edges starting at i are: (i,i+1),...,(i,15) -> 15-i edges
    # Offset for vertex i is: sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)/2
    offset = 15 * i - i * (i - 1) // 2
    return offset + (j - i - 1)

def _precompute_fours():
    """Precompute for each edge, the list of 4-sets (as frozensets of 4 vertices) that contain it."""
    # There are C(16,4) = 1820 four-sets
    # For each edge, which 4-sets contain both endpoints?
    # If edge is (i,j), a 4-set contains it iff it contains i and j and 2 other vertices from the remaining 14.
    # That's C(14,2) = 91 four-sets per edge.
    edge_to_fours = [[] for _ in range(120)]
    all_fours = list(combinations(range(16), 4))
    four_index = {f: idx for idx, f in enumerate(all_fours)}
    
    # For each edge, find all 4-sets containing it
    for i in range(16):
        for j in range(i + 1, 16):
            eidx = _edge_index(i, j)
            others = [v for v in range(16) if v != i and v != j]
            for a, b in combinations(others, 2):
                quad = tuple(sorted((i, j, a, b)))
                edge_to_fours[eidx].append(four_index[quad])
    return edge_to_fours, all_fours

def _colour_count(colours):
    """Return the number of distinct colours used."""
    return len(set(colours))

def _is_valid(colours, edge_to_fours, all_fours):
    """Check if every 4-set spans at least 5 distinct colours."""
    # For each 4-set, collect the colours of its 6 edges
    for quad_idx, quad in enumerate(all_fours):
        a, b, c, d = quad
        edge_indices = [
            _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
            _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
        ]
        distinct = set()
        for eidx in edge_indices:
            distinct.add(colours[eidx])
            if len(distinct) >= 5:
                break
        if len(distinct) < 5:
            return False
    return True

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    
    base = _base_colouring()
    best = list(base)
    best_count = _colour_count(best)
    
    edge_to_fours, all_fours = _precompute_fours()
    
    # Verify base is valid
    if not _is_valid(best, edge_to_fours, all_fours):
        return base
    
    # Try to reduce colour count by merging
    # Strategy: iteratively try to merge two colour classes and check validity
    # Use random restarts with local search
    
    random.seed(42)
    
    # For efficiency, we'll use a faster validation: when merging colours c1 and c2,
    # only need to check 4-sets where the merge could reduce the distinct colour count below 5.
    # A 4-set is at risk if it has at most 5 distinct colours and among them are both c1 and c2,
    # and removing one of them (by merging) drops the count to 4.
    
    # Let's try a simpler approach: random search for 16-colour solutions
    # Start from base, try to merge pairs of colours, check validity
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        current = list(base)
        cur_count = best_count
        
        # Try to merge colours to reduce count
        for step in range(200):
            if time.time() > deadline - 10:
                break
            
            if cur_count <= 14:
                break
            
            # Get list of colours in use
            colours_in_use = sorted(set(current))
            if len(colours_in_use) < 2:
                break
            
            # Pick a random pair of colours to try merging
            c1, c2 = random.sample(colours_in_use, 2)
            
            # Find edges with colour c2
            edges_c2 = [i for i in range(120) if current[i] == c2]
            if not edges_c2:
                continue
            
            # Try merging c2 into c1
            # Only valid if no 4-set drops below 5 colours
            # A 4-set is affected if it contains at least one edge of colour c2
            # and after merging, has < 5 distinct colours
            
            # Quick check: for each 4-set containing a c2-edge, check distinct colours after merge
            valid_merge = True
            affected_fours = set()
            for eidx in edges_c2:
                for fidx in edge_to_fours[eidx]:
                    affected_fours.add(fidx)
            
            # Check each affected 4-set
            for fidx in affected_fours:
                quad = all_fours[fidx]
                a, b, c, d = quad
                edge_indices = [
                    _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
                    _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
                ]
                # Simulate merge: c2 -> c1
                distinct = set()
                for eidx in edge_indices:
                    col = current[eidx]
                    if col == c2:
                        col = c1
                    distinct.add(col)
                if len(distinct) < 5:
                    valid_merge = False
                    break
            
            if valid_merge:
                # Perform the merge
                for i in range(120):
                    if current[i] == c2:
                        current[i] = c1
                cur_count -= 1
                
                if cur_count < best_count:
                    best = list(current)
                    best_count = cur_count
                    if best_count <= 13:
                        return best  # Can't do better than 13
            
            if time.time() > deadline - 5:
                break
    
    return best
