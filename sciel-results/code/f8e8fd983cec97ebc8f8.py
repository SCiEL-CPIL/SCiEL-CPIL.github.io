import time
import random
from itertools import combinations

DEADLINE = 145.0
START = time.time()

def _is_non_2colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    edges: list of lists of 5 vertices.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Precompute edge masks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute vertex to edge index mapping for heuristic?
    # Actually, for the heuristic, we need to know for each unassigned vertex,
    # how "tight" the edges containing it are.
    # We can compute this on the fly or precompute adjacency.
    # Given n <= 26 and m <= 200, on-the-fly might be okay, but adjacency is better.
    
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)

    def popcount(x):
        return bin(x).count('1')

    def propagate(mask0, mask1):
        """
        Returns (new_mask0, new_mask1, success)
        If success is False, contradiction found.
        """
        changed = True
        while changed:
            changed = False
            for e_mask in edge_masks:
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                c_un = 5 - c0 - c1
                
                if c0 == 5 or c1 == 5:
                    return 0, 0, False # Contradiction
                
                if c0 == 4 and c_un == 1:
                    # The unassigned vertex in this edge must be 1
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    # unassigned_mask should have exactly 1 bit set
                    new_mask1 = mask1 | unassigned_mask
                    if new_mask1 != mask1:
                        mask1 = new_mask1
                        changed = True
                elif c1 == 4 and c_un == 1:
                    # The unassigned vertex in this edge must be 0
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    new_mask0 = mask0 | unassigned_mask
                    if new_mask0 != mask0:
                        mask0 = new_mask0
                        changed = True
        return mask0, mask1, True

    def backtrack(mask0, mask1):
        # Find an unassigned vertex to branch on
        # Heuristic: Pick the vertex that is in the most "tight" edges.
        # Tightness: An edge with 4 assigned and 1 unassigned is very tight.
        # If we assign the last vertex, we either satisfy or violate it.
        
        best_v = -1
        best_score = -1
        
        # Iterate through all unassigned vertices
        # To find unassigned vertices: ~(mask0 | mask1) & ((1 << n) - 1)
        unassigned_all = ~(mask0 | mask1) & ((1 << n) - 1)
        
        while unassigned_all:
            # Get lowest set bit
            lsb = unassigned_all & -unassigned_all
            v = lsb.bit_length() - 1
            unassigned_all ^= lsb
            
            # Calculate score for this vertex
            # Score is sum of weights of edges containing v
            # Weight of an edge:
            # 5 unassigned: 0
            # 4 unassigned (1 assigned): 1
            # 3 unassigned (2 assigned): 2
            # 2 unassigned (3 assigned): 5
            # 1 unassigned (4 assigned): 10 (should be handled by propagation, but if not, it's critical)
            
            score = 0
            for ei in vert_edges[v]:
                e_mask = edge_masks[ei]
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                c_un = 5 - c0 - c1
                
                if c_un == 1:
                    score += 10
                elif c_un == 2:
                    score += 5
                elif c_un == 3:
                    score += 2
                elif c_un == 4:
                    score += 1
                # c_un == 5: 0
                
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            # No unassigned vertices in any edge?
            # This means all vertices involved in edges are assigned.
            # Check if any edge is monochromatic.
            for e_mask in edge_masks:
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                if c0 == 5 or c1 == 5:
                    return True # Contradiction, so NOT 2-colorable
            # If we reach here, we have a valid 2-coloring.
            return False # IS 2-colorable
        
        # Branch on best_v
        # Try color 0
        new_mask0 = mask0 | (1 << best_v)
        if new_mask0 & mask1:
            # Contradiction, skip
            pass
        else:
            nm0, nm1, ok = propagate(new_mask0, mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        # Try color 1
        new_mask1 = mask1 | (1 << best_v)
        if new_mask1 & mask0:
            pass
        else:
            nm0, nm1, ok = propagate(mask0, new_mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        return True # No valid coloring found in this branch

    # Initial propagation
    nm0, nm1, ok = propagate(0, 0)
    if not ok:
        return True
    
    return backtrack(nm0, nm1)

def _try_prune(seed, max_time=10.0):
    """
    Attempts to find a small non-2-colorable hypergraph using edge pruning.
    """
    rng = random.Random(seed)
    
    # Choose n
    # Smaller n has fewer total edges, but denser hypergraphs are non-2-colorable.
    # C(9,5)=126. C(10,5)=252.
    # Let's try n from 9 to 14.
    n = rng.randint(9, 14)
    
    # Generate a starting set of edges
    all_edges = list(combinations(range(n), 5))
    
    # Strategy: Start with all edges and prune.
    edges = [list(e) for e in all_edges]
    rng.shuffle(edges)
    
    # Pruning loop
    changed = True
    passes = 0
    start_time = time.time()
    
    while changed and time.time() - start_time < max_time:
        changed = False
        passes += 1
        # Shuffle order for this pass
        current_edges = edges[:]
        rng.shuffle(current_edges)
        
        for e in current_edges:
            if time.time() - start_time > max_time:
                break
            
            # Try to remove e
            edges_without_e = [ed for ed in edges if ed != e]
            
            # Check if still non-2-colorable
            if _is_non_2colorable(n, edges_without_e):
                # Yes, remove it
                edges = edges_without_e
                changed = True
    
    return n, len(edges), edges

def construct():
    best_edges = float('inf')
    best_result = None
    seed = 0
    
    # Fallback: Complete graph on 9 vertices
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    while time.time() < DEADLINE - 5:
        seed += 1
        # Allocate time for this attempt
        # If we have lots of time left, give more time to pruning.
        time_left = DEADLINE - time.time()
        if time_left < 10:
            break
            
        max_time_for_attempt = min(10.0, time_left * 0.5)
        
        try:
            n, count, edges = _try_prune(seed, max_time_for_attempt)
            if count < best_edges:
                best_edges = count
                best_result = {'n': n, 'edges': edges}
        except Exception:
            pass
            
    if best_result is None:
        return fallback
        
    return best_result
