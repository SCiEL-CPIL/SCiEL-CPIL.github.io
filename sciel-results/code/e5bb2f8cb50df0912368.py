"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _neighbors(p):
    a, b = p
    return [
        (a+1, b), (a-1, b),
        (a, b+1), (a, b-1),
        (a+1, b-1), (a-1, b+1)
    ]

def _count_edges(points):
    if not points:
        return 0
    p_set = set(points)
    count = 0
    for p in points:
        for n in _neighbors(p):
            if n in p_set:
                count += 1
    return count // 2

def _perimeter(points):
    if not points:
        return 0
    p_set = set(points)
    count = 0
    for p in points:
        for n in _neighbors(p):
            if n not in p_set:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Beam Search for Maximum Edge Count (Minimum Perimeter)
    # State: frozenset of points
    # Score: -Perimeter (minimize perimeter) or Edge Count (maximize)
    
    # Initial state
    current_beam = {frozenset([(0, 0)]): 6} # Perimeter of 1 point is 6
    
    target_n = 32
    
    # We will grow the shape step by step
    for step in range(1, target_n):
        next_beam = {}
        
        # Check time
        if time.time() > deadline:
            break
        
        for state, perim in current_beam.items():
            # Generate candidates
            # To avoid duplicates and keep it fast, we only add points that are neighbors of the current set
            # and not already in the set.
            candidates = set()
            for p in state:
                for n in _neighbors(p):
                    if n not in state:
                        candidates.add(n)
            
            for c in candidates:
                new_state = state | {c}
                
                # Calculate new perimeter efficiently?
                # Perimeter change: 
                # If c has k neighbors in state, 
                # it removes k boundary edges from state and adds (6-k) boundary edges for c.
                # Delta P = (6-k) - k = 6 - 2k.
                # So New P = Old P + 6 - 2k.
                
                k = sum(1 for n in _neighbors(c) if n in state)
                new_perim = perim + 6 - 2*k
                
                # We want to minimize perimeter.
                # If this new state is already in next_beam, keep the one with lower perimeter.
                # However, two different states might have the same frozenset? No, frozenset is unique.
                # But we might generate the same frozenset from different parents? Yes.
                # So we check if it's in next_beam.
                
                if new_state in next_beam:
                    if new_perim < next_beam[new_state]:
                        next_beam[new_state] = new_perim
                else:
                    next_beam[new_state] = new_perim
        
        # Prune the beam
        if not next_beam:
            break
            
        # Sort by perimeter (ascending)
        # We keep the top W states
        W = 1000 # Beam width
        sorted_states = sorted(next_beam.items(), key=lambda x: x[1])
        
        # If we have fewer than W, keep all.
        if len(sorted_states) > W:
            sorted_states = sorted_states[:W]
            
        current_beam = dict(sorted_states)
        
        # Optimization: If we found a state with perimeter 20 (which gives E=76), 
        # we can stop early if we are confident it's optimal? 
        # Harborth says 76 is max. So P=20 is min.
        # If any state has P=20, it's optimal.
        min_p = min(p for _, p in current_beam)
        if min_p == 20:
            # We found an optimal shape. 
            # We can break and return it.
            # But we need to finish the loop to ensure we have 32 points? 
            # The loop runs to target_n. 
            # If we break early, we return a shape of size < 32. 
            # So we must continue to size 32, but we can prune more aggressively.
            # Actually, if P=20 at step < 32, adding points will increase P.
            # So we should just continue the beam search to reach n=32.
            pass

    # If the loop completed or was interrupted, pick the best state from current_beam
    # Note: If interrupted, current_beam might have states of size < 32.
    # We should check the size.
    best_state = None
    best_perim = float('inf')
    
    for state, perim in current_beam.items():
        if len(state) == target_n:
            if perim < best_perim:
                best_perim = perim
                best_state = state
    
    if best_state is None:
        # Fallback: If beam search failed to reach size 32 or was interrupted too early
        # Use a simple greedy fallback
        points = [(0,0)]
        p_set = {points[0]}
        for _ in range(1, target_n):
            if time.time() > deadline:
                break
            best_cand = None
            best_score = -1
            cands = set()
            for p in points:
                for n in _neighbors(p):
                    if n not in p_set:
                        cands.add(n)
            for c in cands:
                k = sum(1 for n in _neighbors(c) if n in p_set)
                if k > best_score:
                    best_score = k
                    best_cand = c
            if best_cand:
                points.append(best_cand)
                p_set.add(best_cand)
        best_state = frozenset(points)
        best_perim = _perimeter(best_state)

    # Convert to list of lists
    result = [[p[0], p[1]] for p in best_state]
    return result

# EVOLVE-BLOCK-END
