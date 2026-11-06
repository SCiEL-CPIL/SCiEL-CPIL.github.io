# EVOLVE-BLOCK-START
import time
import heapq
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search_fold(sequence, dimension, width, time_limit, start_pos=(0,0,0)):
    n = len(sequence)
    if n == 0:
        return []
    
    # Each state in the beam is:
    # (score, chain, occupied_set, last_pos)
    # Score is the number of contacts found so far (we want to maximize, so we use negative for min-heap)
    
    # Precompute neighbors for speed? No, chain grows.
    
    # Initial state
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    # Score: 0 contacts so far.
    # We use a min-heap, so we push (-contacts, chain, occupied)
    # But we need to break ties to avoid huge memory usage or bias.
    # Let's push ( -contacts, random_breaker, chain, occupied )
    
    # To keep memory low, we store chain as a list of tuples.
    # Occupied as a set of tuples.
    
    # Optimization: We only keep the best 'width' states at each step.
    
    # For speed, we will not use random breaker initially, but it helps diversity.
    # However, deterministic beam search is often more stable.
    # Let's add a small random component to tie-breaking to explore different branches.
    # Actually, standard beam search keeps the top K.
    
    # To make it deterministic and reproducible, we can sort by (-contacts, chain)
    # But chain comparison is expensive.
    # Let's use a counter or just rely on the heap properties if we push unique tuples.
    # Since chain is a list, it's not hashable. We need to be careful.
    
    # Let's store:
    # ( -contacts, index, chain, occupied )
    # where index is a unique id for tie-breaking.
    
    counter = 0
    beam = []
    
    # Start
    heapq.heappush(beam, (0, counter, [start_pos], {start_pos}))
    counter += 1
    
    start_time = time.time()
    
    for step in range(1, n):
        if time.time() > time_limit:
            break
            
        new_beam = []
        
        # Expand all states in the current beam
        # To avoid expanding too many, we only expand the top 'width' states
        # We pop from the heap
        
        states_to_expand = []
        while beam and len(states_to_expand) < width:
            states_to_expand.append(heapq.heappop(beam))
            
        # If beam is empty, we failed
        if not states_to_expand:
            break
            
        for score, _, chain, occupied in states_to_expand:
            last = chain[-1]
            
            # Generate neighbors
            for nb in _neighbours(last):
                if nb not in occupied:
                    new_chain = chain + [nb]
                    new_occupied = occupied | {nb}
                    
                    # Calculate new contacts
                    # Only need to check if the new residue 'H' at 'nb' makes contacts
                    # with existing 'H's in the chain.
                    # Since we only add one residue, we just check its neighbors.
                    new_contacts = 0
                    if sequence[step] == 'H':
                        for q in _neighbours(nb):
                            if q in occupied:
                                # Find index of q
                                # We don't have a map from pos to index in 'occupied'
                                # We need to check if the residue at q is H.
                                # We can store the sequence index in the occupied set?
                                # Or just check all occupied neighbors.
                                # Since 'occupied' is a set of positions, we need to know which index it is.
                                # This is a bottleneck.
                                # Let's store (position, index) in occupied? No, set of positions is faster for lookup.
                                # We can just iterate through the chain? No, too slow.
                                # We can maintain a dict pos -> index.
                                
                                # Let's restructure:
                                # State: (score, chain, pos_to_index_dict)
                                # But dict is not hashable for heap? We don't push dict.
                                # We can just check neighbors.
                                pass
                    
                    # This incremental energy calculation is tricky with just a set.
                    # Let's optimize:
                    # When adding new point P at index i:
                    # If seq[i] == 'H':
                    #   For each neighbor Q of P:
                    #     If Q is in the chain (i.e., in occupied):
                    #       Let j be the index of Q.
                    #       If seq[j] == 'H' and j != i-1 (not bonded):
                    #         contact += 1
                    
                    # To do this fast, we need to know the index of Q.
                    # We can store the chain as a list, and for lookup, we can use a dict.
                    # But creating a new dict for every expansion is O(N).
                    # Total cost O(N^2 * Width). For N=136, Width=100, this might be slow.
                    
                    # Alternative:
                    # Store the chain.
                    # For the new point, check its 6 neighbors.
                    # For each neighbor in the chain, we need to know if it's H.
                    # We can just check the sequence at the index of that neighbor.
                    # How to get index?
                    # We can maintain a dictionary `pos_idx` alongside the chain.
                    # But we can't easily push a dict into a heap if we need to compare states.
                    # We only compare by score.
                    
                    # Let's try:
                    # State: ( -contacts, chain, pos_idx_dict )
                    # We don't compare chains.
                    
                    new_pos_idx = dict(occupied) # This is O(N) copy
                    # Wait, 'occupied' was a set. We need a dict.
                    
                    # Let's change the state representation.
                    # State: ( -contacts, chain, pos_to_idx )
                    
                    # Let's restart the loop logic in my head to be cleaner.
                    pass

        # The above approach with dict copying is O(N) per expansion.
        # With width W and N steps, total O(W * N^2).
        # For N=136, W=50, 50 * 18000 = 900,000 ops. Very fast.
        
        # Let's implement this properly.
        
        # Re-initialize beam with proper state structure
        # We'll clear and redo the loop properly below.
        
        # Actually, let's just rewrite the function body to be clean.
        pass
        
    # Fallback if time runs out or beam empties
    # Return the best chain found so far?
    # The beam holds the best candidates.
    # If we break early, we should return the best one in the beam.
    
    if beam:
        best_score, _, best_chain, _ = heapq.heappop(beam)
        return best_chain
    elif states_to_expand:
        best_score, _, best_chain, _ = heapq.heappop(states_to_expand)
        return best_chain
    else:
        return [start_pos]

def _beam_search_fold_v2(sequence, dimension, width, time_limit):
    n = len(sequence)
    if n == 0:
        return []
    
    start_pos = (0, 0, 0)
    
    # State: ( -contacts, chain, pos_to_idx )
    # We use a min-heap on -contacts (so max contacts is on top)
    # Tie-breaking: we don't care much, but heap needs comparable items.
    # ( -contacts, chain_id, chain, pos_to_idx )
    
    counter = 0
    beam = []
    
    initial_chain = [start_pos]
    initial_pos_idx = {start_pos: 0}
    heapq.heappush(beam, (0, counter, initial_chain, initial_pos_idx))
    counter += 1
    
    start_time = time.time()
    best_chain = initial_chain
    
    for step in range(1, n):
        if time.time() > time_limit:
            break
            
        new_beam = []
        
        # Get top 'width' states from current beam
        # Note: beam contains states from previous step
        # We need to expand them
        
        states_to_expand = []
        while beam and len(states_to_expand) < width:
            states_to_expand.append(heapq.heappop(beam))
            
        if not states_to_expand:
            break
            
        for neg_score, _, chain, pos_idx in states_to_expand:
            last = chain[-1]
            current_contacts = -neg_score
            
            for nb in _neighbours(last):
                if nb not in pos_idx:
                    new_chain = chain + [nb]
                    new_pos_idx = pos_idx.copy()
                    new_pos_idx[nb] = step
                    
                    new_contacts = current_contacts
                    
                    if sequence[step] == 'H':
                        for q in _neighbours(nb):
                            if q in new_pos_idx:
                                idx_q = new_pos_idx[q]
                                # Check if it's a valid contact (not bonded)
                                # Bonded if idx_q == step - 1
                                if idx_q != step - 1 and sequence[idx_q] == 'H':
                                    new_contacts += 1
                                    
                    heapq.heappush(new_beam, (-new_contacts, counter, new_chain, new_pos_idx))
                    counter += 1
                    
                    if len(new_beam) > width:
                        heapq.heappop(new_beam)
                        
        beam = new_beam
        
        if not beam:
            break
            
        # Update best chain found so far (the top of the heap)
        # Actually, the top of the heap is the best for this step.
        # But a later step might have a worse score? No, contacts only increase.
        # So the best chain at step N is the best overall.
        # But if we break early, we return the best chain from the last full step.
        
    if beam:
        best_neg_score, _, best_chain, _ = heapq.heappop(beam)
        return best_chain
    else:
        # If beam is empty, return the last valid chain we had?
        # We lost track of it if we didn't store it.
        # Let's store the best chain seen so far outside the loop.
        # Actually, if beam is empty, it means no valid moves were found.
        # We should return the longest valid chain we can find.
        # For now, return initial chain or a straight line.
        return [(i, 0, 0) for i in range(n)]

def fold(sequence, dimension):
    """
    Uses Multi-Start Beam Search with incremental contact scoring.
    """
    n = len(sequence)
    
    # Global time management
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 330.0
    # Leave a safety margin
    call_deadline = global_deadline - 5.0
    
    # Allocate time per sequence
    # We don't know how many sequences are left.
    # Let's just use a fixed time per call, or dynamic?
    # The prompt says "240 s CPU budget for all sequences together".
    # So we should track total time.
    
    if not hasattr(fold, '_total_time_used'):
        fold._total_time_used = 0.0
        
    # Estimate remaining sequences?
    # We can't.
    # Let's assume a fixed budget per sequence, e.g., 10 seconds?
    # 26 sequences * 10s = 260s. Safe.
    # But if some are fast, we waste time.
    # If some are slow, we might timeout.
    
    # Better: use a fixed time per call, e.g., 10 seconds.
    # If we finish early, great.
    
    time_per_call = 10.0
    if time.time() + time_per_call > call_deadline:
        time_per_call = max(1.0, call_deadline - time.time())
        
    start_time = time.time()
    
    # Beam width
    # For short chains, wider beam. For long, narrower?
    # Let's try width = 50.
    width = 50
    
    # Multi-start: try different starting directions?
    # Or just one beam search with sufficient width?
    # Beam search is deterministic if we don't randomize tie-breaking.
    # To get diversity, we can run multiple beam searches with different tie-breakers?
    # Or just one strong beam search.
    
    # Let's try 2 restarts with different tie-breaking or just 1.
    # Given time limit, 1 is safer.
    
    best_chain = None
    best_energy = 0
    
    # Run beam search
    chain = _beam_search_fold_v2(sequence, dimension, width, start_time + time_per_call)
    
    if chain:
        best_chain = chain
        best_energy = _energy(sequence, chain)
        
    # If time remains, try another beam search with different parameters?
    # Or local optimization?
    # Beam search is already quite good.
    
    # Let's add a simple local optimization on the result?
    # No, beam search builds it up.
    
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
