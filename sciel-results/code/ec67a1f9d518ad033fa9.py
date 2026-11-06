"""New approach: Beam Search for two Sidon sets with disjoint differences."""
import time

def construct():
    n = 256
    max_size = 64
    beam_width = 100  # Number of states to keep
    
    # State: (A, B, sums_A, sums_B, diffs_A, diffs_B)
    # We use tuples for A and B to make them hashable if needed, but lists are fine for storage.
    
    def is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
        if len(A) >= max_size:
            return False
        # Sidon check
        if (2 * x) in sums_A:
            return False
        for a in A:
            if (a + x) in sums_A:
                return False
        # Disjoint difference check
        for a in A:
            d = x - a
            if d != 0 and d in diffs_B:
                return False
        return True

    def is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
        if len(B) >= max_size:
            return False
        # Sidon check
        if (2 * x) in sums_B:
            return False
        for b in B:
            if (b + x) in sums_B:
                return False
        # Disjoint difference check
        for b in B:
            d = x - b
            if d != 0 and d in diffs_A:
                return False
        return True

    def add_to_A(A, sums_A, diffs_A, x):
        new_A = A + [x]
        new_sums_A = set(sums_A)
        new_diffs_A = set(diffs_A)
        for a in A:
            new_sums_A.add(a + x)
            d = x - a
            if d != 0:
                new_diffs_A.add(d)
                new_diffs_A.add(-d)
        new_sums_A.add(2 * x)
        return new_A, new_sums_A, new_diffs_A

    def add_to_B(B, sums_B, diffs_B, x):
        new_B = B + [x]
        new_sums_B = set(sums_B)
        new_diffs_B = set(diffs_B)
        for b in B:
            new_sums_B.add(b + x)
            d = x - b
            if d != 0:
                new_diffs_B.add(d)
                new_diffs_B.add(-d)
        new_sums_B.add(2 * x)
        return new_B, new_sums_B, new_diffs_B

    # Initial state
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    beam = [(A, B, sums_A, sums_B, diffs_A, diffs_B)]
    
    start_time = time.time()
    
    for x in range(1, n + 1):
        if time.time() - start_time > 55:
            break
            
        new_beam = []
        
        for (A, B, sums_A, sums_B, diffs_A, diffs_B) in beam:
            # Option 1: Skip x
            new_beam.append((A, B, sums_A, sums_B, diffs_A, diffs_B))
            
            # Option 2: Add x to A
            if is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
                nA, nSumsA, nDiffsA = add_to_A(A, sums_A, diffs_A, x)
                new_beam.append((nA, B, nSumsA, sums_B, nDiffsA, diffs_B))
            
            # Option 3: Add x to B
            if is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
                nB, nSumsB, nDiffsB = add_to_B(B, sums_B, diffs_B, x)
                new_beam.append((A, nB, sums_A, nSumsB, diffs_A, nDiffsB))
            
            # Option 4: Add x to Both
            # Check if valid for A first
            if is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
                nA, nSumsA, nDiffsA = add_to_A(A, sums_A, diffs_A, x)
                # Now check if valid for B, considering the new diffs in A?
                # Wait, if x is in A, the diffs in A include those involving x.
                # The check for B must ensure B's new diffs don't conflict with A's *current* diffs (which now include x's).
                # is_valid_add_B checks against diffs_A.
                # So we must use nDiffsA for the check against B.
                if is_valid_add_B(B, sums_B, nDiffsA, diffs_B, x):
                    nB, nSumsB, nDiffsB = add_to_B(B, sums_B, diffs_B, x)
                    new_beam.append((nA, nB, nSumsA, nSumsB, nDiffsA, nDiffsB))
        
        # Prune beam: Keep top beam_width states by score (len(A) + len(B))
        # To avoid keeping identical states, we can use a set of tuples of frozensets?
        # But A and B are lists.
        # We can just sort by score and take top K.
        # If there are many ties, we might lose diversity.
        # Let's shuffle the new_beam before sorting to break ties randomly.
        import random
        random.shuffle(new_beam)
        
        new_beam.sort(key=lambda s: len(s[0]) + len(s[1]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Optimization: If all states in beam have max size, we can break?
        # Or if the best score in beam is the same as previous best, we might be stuck.
        
    # Get the best state from the final beam
    if beam:
        best_state = max(beam, key=lambda s: len(s[0]) + len(s[1]))
        return [best_state[0], best_state[1]]
    else:
        return [[1], [2]] # Fallback

