import time
from fractions import Fraction
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute subset sums for each x in [1,24]
    # For each x, candidates are {y in [1,24] : y > x}
    # We need to know if 1/x can be formed as a sum of reciprocals of a subset of candidates
    # Precompute all subset sums for each x
    # But the subset can be any subset of the OTHER selected elements, not just all candidates
    
    # Better approach: for a given set S and element x in S,
    # we need to check if 1/x is a subset sum of {1/y : y in S, y != x}
    # But we can restrict to y > x since if y < x, 1/y > 1/x, 
    # actually that's not quite right. Let me reconsider.
    
    # Actually, the problem says: "no reciprocal of a selected element may equal the sum 
    # of reciprocals of a subset of the OTHER selected elements"
    # So for x in S, we check if 1/x = sum(1/y for y in Y) where Y is a subset of S\{x}
    # Y can include elements both smaller and larger than x
    
    # However, if y < x, then 1/y > 1/x, so a single such y would already exceed the target.
    # But a subset could include multiple small elements whose reciprocals sum to 1/x.
    # Wait, if y < x, 1/y > 1/x. So any subset containing y < x would have sum >= 1/y > 1/x.
    # Therefore, we only need to consider candidates y > x!
    
    # This is a key insight: for checking if 1/x is a subset sum, we only need y > x.
    
    # Precompute for each x the set of all possible subset sums of {1/y : y in [x+1, 24]}
    # But we need to know which subsets sum to 1/x, and we need to track which elements are used
    
    # Let's precompute: for each x, all subset sums of {1/y : y in range(x+1, 25)}
    # and store which subsets achieve each sum
    
    # Actually, for efficiency, let's precompute for each x a set of achievable sums
    # and for each achievable sum, store one witness subset (or all witnesses)
    
    # But we need to check against the actual selected set S, not all possible candidates.
    # So precomputation of all subset sums of the full candidate pool helps, but we still
    # need to check if the witness subset is contained in S.
    
    # Alternative: precompute for each x, a dictionary mapping sum -> set of elements used
    # Then for a given S, check if any witness for 1/x is a subset of S\{x}
    
    # Let's do this: for each x in [1,24], compute all subset sums of {1/y : y in [x+1,24]}
    # and for each sum, store the set of elements that achieve it (we can store multiple witnesses)
    
    # For efficiency, let's store for each x and each possible sum, the set of elements
    # But the number of distinct sums might be large. Let's just store one witness per sum.
    
    # Actually, for the conflict check, we need: does there exist a subset Y of S\{x} 
    # such that sum(1/y for y in Y) = 1/x?
    # Given our insight, Y must be a subset of {y in S : y > x}
    
    # Precompute: for each x, all achievable subset sums of {1/y : y in [x+1,24]}
    # Map each sum to a witness set (one example)
    
    precomputed_witnesses = {}
    for x in range(1, 25):
        candidates = list(range(x + 1, 25))
        if not candidates:
            precomputed_witnesses[x] = {}
            continue
        
        # Generate all subset sums
        # Use DFS to find all achievable sums and one witness for each
        sums_map = {}  # sum -> witness set
        
        def gen_sums(idx, current_sum, current_set):
            if current_sum in sums_map:
                # We already have a witness; keep the smaller one for efficiency
                if len(current_set) < len(sums_map[current_sum]):
                    sums_map[current_sum] = set(current_set)
            else:
                sums_map[current_sum] = set(current_set)
            
            if idx == len(candidates):
                return
            
            # Skip
            gen_sums(idx + 1, current_sum, current_set)
            
            # Take
            y = candidates[idx]
            gen_sums(idx + 1, current_sum + Fraction(1, y), current_set | {y})
        
        gen_sums(0, Fraction(0), set())
        precomputed_witnesses[x] = sums_map
    
    def has_conflict(S, x):
        """Check if 1/x is a subset sum of {1/y : y in S, y > x}. Return witness or None."""
        target = Fraction(1, x)
        witnesses = precomputed_witnesses.get(x, {})
        if target not in witnesses:
            return None
        witness = witnesses[target]
        # Check if witness is a subset of S
        if witness.issubset(S):
            return witness
        return None
    
    def is_valid(S):
        for x in S:
            if has_conflict(S, x) is not None:
                return False
        return True
    
    def find_any_conflict(S):
        """Find any conflict in S. Return (x, witness) or (None, None)."""
        for x in sorted(S):
            witness = has_conflict(S, x)
            if witness is not None:
                return x, witness
        return None, None
    
    best_S = set()
    
    # Strategy 1: Greedy removal with multiple heuristics and restarts
    heuristics = [
        lambda cands: max(cands),  # Remove largest (smallest reciprocal)
        lambda cands: min(cands),  # Remove smallest (largest reciprocal)
        lambda cands: random.choice(list(cands)),
    ]
    
    for trial in range(30):
        if time.time() > deadline:
            break
        
        random.seed(trial)
        S = set(range(1, 25))
        
        # Greedy removal phase
        while True:
            x, witness = find_any_conflict(S)
            if x is None:
                break
            
            # Remove one element from {x} ∪ witness
            # Heuristic: try different strategies
            if trial % 3 == 0:
                z = max({x} | witness)
            elif trial % 3 == 1:
                z = min({x} | witness)
            else:
                z = random.choice(list({x} | witness))
            
            S.remove(z)
        
        # Local improvement: try to add back removed elements
        removed = set(range(1, 25)) - S
        for elem in sorted(removed, reverse=True):  # Try adding larger elements first (easier to satisfy)
            if time.time() > deadline:
                break
            S_candidate = S | {elem}
            if is_valid(S_candidate):
                S = S_candidate
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Strategy 2: Start from smaller sets and grow
    for start_size in range(10, 16):
        if time.time() > deadline:
            break
        
        for _ in range(10):
            if time.time() > deadline:
                break
            
            random.seed(random.randint(0, 10000))
            # Random initial set of size start_size
            S = set(random.sample(range(1, 25), start_size))
            
            # Remove conflicts
            while True:
                x, witness = find_any_conflict(S)
                if x is None:
                    break
                z = random.choice(list({x} | witness))
                S.remove(z)
            
            # Try to add elements
            removed = set(range(1, 25)) - S
            for elem in sorted(removed, reverse=True):
                if time.time() > deadline:
                    break
                S_candidate = S | {elem}
                if is_valid(S_candidate):
                    S = S_candidate
            
            if len(S) > len(best_S):
                best_S = set(S)
    
    # Final validation
    if not best_S or not is_valid(best_S):
        best_S = {24}
    
    return list(best_S)
