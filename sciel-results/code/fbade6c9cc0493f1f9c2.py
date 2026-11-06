"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _precompute_conflicts(N=180):
    """For each pair (b, c) with b < c, find if there exists a distinct from b,c
    such that 1/a = 1/b + 1/c, i.e., a = bc/(b+c).
    Store conflicts as: for each element x, the set of pairs (i,j) that would
    conflict if x is added.
    
    Actually, let's think differently. A triple (a,b,c) with a<b<c satisfies
    1/a = 1/b + 1/c iff a = bc/(b+c). So for any pair (b,c) with b<c, if bc/(b+c)
    is an integer a < b, then {a,b,c} is a forbidden triple.
    
    We'll precompute: for each candidate element x, which existing pairs (i,j) 
    in our set would form a triple with x.
    """
    # For each x in [1,N], find all pairs (b,c) with b<c, b,c != x, such that
    # x = bc/(b+c), i.e., b*c = x*(b+c), i.e., (x-b)(x-c) = x^2
    # Or equivalently: for each x, and each b in [1,N] with b != x,
    # c = xb/(x-b) must be integer, c != x, c != b, 1 <= c <= N
    # 
    # Let's precompute for each x the set of (b,c) pairs that conflict with x.
    # But that's too much. Instead, let's precompute all valid triples.
    
    triples = []
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b and a != c:  # a < b < c, all distinct
                    triples.append((a, b, c))
    
    return triples

def _check_valid(triples, S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 180
    triples = _precompute_conflicts(N)
    
    # Start with the known good base: [91, 180]
    S = list(range(91, N + 1))
    
    # Try to add elements from 1 to 90 in order of "least conflict"
    # For each candidate x, count how many triples it would create with current S
    candidates = list(range(1, 91))
    
    # Greedy: repeatedly add the candidate that creates the fewest new conflicts
    # (i.e., conflicts with the fewest existing elements in S)
    while time.time() < deadline:
        # For each candidate not in S, compute how many elements in S it conflicts with
        # A candidate x conflicts with an element y in S if there exists z in S
        # such that {x,y,z} is a forbidden triple.
        
        S_set = set(S)
        best_candidate = None
        best_conflict_count = float('inf')
        
        for x in candidates:
            if x in S_set:
                continue
            
            # Count how many pairs in S would form a triple with x
            conflict_count = 0
            for (a, b, c) in triples:
                # Check if x is one of a,b,c and the other two are in S
                if x == a and b in S_set and c in S_set:
                    conflict_count += 1
                elif x == b and a in S_set and c in S_set:
                    conflict_count += 1
                elif x == c and a in S_set and b in S_set:
                    conflict_count += 1
            
            if conflict_count < best_conflict_count:
                best_conflict_count = conflict_count
                best_candidate = x
        
        if best_candidate is None or best_conflict_count > 0:
            break
        
        S.append(best_candidate)
        S_set.add(best_candidate)
    
    return sorted(S)
# EVOLVE-BLOCK-END
