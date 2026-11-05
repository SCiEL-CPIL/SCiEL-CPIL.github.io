"""Program for the suite task: maximize subset of [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Base set: {51, ..., 100}
    # Any subset summing to 200 must have size <= 3 because 4*51 = 204 > 200.
    # We find all triples (a,b,c) in this set such that a+b+c=200.
    # We then greedily remove elements to break all such triples.
    # Strategy: Repeatedly remove the element involved in the most remaining triples.
    
    # Start with a larger base set: {49, ..., 100}
    # Min sum of 4 distinct elements is 49+50+51+52 = 202 > 200.
    # So only triples (and pairs, but no pairs sum to 200 in [1,100] with distinct elements) need to be broken.
    candidates = list(range(49, 101))
    
    # Generate all conflicting triples
    triples = []
    for a, b, c in itertools.combinations(candidates, 3):
        if a + b + c == 200:
            triples.append((a, b, c))
    
    active_triples = triples
    
    removed = set()
    
    while active_triples:
        # Count occurrences of each candidate in active_triples
        counts = {}
        for t in active_triples:
            for x in t:
                if x not in removed:
                    counts[x] = counts.get(x, 0) + 1
        
        if not counts:
            break
            
        # Find the element with the maximum count
        # If tie, pick any (or smallest, doesn't matter much)
        max_count = max(counts.values())
        # Pick one with max count
        # To be deterministic, pick the smallest index if tie
        target = min([k for k, v in counts.items() if v == max_count])
        
        # Remove this element
        removed.add(target)
        
        # Update active_triples: remove any triple containing target
        active_triples = [t for t in active_triples if target not in t]
        
    # The result is the candidates that were not removed
    result = [x for x in candidates if x not in removed]
    
    # Ensure nonempty
    if not result:
        return [1]
        
    return result
# EVOLVE-BLOCK-END
