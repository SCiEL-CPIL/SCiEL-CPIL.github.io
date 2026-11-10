import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples as tuples of sorted points
    all_triples = list(combinations(points, 3))
    
    best_alpha = 16
    best_hypergraph = []
    
    # Helper to calculate alpha
    def calc_alpha(hypergraph):
        if not hypergraph:
            return n
        # Convert to bitmasks for speed
        triple_masks = []
        for t in hypergraph:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check from largest k down
        # We know alpha is at least 1
        # We can skip k=16, 15 if we have enough edges?
        # Actually, just iterate k from n down to 1
        for k in range(n, 0, -1):
            found = False
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is contained in mask
                is_independent = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
        return 1

    # Greedy construction
    def greedy_construct():
        used_pairs = set()
        hypergraph = []
        
        # Create a shuffled list of all triples
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            # Check if t is linear with current hypergraph
            # i.e., no pair in t is already used
            p0, p1, p2 = t
            pairs = [(p0, p1), (p0, p2), (p1, p2)]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                hypergraph.append(t)
                used_pairs.update(pairs)
                
        return hypergraph

    # Main loop
    iterations = 0
    while time.time() < deadline:
        h = greedy_construct()
        a = calc_alpha(h)
        
        if a < best_alpha:
            best_alpha = a
            best_hypergraph = h
            # If we reach a very good score, we might want to keep searching a bit
            # to see if we can do better, but 6 is likely optimal or near optimal.
            # If we find 5, we can probably stop or keep going.
            if a <= 5:
                # 5 is extremely unlikely, but if found, it's great.
                # We can break if we are confident, but let's run a few more.
                pass
        
        iterations += 1
        # Optional: Early stopping if we've found 6 and run enough iterations
        if best_alpha == 6 and iterations > 100:
            # Check time, if we have plenty, continue. If not, break.
            if time.time() > deadline - 20.0:
                break

    # Return the best hypergraph found
    # The task requires returning a list of triples [a, b, c]
    return [list(t) for t in best_hypergraph]

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"Found {len(res)} triples")
    # Quick alpha check
    # ...
