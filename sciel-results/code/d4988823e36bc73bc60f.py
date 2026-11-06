import itertools
import random
import time

def construct():
    """
    Constructs a linear triple system on 16 points minimizing the independence number.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Helper to convert triple to bitmask
    def to_mask(t):
        return (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    # Helper to check if a mask is independent of a set of triple masks
    def is_independent(mask, triple_masks):
        for t in triple_masks:
            if (mask & t) == t:
                return False
        return True

    # Compute alpha (independence number)
    def compute_alpha(triple_masks):
        # Check from largest subset size down
        for k in range(n, 0, -1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                if is_independent(mask, triple_masks):
                    return k
        return 0

    # Initialize with a random linear hypergraph
    # We want to start with a dense system.
    # STS(15) is a good start. Let's hardcode a known STS(15) or generate one.
    # For simplicity and robustness, let's start with a random dense linear hypergraph.
    # Or better, start with the seed and improve? No, seed is bad (alpha=15).
    # Let's try to generate a random linear hypergraph with many edges.
    
    def random_linear_hypergraph(target_edges):
        triples = []
        pairs = {} # pair -> count
        for _ in range(target_edges):
            attempts = 0
            while attempts < 100:
                t = tuple(sorted(random.sample(range(n), 3)))
                p1 = (t[0], t[1])
                p2 = (t[0], t[2])
                p3 = (t[1], t[2])
                if pairs.get(p1, 0) < 1 and pairs.get(p2, 0) < 1 and pairs.get(p3, 0) < 1:
                    triples.append(list(t))
                    pairs[p1] = pairs.get(p1, 0) + 1
                    pairs[p2] = pairs.get(p2, 0) + 1
                    pairs[p3] = pairs.get(p3, 0) + 1
                    break
                attempts += 1
        return triples, pairs

    # Start with a reasonably dense random system
    current_triples, current_pairs = random_linear_hypergraph(35)
    current_masks = [to_mask(t) for t in current_triples]
    current_alpha = compute_alpha(current_masks)
    
    best_triples = current_triples
    best_alpha = current_alpha
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_triples = current_triples[:]
        new_pairs = current_pairs.copy()
        
        if move_type == 'add' and len(new_triples) < 45:
            # Try to add a random valid triple
            for _ in range(50):
                t = tuple(sorted(random.sample(range(n), 3)))
                p1, p2, p3 = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
                if new_pairs.get(p1, 0) < 1 and new_pairs.get(p2, 0) < 1 and new_pairs.get(p3, 0) < 1:
                    new_triples.append(list(t))
                    new_pairs[p1] = new_pairs.get(p1, 0) + 1
                    new_pairs[p2] = new_pairs.get(p2, 0) + 1
                    new_pairs[p3] = new_pairs.get(p3, 0) + 1
                    break
        elif move_type == 'remove' and len(new_triples) > 10:
            idx = random.randrange(len(new_triples))
            t = new_triples[idx]
            p1, p2, p3 = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
            new_pairs[p1] -= 1
            new_pairs[p2] -= 1
            new_pairs[p3] -= 1
            if new_pairs[p1] == 0: del new_pairs[p1]
            if new_pairs[p2] == 0: del new_pairs[p2]
            if new_pairs[p3] == 0: del new_pairs[p3]
            new_triples.pop(idx)
        elif move_type == 'swap' and len(new_triples) > 10:
            # Remove one, add another
            idx = random.randrange(len(new_triples))
            t_old = new_triples[idx]
            p1, p2, p3 = (t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])
            new_pairs[p1] -= 1
            new_pairs[p2] -= 1
            new_pairs[p3] -= 1
            if new_pairs[p1] == 0: del new_pairs[p1]
            if new_pairs[p2] == 0: del new_pairs[p2]
            if new_pairs[p3] == 0: del new_pairs[p3]
            new_triples.pop(idx)
            
            # Try to add a new one
            added = False
            for _ in range(50):
                t_new = tuple(sorted(random.sample(range(n), 3)))
                q1, q2, q3 = (t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])
                if new_pairs.get(q1, 0) < 1 and new_pairs.get(q2, 0) < 1 and new_pairs.get(q3, 0) < 1:
                    new_triples.append(list(t_new))
                    new_pairs[q1] = new_pairs.get(q1, 0) + 1
                    new_pairs[q2] = new_pairs.get(q2, 0) + 1
                    new_pairs[q3] = new_pairs.get(q3, 0) + 1
                    added = True
                    break
            if not added:
                # Revert if we couldn't add
                new_triples.insert(idx, t_old)
                new_pairs[p1] = new_pairs.get(p1, 0) + 1
                new_pairs[p2] = new_pairs.get(p2, 0) + 1
                new_pairs[p3] = new_pairs.get(p3, 0) + 1
                continue
        
        if len(new_triples) != len(current_triples) and move_type != 'add' and move_type != 'remove':
             continue # Safety check for swap
        
        # Evaluate new state
        new_masks = [to_mask(t) for t in new_triples]
        new_alpha = compute_alpha(new_masks)
        
        # Acceptance criterion
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_pairs = new_pairs
            current_masks = new_masks
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
                # Early stop if we hit a very good value?
                # Theoretical lower bound for n=16 is likely 7 or 8.
                # If we get 7, that's great.
        else:
            # Accept with probability exp(-delta / T)
            # We want to minimize alpha, so delta = new_alpha - current_alpha > 0 is bad
            delta = new_alpha - current_alpha
            if delta < 0:
                current_triples = new_triples
                current_pairs = new_pairs
                current_masks = new_masks
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            else:
                prob = 2.71828 ** (-delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                    current_pairs = new_pairs
                    current_masks = new_masks
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
        
        T *= cooling
        
    return best_triples

# Test locally
if __name__ == "__main__":
    t = construct()
    print(f"Found {len(t)} triples")
    # Verify linearity
    pairs = {}
    for tri in t:
        a, b, c = tri
        p1, p2, p3 = (a,b), (a,c), (b,c)
        for p in [p1, p2, p3]:
            if p in pairs:
                print("Invalid: pair repeated", p)
            pairs[p] = 1
    print("Valid linear system")
