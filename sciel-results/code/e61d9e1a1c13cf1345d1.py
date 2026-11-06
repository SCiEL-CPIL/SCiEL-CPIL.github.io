import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0

    points = list(range(16))
    
    # --- SQS(16) Construction ---
    # F4 arithmetic
    # 0: 0, 1: 1, 2: w, 3: w+1
    # w^2 = w+1
    # Table for multiplication
    # 0*x=0
    # 1*x=x
    # 2*0=0, 2*1=2, 2*2=3, 2*3=1 (w*(w+1) = w^2+w = (w+1)+w = 1)
    # 3*0=0, 3*1=3, 3*2=1, 3*3=2 ((w+1)*(w+1) = w^2+2w+1 = (w+1)+w+1 = w = 2)
    
    def f4_add(a, b):
        return a ^ b # GF(4) addition is XOR if we map 0,1,2,3 to 00,01,10,11?
        # Let's check: 0+1=1 (00^01=01), 1+2=3 (01^10=11), 2+3=1 (10^11=01).
        # 1+1=0 (01^01=00).
        # Yes, if we map 0->0, 1->1, 2->2, 3->3, addition is XOR.
        
    def f4_mul(a, b):
        if a == 0 or b == 0: return 0
        if a == 1: return b
        if b == 1: return a
        if a == 2 and b == 2: return 3
        if a == 2 and b == 3: return 1
        if a == 3 and b == 2: return 1
        if a == 3 and b == 3: return 2
        return 0 # fallback
        
    # Points are (x, y) in F4x2
    # Map (x,y) to index 4*x + y
    def point_to_idx(x, y):
        return 4*x + y
        
    def idx_to_point(idx):
        return (idx // 4, idx % 4)

    blocks = []
    # Vertical lines x=c
    for c in range(4):
        pts = [point_to_idx(c, y) for y in range(4)]
        blocks.append(tuple(sorted(pts)))
        
    # Sloped lines y = mx + c
    for m in range(4):
        for c in range(4):
            pts = []
            for x in range(4):
                y = f4_add(f4_mul(m, x), c)
                pts.append(point_to_idx(x, y))
            blocks.append(tuple(sorted(pts)))
            
    # Generate all triples from these blocks
    sqs_triples = []
    for b in blocks:
        for t in itertools.combinations(b, 3):
            sqs_triples.append(tuple(t))
            
    # sqs_triples has 80 entries
    # Build conflict graph
    # Map pair to list of triples in sqs_triples containing it
    pair_to_sqs_triples = defaultdict(list)
    for t in sqs_triples:
        for p in itertools.combinations(t, 2):
            pair_to_sqs_triples[p].append(t)
            
    # Conflict graph: vertices are indices in sqs_triples
    # Edge if they share a pair
    n_sqs = len(sqs_triples)
    conflict_adj = [[] for _ in range(n_sqs)]
    for i in range(n_sqs):
        t_i = sqs_triples[i]
        for p in itertools.combinations(t_i, 2):
            for j in pair_to_sqs_triples[p]:
                if j != i:
                    conflict_adj[i].append(j)
    # Remove duplicates in adjacency lists
    for i in range(n_sqs):
        conflict_adj[i] = list(set(conflict_adj[i]))
        
    # Find a large independent set in this graph
    # Use greedy coloring or local search
    # We want size 35
    
    def get_alpha(triples):
        # Optimized alpha calculation
        # Use bitmasks
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
            
        # Check k from 16 down to 1
        # We can stop early if we find one
        # But we want to know the max.
        # For speed, we can just check if alpha >= 7.
        # If not, check 6, etc.
        
        # Precompute combinations of points for each k?
        # No, just iterate.
        
        # Optimization: If we know alpha is likely small, check from 5 up?
        # No, we want the max.
        # Let's just check 16 down to 1.
        
        # To speed up, we can use a necessary condition:
        # If a set S is independent, it cannot contain any triple.
        
        # Let's use the same method as before but maybe cache?
        # No, let's just run it.
        
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for t_mask in triple_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 1

    # Heuristic to find independent set of size ~35
    # Start with empty set, add random non-conflicting triples
    best_sqs_set = []
    best_sqs_alpha = 16
    
    for _ in range(100): # 100 restarts
        random.shuffle(sqs_triples)
        current_set = []
        used_pairs = set()
        
        for t in sqs_triples:
            if all(p not in used_pairs for p in itertools.combinations(t, 2)):
                current_set.append(t)
                for p in itertools.combinations(t, 2):
                    used_pairs.add(p)
                    
        alpha = get_alpha(current_set)
        if alpha < best_sqs_alpha:
            best_sqs_alpha = alpha
            best_sqs_set = current_set[:]
            
        if best_sqs_alpha <= 5:
            break
            
    # Local search on the SQS-derived set?
    # Or on the general space?
    # Let's do general local search starting from best_sqs_set
    
    current_system = best_sqs_set[:]
    current_alpha = best_sqs_alpha
    best_alpha = current_alpha
    best_system = current_system[:]
    
    # Precompute all triples
    all_triples = list(itertools.combinations(points, 3))
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Map pair to all triples (for general search)
    # This is expensive to build fully, but we only need it for candidates.
    # Instead, for a given system, we find candidates by checking all_triples?
    # 560 triples.
    
    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])
        
    # Local Search
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < deadline:
        if not current_system:
            break
            
        # Pick a random triple to remove
        idx = random.randint(0, len(current_system) - 1)
        t_remove = current_system[idx]
        
        # Find candidates to add
        # Temporarily remove t_remove
        used_pairs = set()
        for i, t in enumerate(current_system):
            if i != idx:
                used_pairs.update(triple_pairs[t])
                
        candidates = []
        for t in all_triples:
            if t == t_remove:
                continue
            if can_add(t, used_pairs):
                candidates.append(t)
                
        if not candidates:
            continue
            
        t_add = random.choice(candidates)
        
        # Create new system
        new_system = current_system[:]
        new_system[idx] = t_add
        
        new_alpha = get_alpha(new_system)
        
        delta = new_alpha - current_alpha
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_system = new_system
            current_alpha = new_alpha
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_system = current_system[:]
                if best_alpha <= 5:
                    break
        else:
            # Revert (do nothing)
            pass
            
        T *= cooling
        if T < 0.1:
            T = 0.1 # Floor temperature
            
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
