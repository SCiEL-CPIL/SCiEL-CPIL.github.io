import time
import random
from itertools import combinations

# Precompute all 560 triples on 0..15 as sorted tuples
ALL_TRIPLES = list(combinations(range(16), 3))
NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute bitmask for each triple
TRIPLE_MASKS = [0] * NUM_TRIPLES
for i, t in enumerate(ALL_TRIPLES):
    m = 0
    for p in t:
        m |= (1 << p)
    TRIPLE_MASKS[i] = m

# Precompute the 3 unordered pairs (as frozensets or sorted tuples) for each triple
# We use a dictionary to map pair to list of triple indices containing that pair
# This helps in quickly finding conflicts or valid additions
PAIR_TO_TRIPLES = {}
for i, t in enumerate(ALL_TRIPLES):
    pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
    for p in pairs:
        if p not in PAIR_TO_TRIPLES:
            PAIR_TO_TRIPLES[p] = []
        PAIR_TO_TRIPLES[p].append(i)

# Precompute triple pairs for quick access
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    TRIPLE_PAIRS.append((
        (t[0], t[1]),
        (t[0], t[2]),
        (t[1], t[2]),
    ))

def calc_alpha(edges):
    """
    Compute the independence number of the hypergraph given by edges (list of triple indices).
    Uses bitmask containment tests.
    """
    if not edges:
        return 16
    
    # Build masks for the current edges
    edge_masks = [TRIPLE_MASKS[i] for i in edges]
    
    n = 16
    # Check for independent sets from largest to smallest
    # We can start from 8 downwards, as alpha is unlikely to be > 8 for dense packings
    # But to be safe, start from n.
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            indep = True
            for em in edge_masks:
                if (mask & em) == em:
                    indep = False
                    break
            if indep:
                return k
    return 1

def get_valid_candidates(used_pairs, current_edges_set):
    """
    Return a list of triple indices that are valid to add given used_pairs and current_edges_set.
    """
    candidates = []
    for i, t in enumerate(ALL_TRIPLES):
        if i in current_edges_set:
            continue
        # Check if any pair in t is in used_pairs
        valid = True
        for p in TRIPLE_PAIRS[i]:
            if p in used_pairs:
                valid = False
                break
        if valid:
            candidates.append(i)
    return candidates

def build_initial():
    """Random greedy construction of a linear triple system."""
    used_pairs = set()
    edges = []
    edges_set = set()
    candidates = list(range(NUM_TRIPLES))
    random.shuffle(candidates)
    
    for i in candidates:
        t = ALL_TRIPLES[i]
        pairs = TRIPLE_PAIRS[i]
        ok = True
        for p in pairs:
            if p in used_pairs:
                ok = False
                break
        if ok:
            edges.append(i)
            edges_set.add(i)
            used_pairs.update(pairs)
            
    return edges, used_pairs, edges_set

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Initial construction
    best_edges, best_used_pairs, best_edges_set = build_initial()
    best_alpha = calc_alpha(best_edges)
    
    # Simulated Annealing / Local Search
    # We will use a move that removes 2 edges and adds 2 edges to escape local minima
    # This is a structural change from the single-edge swap in the current program.
    
    current_edges = best_edges[:]
    current_used_pairs = best_used_pairs.copy()
    current_edges_set = best_edges_set.copy()
    current_alpha = best_alpha
    
    temperature = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        if current_alpha <= 5:
            break
            
        # Choose a move: Remove 2 random edges
        if len(current_edges) < 2:
            break
            
        idx1, idx2 = random.sample(range(len(current_edges)), 2)
        # Ensure unique indices
        if idx1 == idx2:
            continue
            
        e1 = current_edges[idx1]
        e2 = current_edges[idx2]
        
        # Remove e1 and e2
        # Sort indices to remove safely
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            
        removed_e2 = current_edges.pop(idx2)
        removed_e1 = current_edges.pop(idx1)
        
        current_edges_set.discard(removed_e1)
        current_edges_set.discard(removed_e2)
        
        # Update used_pairs
        for p in TRIPLE_PAIRS[removed_e1]:
            current_used_pairs.discard(p)
        for p in TRIPLE_PAIRS[removed_e2]:
            current_used_pairs.discard(p)
            
        # Find candidates to add
        candidates = get_valid_candidates(current_used_pairs, current_edges_set)
        
        if len(candidates) < 2:
            # Cannot add 2 edges, restore
            current_edges.append(removed_e1)
            current_edges.append(removed_e2)
            current_edges_set.add(removed_e1)
            current_edges_set.add(removed_e2)
            current_used_pairs.update(TRIPLE_PAIRS[removed_e1])
            current_used_pairs.update(TRIPLE_PAIRS[removed_e2])
            continue
            
        # Pick 2 random candidates to add
        # We could pick the best 2 based on alpha, but that's expensive.
        # Random selection with SA acceptance is usually sufficient.
        new_e1, new_e2 = random.sample(candidates, 2)
        
        # Add new edges
        current_edges.append(new_e1)
        current_edges.append(new_e2)
        current_edges_set.add(new_e1)
        current_edges_set.add(new_e2)
        current_used_pairs.update(TRIPLE_PAIRS[new_e1])
        current_used_pairs.update(TRIPLE_PAIRS[new_e2])
        
        # Calculate new alpha
        new_alpha = calc_alpha(current_edges)
        
        # Acceptance
        delta = new_alpha - current_alpha
        if delta < 0 or (temperature > 0 and random.random() < pow(2.718, -delta / temperature)):
            # Accept
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = current_edges[:]
                best_used_pairs = current_used_pairs.copy()
                best_edges_set = current_edges_set.copy()
        else:
            # Reject: Restore
            current_edges.pop() # new_e2
            current_edges.pop() # new_e1
            current_edges_set.discard(new_e1)
            current_edges_set.discard(new_e2)
            current_used_pairs.difference_update(TRIPLE_PAIRS[new_e1])
            current_used_pairs.difference_update(TRIPLE_PAIRS[new_e2])
            
            current_edges.append(removed_e1)
            current_edges.append(removed_e2)
            current_edges_set.add(removed_e1)
            current_edges_set.add(removed_e2)
            current_used_pairs.update(TRIPLE_PAIRS[removed_e1])
            current_used_pairs.update(TRIPLE_PAIRS[removed_e2])
            
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    return [list(ALL_TRIPLES[i]) for i in best_edges]

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
