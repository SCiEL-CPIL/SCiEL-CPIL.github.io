import time
import random
import itertools
import numpy as np

# Global variables for speed
V = 16
ALL_TRIPLES = []
TRIPLE_TO_MASK = {}
MASK_TO_TRIPLE = {}
PAIRS_IN_TRIPLE = {}
SUBSET_SIZES = [0] * (1 << V)
IS_INDEPENDENT = [True] * (1 << V)
# Precompute subset sizes
for i in range(1 << V):
    SUBSET_SIZES[i] = bin(i).count('1')

# Precompute subsets by size for fast alpha calculation
SUBSETS_BY_SIZE = [[] for _ in range(V + 1)]
for i in range(1 << V):
    SUBSETS_BY_SIZE[SUBSET_SIZES[i]].append(i)
# Convert to numpy arrays for vectorized access
SUBSETS_BY_SIZE_ARR = [np.array(subs, dtype=np.int32) for subs in SUBSETS_BY_SIZE]

SUPERSETS_OF_MASK = {}

def init_data():
    global ALL_TRIPLES, TRIPLE_TO_MASK, MASK_TO_TRIPLE, PAIRS_IN_TRIPLE
    ALL_TRIPLES = []
    for t in itertools.combinations(range(V), 3):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        ALL_TRIPLES.append((t, mask))
        TRIPLE_TO_MASK[t] = mask
        MASK_TO_TRIPLE[mask] = t
        # Pairs in triple
        pairs = ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))
        # Normalize pairs
        norm_pairs = set()
        for p in pairs:
            if p[0] > p[1]:
                p = (p[1], p[0])
            norm_pairs.add(p)
        PAIRS_IN_TRIPLE[t] = norm_pairs
        
        # Precompute supersets for this mask
        # Supersets are S such that S & mask == mask
        # Generate all 2^16 masks? No, use the fact that there are only 2^13 supersets.
        # But generating them explicitly is complex.
        # Let's just use the vectorized check for precomputation, it's simple.
        # 560 * 65536 is ~36M ops. NumPy handles this in <1s usually.
        all_masks = np.arange(1 << V, dtype=np.int32)
        supersets = all_masks[(all_masks & mask) == mask]
        SUPERSETS_OF_MASK[mask] = supersets

# Initialize data on import
init_data()

def get_alpha_from_cover_count(cover_count):
    """
    Compute the independence number given the cover_count array.
    cover_count[S] is the number of edges contained in S.
    S is independent if cover_count[S] == 0.
    """
    # Check from largest size down
    for k in range(V, -1, -1):
        subsets = SUBSETS_BY_SIZE_ARR[k]
        if len(subsets) == 0:
            continue
        if np.any(cover_count[subsets] == 0):
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s margin
    
    # 1. Greedy Construction of a large packing
    edges = []
    edge_masks = []
    used_pairs = set()
    
    # Shuffle triples for randomness
    shuffled_triples = ALL_TRIPLES[:]
    random.shuffle(shuffled_triples)
    
    # Add as many as possible
    for t, m in shuffled_triples:
        # Check if valid
        valid = True
        for p in PAIRS_IN_TRIPLE[t]:
            if p in used_pairs:
                valid = False
                break
        if valid:
            edges.append(t)
            edge_masks.append(m)
            for p in PAIRS_IN_TRIPLE[t]:
                used_pairs.add(p)
                
    # Initialize cover_count
    cover_count = np.zeros(1 << V, dtype=np.int32)
    for m in edge_masks:
        cover_count[SUPERSETS_OF_MASK[m]] += 1
        
    # Initial alpha
    current_alpha = get_alpha_from_cover_count(cover_count)
    best_alpha = current_alpha
    best_edges = edges[:]
    best_masks = edge_masks[:]
    
    # 2. Local Search / Simulated Annealing
    # We want to minimize alpha.
    # Moves:
    # a) Add a random valid triple.
    # b) Remove a random triple.
    # c) Swap a triple.
    
    # Since adding triples reduces alpha (or keeps it), we should prefer additions.
    # But we might get stuck in a local minimum where we can't add more, 
    # but a different set of triples would allow a denser packing or lower alpha.
    
    # Strategy:
    # While time < deadline:
    #    Try random moves.
    #    If new_alpha < current_alpha:
    #        Accept, update best.
    #    Else:
    #        Accept with probability p (simulated annealing)
        
    T = 1.0  # Temperature
    T_min = 0.01
    cooling = 0.999
    
    # Precompute list of all triples not in current edges?
    # Or just pick random triple from ALL_TRIPLES and check if valid to add.
    
    while time.time() < deadline:
        # Choose a move type
        # If we haven't reached max packing size, try to add.
        # Max packing size for v=16 is 36 or 37?
        # Let's just try to add/remove/swap.
        
        move_type = random.random()
        
        new_edges = edges[:]
        new_masks = edge_masks[:]
        new_used_pairs = used_pairs.copy()
        new_cover_count = cover_count.copy() # Copy is 65536 ints, fast
        
        if move_type < 0.4:
            # Try to add a triple
            # Pick a random triple
            t, m = random.choice(ALL_TRIPLES)
            # Check validity
            valid = True
            for p in PAIRS_IN_TRIPLE[t]:
                if p in new_used_pairs:
                    valid = False
                    break
            if valid:
                new_edges.append(t)
                new_masks.append(m)
                for p in PAIRS_IN_TRIPLE[t]:
                    new_used_pairs.add(p)
                # Update cover count
                new_cover_count[SUPERSETS_OF_MASK[m]] += 1
            else:
                continue # Skip this iteration if invalid add
        
        elif move_type < 0.7:
            # Try to remove a triple
            if not new_edges:
                continue
            idx = random.randint(0, len(new_edges)-1)
            t = new_edges[idx]
            m = new_masks[idx]
            # Remove pairs
            for p in PAIRS_IN_TRIPLE[t]:
                new_used_pairs.remove(p)
            new_edges.pop(idx)
            new_masks.pop(idx)
            # Update cover count
            new_cover_count[SUPERSETS_OF_MASK[m]] -= 1
            
        else:
            # Try to swap: remove one, add another
            if not new_edges:
                continue
            # Remove a random one
            idx = random.randint(0, len(new_edges)-1)
            t_old = new_edges[idx]
            m_old = new_masks[idx]
            for p in PAIRS_IN_TRIPLE[t_old]:
                new_used_pairs.remove(p)
            new_edges.pop(idx)
            new_masks.pop(idx)
            new_cover_count[SUPERSETS_OF_MASK[m_old]] -= 1
            
            # Try to add a random new one
            # To increase chance of success, pick from a few candidates?
            # Just pick random
            t_new, m_new = random.choice(ALL_TRIPLES)
            valid = True
            for p in PAIRS_IN_TRIPLE[t_new]:
                if p in new_used_pairs:
                    valid = False
                    break
            if valid:
                new_edges.append(t_new)
                new_masks.append(m_new)
                for p in PAIRS_IN_TRIPLE[t_new]:
                    new_used_pairs.add(p)
                new_cover_count[SUPERSETS_OF_MASK[m_new]] += 1
            else:
                # Revert removal? Or just skip?
                # If we can't add, maybe don't remove?
                # Let's just skip this move to avoid degradation without benefit
                continue

        # Evaluate new state
        new_alpha = get_alpha_from_cover_count(new_cover_count)
        
        if new_alpha < current_alpha:
            # Accept
            current_alpha = new_alpha
            edges = new_edges
            edge_masks = new_masks
            used_pairs = new_used_pairs
            cover_count = new_cover_count
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = edges[:]
                best_masks = edge_masks[:]
                
        elif new_alpha == current_alpha:
            # Accept with high probability (plateau)
            if random.random() < 0.5:
                current_alpha = new_alpha
                edges = new_edges
                edge_masks = new_masks
                used_pairs = new_used_pairs
                cover_count = new_cover_count
            # T stays same or cools slowly?
            T = max(T_min, T * cooling)
        else:
            # Worse
            # Accept with prob exp((current - new) / T)
            # Since we minimize, delta = new - current > 0.
            # prob = exp(-delta / T)
            delta = new_alpha - current_alpha
            prob = np.exp(-delta / T) if T > 0 else 0
            if random.random() < prob:
                current_alpha = new_alpha
                edges = new_edges
                edge_masks = new_masks
                used_pairs = new_used_pairs
                cover_count = new_cover_count
            
            T = max(T_min, T * cooling)
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_edges

# Quick test
# print(construct())
