import random
import time
import numpy as np

def compute_alpha_np(triples):
    """
    Computes the independence number of a linear triple system on 16 points.
    Uses NumPy for vectorized subset checking.
    """
    if not triples:
        return 16
        
    # Precompute masks for triples
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    triple_masks_np = np.array(triple_masks, dtype=np.int32)
    
    # Generate all subset masks 0..65535
    # We can cache this
    if not hasattr(compute_alpha_np, "all_masks"):
        compute_alpha_np.all_masks = np.arange(1 << 16, dtype=np.int32)
        # Precompute popcount
        pc = np.zeros(1 << 16, dtype=np.int8)
        for i in range(1, 1 << 16):
            pc[i] = pc[i >> 1] + (i & 1)
        compute_alpha_np.popcount = pc
        
    all_masks = compute_alpha_np.all_masks
    popcount = compute_alpha_np.popcount
    
    # Check if any triple is contained in each mask
    # Shape: (65536, num_triples)
    # (mask & triple) == triple
    # This might be memory heavy if num_triples is large, but 35 is small.
    # 65536 * 35 * 4 bytes = ~9MB. Acceptable.
    
    # To save memory/time, we can do it in chunks or just let it be.
    # Vectorized operation:
    # intersects = (all_masks[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    # has_triple = intersects.any(axis=1)
    # independent = ~has_triple
    # max_alpha = popcount[independent].max()
    
    # Optimization: If we find an independent set of size 16, return 16 immediately?
    # Unlikely.
    
    # Let's try the direct vectorized approach.
    # It might be slow in Python if the array is too large for vectorization overhead?
    # 65536 * 35 is ~2.3M elements. NumPy handles this easily.
    
    intersects = (all_masks[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    has_triple = intersects.any(axis=1)
    independent = ~has_triple
    
    if not independent.any():
        return 0 # Should not happen for empty set, but safe
        
    max_alpha = popcount[independent].max()
    return int(max_alpha)

def generate_all_triples():
    """Generate all possible triples on 16 points."""
    points = range(16)
    triples = []
    for a in points:
        for b in points:
            if b <= a: continue
            for c in points:
                if c <= b: continue
                triples.append((a, b, c))
    return triples

def greedy_construct(all_triples, seed=None):
    """
    Randomized greedy construction of a linear triple system.
    """
    if seed is not None:
        random.seed(seed)
        
    candidates = list(all_triples)
    random.shuffle(candidates)
    
    used_pairs = set()
    triples = []
    
    for t in candidates:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        
        # Check if any pair is already used
        conflict = False
        for p in pairs:
            if p in used_pairs:
                conflict = True
                break
                
        if not conflict:
            triples.append(t)
            for p in pairs:
                used_pairs.add(p)
                
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    all_triples = generate_all_triples()
    
    best_alpha = 16
    best_triples = []
    
    # Pre-compute alpha for empty set? No.
    
    # Initial run
    # Try a few fixed seeds first to get a baseline
    for i in range(10):
        if time.time() > deadline:
            break
        triples = greedy_construct(all_triples, seed=i)
        alpha = compute_alpha_np(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 4: # Likely optimal or near optimal
                # We can stop early if we find a very good one, 
                # but let's run a bit more to be sure or find 3?
                pass

    # Random search
    # The greedy algorithm is deterministic for a given seed.
    # We can just iterate seeds.
    # Or use random seeds.
    
    current_seed = 100
    while time.time() < deadline:
        triples = greedy_construct(all_triples, seed=current_seed)
        current_seed += 1
        
        # Optional: Local search? 
        # For now, just evaluate.
        alpha = compute_alpha_np(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            # If we hit 3, it's likely the theoretical minimum for this class?
            # f(16) is known to be small.
            if best_alpha <= 3:
                # We can continue to see if we can get 2? 
                # Independent set of size 2 means no triple is a subset of {a,b}? 
                # Impossible since triples have size 3.
                # So min alpha is at least 3? 
                # Wait, if there are no triples, alpha=16.
                # If we have triples, alpha can be small.
                # Can alpha be 2? 
                # An independent set of size 2 contains no triple. 
                # Any set of size 2 contains no triple (size 3).
                # So alpha is always at least... wait.
                # The definition is "largest point set containing no triple".
                # If the hypergraph has at least one triple, any set of size 2 is independent?
                # No, "containing no triple" means no edge is a subset of the set.
                # A set of size 2 cannot contain a triple of size 3.
                # So any set of size 2 is independent.
                # Thus alpha >= 2 always?
                # Actually, if the system has no triples, alpha=16.
                # If it has triples, alpha is at least 2?
                # Yes, because any 2 points form a set with no triple.
                # So the minimum possible alpha is 2?
                # Wait, if I have a triple {0,1,2}, the set {0,1} is independent.
                # Can I have alpha=1? No, because {0} is independent.
                # So alpha >= 1.
                # But usually we look for larger structures.
                # For a dense system, alpha will be small.
                pass

    return [list(t) for t in best_triples]

# Fallback if numpy is not available or fails
def compute_alpha_fallback(triples):
    if not triples:
        return 16
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
        
    max_size = 0
    # Iterate from high size down
    # This is slow, but fallback
    for mask in range(1 << 16):
        is_indep = True
        for m in triple_masks:
            if (mask & m) == m:
                is_indep = False
                break
        if is_indep:
            size = bin(mask).count("1")
            if size > max_size:
                max_size = size
    return max_size

# Wrap compute_alpha_np to handle errors
def safe_compute_alpha(triples):
    try:
        return compute_alpha_np(triples)
    except Exception:
        return compute_alpha_fallback(triples)

# Override the function used in construct
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    all_triples = generate_all_triples()
    
    best_alpha = 16
    best_triples = []
    
    # Initial seeds
    for i in range(20):
        if time.time() > deadline:
            break
        triples = greedy_construct(all_triples, seed=i)
        alpha = safe_compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples

    current_seed = 100
    while time.time() < deadline:
        triples = greedy_construct(all_triples, seed=current_seed)
        current_seed += 1
        alpha = safe_compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 3:
                # Very good, keep searching for 2? 
                # But 2 is trivial (any set of 2 is independent).
                # Wait, if the question implies "maximal independent set", 
                # and we want to MINIMIZE it.
                # If alpha=2, it means every set of 3 points contains a triple?
                # That would be a Steiner Triple System STS(16), which doesn't exist.
                # So alpha must be at least 3?
                # If alpha=3, it means every set of 4 points contains a triple.
                pass

    return [list(t) for t in best_triples]
