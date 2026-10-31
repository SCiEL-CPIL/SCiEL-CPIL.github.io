import math
import time
import zlib

# Global state for dynamic priority
_start_time = None
_DEADLINE_FRACTION = 0.95
_selected_vectors = []  # Track selected vectors for dynamic scoring
_selected_set = set()   # For O(1) lookup
_symbol_counts = [0, 0, 0]  # Global symbol counts in selected set
_n_value = None

def priority(vector, n):
    """
    Dynamic priority function for greedy cap set construction in {0,1,2}^n.
    
    Structural change: Uses global state to compute a dynamic priority that
    considers the already-selected set. This allows the priority to adapt
    as the set grows, favoring vectors that maintain good spread relative
    to what's already chosen.
    
    Strategy:
    1. Track which vectors have been selected and their symbol distributions.
    2. Compute a dynamic score based on:
       - Balance: How well this vector complements the current symbol distribution
       - Spread: Distance from existing vectors (proxy for triple avoidance)
       - Structure: Local pattern quality of the vector itself
    3. Use a hash for deterministic tie-breaking.
    """
    global _start_time, _selected_vectors, _selected_set, _symbol_counts, _n_value
    
    if _start_time is None:
        _start_time = time.time()
        _n_value = n
    
    # Check deadline
    if time.time() - _start_time > 300 * _DEADLINE_FRACTION:
        return -1e10

    vec_key = tuple(vector)
    
    # Count symbols in this vector
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # === DYNAMIC COMPONENT 1: Balance with current set ===
    # Reward vectors that help balance the overall symbol distribution
    total_selected = len(_selected_vectors)
    if total_selected > 0:
        # Target: each symbol should be ~1/3 of total
        target_per_symbol = (total_selected * n) / 3.0
        current_per_symbol = [s * n for s in _symbol_counts]
        
        # How much does this vector help balance?
        balance_score = 0.0
        for i, c in enumerate([c0, c1, c2]):
            deficit = target_per_symbol - current_per_symbol[i]
            if deficit > 0:
                # This symbol is underrepresented; reward adding more
                balance_score += min(c, deficit) / max(deficit, 1)
            else:
                # This symbol is overrepresented; penalize adding more
                balance_score -= min(c, -deficit) / max(-deficit, 1) * 0.5
        balance_score /= 3.0
    else:
        # No vectors selected yet; use structural preference
        distinct_count = sum(1 for c in [c0, c1, c2] if c > 0)
        balance_score = distinct_count / 3.0
    
    # === DYNAMIC COMPONENT 2: Spread from existing vectors ===
    # Approximate distance-based spread using symbol distribution differences
    # and a lightweight coordinate check against a sample of existing vectors
    spread_score = 0.0
    if _selected_vectors:
        # Sample up to 20 existing vectors for distance estimation
        sample_size = min(20, len(_selected_vectors))
        step = max(1, len(_selected_vectors) // sample_size)
        sample = _selected_vectors[::step][:sample_size]
        
        for existing in sample:
            # Compute a lightweight distance: fraction of coordinates where
            # the pair could form a "bad" alignment (same or consecutive mod 3)
            diff_count = 0
            for i in range(n):
                a = vector[i]
                b = existing[i]
                # In F_3, a,b,c form an AP if a+b+c ≡ 0 (mod 3) with a,b,c
                # For two points, we care about whether they're "close" in a way
                # that makes it hard to avoid triples
                d = (a - b) % 3
                if d != 0:
                    diff_count += 1
            # Higher diff_count means more spread
            spread_score += diff_count / n
        
        spread_score /= sample_size
    else:
        # No existing vectors; give full spread credit
        spread_score = 1.0
    
    # === STATIC COMPONENT: Structural quality of the vector ===
    # Uniformity within the vector
    uniform_target = n / 3.0
    uniformity_penalty = 0.0
    for c in [c0, c1, c2]:
        diff = c - uniform_target
        uniformity_penalty += diff * diff
    uniformity_score = 1.0 - uniformity_penalty / (n * n)
    
    # Entropy
    entropy = 0.0
    for c in [c0, c1, c2]:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    entropy_normalized = entropy / math.log(3) if n > 0 else 0.0
    
    # === COMBINE SCORES ===
    # Weight dynamic components more heavily as the set grows
    dynamic_weight = min(1.0, total_selected / 50.0)
    
    score = (
        (1.0 - dynamic_weight) * (uniformity_score * 0.3 + entropy_normalized * 0.3 + balance_score * 0.4) +
        dynamic_weight * (balance_score * 0.4 + spread_score * 0.4 + (uniformity_score * 0.5 + entropy_normalized * 0.5) * 0.2)
    )
    
    # Tie-breaking hash
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 13) % (10**9 + 7)
    hash_score = h / (10**9 + 7) * 0.01
    
    final_score = score + hash_score
    
    # Update global state (this vector is being selected)
    _selected_vectors.append(vec_key)
    _selected_set.add(vec_key)
    _symbol_counts[0] += c0
    _symbol_counts[1] += c1
    _symbol_counts[2] += c2
    
    return float(final_score)
