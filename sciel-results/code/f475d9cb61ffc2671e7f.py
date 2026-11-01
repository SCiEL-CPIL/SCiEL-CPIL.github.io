import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases (not expected in this specific task instance but good for robustness)
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute ball membership: for each word, which balls contain it
    # This is the transpose of the balls matrix (sparse conceptually)
    # ball_of_word[w] = list of ball indices that cover word w
    # We'll use a different approach: for each ball, we know which words it covers
    
    best_size = total
    best_set = None
    
    def greedy_construct(seed_offset=0, rng=None):
        """Vectorized greedy set cover - hyperedge coloring formulation.
        
        Vertices = 6561 words, hyperedges = 17-word radius-1 balls.
        Select hyperedges to minimize count while covering all vertices.
        """
        if rng is None:
            rng = np.random.RandomState(seed_offset)
        
        uncovered_mask = np.ones(total, dtype=np.int32)  # 1 if uncovered
        selected = []
        
        while np.any(uncovered_mask):
            # Vectorized: count uncovered vertices in each ball
            # uncovered_mask[balls] has shape (total, ball_size), sum over axis=1
            counts = uncovered_mask[balls].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Random tie-breaking among candidates with max count
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update: mark words in this ball as covered
            ball_words = balls[best_idx]
            uncovered_mask[ball_words] = 0
            
            if time.time() > deadline - 8.0:
                return None  # timeout
        
        return selected
    
    def compute_cover_count(selected):
        """Compute coverage count for each word given selected balls."""
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        return cover_count
    
    def prune_set(selected, cover_count):
        """Remove redundant codewords - vectorized where possible."""
        changed = True
        while changed and time.time() < deadline - 4.0:
            changed = False
            new_selected = []
            for c in selected:
                # If all words in ball(c) are covered by at least 2 codewords, c is redundant
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count
    
    def local_search_swap(selected, cover_count, max_rounds=30):
        """Try 1-for-1 swaps: remove one codeword, add another that covers the gap.
        
        For each codeword c, find words uniquely covered by c (critical words).
        Try to find a replacement ball that covers all critical words.
        """
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rounds += 1
            
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                
                c = selected[i]
                ball_c = balls[c]
                
                # Find critical words: covered only by c
                critical = ball_c[cover_count[ball_c] == 1]
                
                if len(critical) == 0:
                    # c is redundant, remove it
                    cover_count[ball_c] -= 1
                    selected.pop(i)
                    improved = True
                    break
                elif len(critical) == 1:
                    # Only one critical word - try to find a ball that covers it
                    crit_word = critical[0]
                    # Find all balls that cover crit_word
                    # A ball b covers word w if w is in balls[b], i.e., b is in "balls containing w"
                    # We can check: for each candidate ball b, is crit_word in balls[b]?
                    # Vectorized: check which balls contain crit_word
                    contains_crit = (balls == crit_word).any(axis=1)
                    candidates = np.where(contains_crit)[0]
                    
                    # Exclude already selected balls and c itself
                    sel_set = set(selected)
                    candidates = [cand for cand in candidates if cand not in sel_set]
                    
                    if len(candidates) > 0:
                        # Try each candidate - pick one that covers most uncovered-after-removal words
                        # First, simulate removing c
                        cover_test = cover_count.copy()
                        cover_test[ball_c] -= 1
                        
                        best_cand = None
                        best_gain = -1
                        for cand in candidates[:20]:  # Limit candidates for speed
                            ball_cand = balls[cand]
                            # How many words would this cover that are currently uncovered?
                            gain = (cover_test[ball_cand] == 0).sum()
                            if gain > best_gain:
                                best_gain = gain
                                best_cand = cand
                        
                        if best_cand is not None and best_gain > 0:
                            # Perform the swap
                            cover_count[ball_c] -= 1
                            cover_count[balls[best_cand]] += 1
                            selected[i] = best_cand
                            improved = True
                            break
                    # If no good swap found, continue to next codeword
                # If multiple critical words, skip swap (too complex for now)
        
        return selected, cover_count
    
    # Multiple restarts with different seeds
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        rng_seed = 42 + num_restarts * 1000
        rng = np.random.RandomState(rng_seed)
        
        selected = greedy_construct(seed_offset=rng_seed, rng=rng)
        if selected is None:
            continue
        
        # Compute cover counts for pruning
        cover_count = compute_cover_count(selected)
        
        # Prune redundant codewords
        selected, cover_count = prune_set(selected, cover_count)
        
        # Local search: try swaps to reduce size
        selected, cover_count = local_search_swap(selected, cover_count, max_rounds=30)
        
        # Additional pruning after swaps
        selected, cover_count = prune_set(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
    
    # Final polish: try to find a smaller solution from the best found
    if best_set is not None and time.time() < deadline - 3.0:
        cover_count = compute_cover_count(best_set)
        
        # Aggressive pruning
        for _ in range(5):
            changed = False
            new_best = []
            for c in best_set:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_best.append(c)
            best_set = new_best
            if not changed:
                break
        
        # Try swaps on the best solution
        best_set, cover_count = local_search_swap(best_set, cover_count, max_rounds=50)
        
        # Final pruning
        for _ in range(3):
            changed = False
            new_best = []
            for c in best_set:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_best.append(c)
            best_set = new_best
            if not changed:
                break
        
        best_size = len(best_set)
    
    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback: return all words
        result = []
        for w in range(total):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
        return result
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
