"""Genetic algorithm with simulated annealing for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _count_tsubsets_in_block(block, t):
    """Return list of t-subsets (as frozensets) contained in block."""
    return [frozenset(c) for c in itertools.combinations(block, t)]


def _build_coverage_counter(blocks, t):
    """Build a dict mapping each t-subset (frozenset) to count of blocks covering it."""
    counter = {}
    for b in blocks:
        for ts in itertools.combinations(b, t):
            fs = frozenset(ts)
            counter[fs] = counter.get(fs, 0) + 1
    return counter


def _greedy_covering(v, k, t, rng, max_blocks=None):
    """Randomized greedy covering. Returns list of blocks."""
    all_tsubs = list(itertools.combinations(range(v), t))
    rng.shuffle(all_tsubs)
    
    uncovered = set(all_tsubs)
    blocks = []
    
    if max_blocks is None:
        max_blocks = len(all_tsubs)  # safe upper bound
    
    while uncovered and len(blocks) < max_blocks:
        # Pick a random uncovered t-subset to seed the block
        seed_ts = rng.choice(list(uncovered))
        block = list(seed_ts)
        rng.shuffle(block)
        
        # Greedily add points that cover most uncovered t-subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_ts = tuple(sorted(rest + (x,)))
                    if new_ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and best_x is not None:
                    # Tie-break randomly
                    if rng.random() < 0.5:
                        best_x = x
            
            if best_x is None or best_gain <= 0:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return blocks


def _try_replace_block(blocks, idx, v, k, t, uncovered_set, rng, time_limit):
    """Try to replace block at idx with a better one. Returns new block or None."""
    start_time = time.time()
    old_block = blocks[idx]
    
    # Find t-subsets that are ONLY covered by this block
    # First, count coverage
    cover_count = {}
    for i, b in enumerate(blocks):
        if i == idx:
            continue
        for ts in itertools.combinations(b, t):
            fs = frozenset(ts)
            cover_count[fs] = cover_count.get(fs, 0) + 1
    
    # T-subsets only covered by old_block
    exclusive = set()
    for ts in itertools.combinations(old_block, t):
        fs = frozenset(ts)
        if cover_count.get(fs, 0) == 0:
            exclusive.add(fs)
    
    if not exclusive:
        # Can remove this block entirely? Return empty marker
        return None  # Signal removal
    
    # Try to find a new block that covers all exclusive t-subsets
    # Start with points from exclusive t-subsets
    required_points = set()
    for ts in exclusive:
        required_points.update(ts)
    
    # If required_points > k, we can't cover all with one block - skip
    if len(required_points) > k:
        return None
    
    new_block = list(required_points)
    rng.shuffle(new_block)
    
    # Add more points to maximize coverage of currently uncovered or weakly covered t-subsets
    while len(new_block) < k and time.time() - start_time < 0.01:
        best_x = None
        best_score = -1
        for x in range(v):
            if x in new_block:
                continue
            score = 0
            for rest in itertools.combinations(new_block, t - 1):
                new_ts = tuple(sorted(rest + (x,)))
                fs = frozenset(new_ts)
                if fs in uncovered_set:
                    score += 2  # High priority: currently uncovered
                elif cover_count.get(fs, 0) <= 1:
                    score += 1  # Medium: weakly covered
            if score > best_score:
                best_score = score
                best_x = x
            elif score == best_score and best_x is not None and rng.random() < 0.3:
                best_x = x
        
        if best_x is None or best_score <= 0:
            break
        new_block.append(best_x)
    
    if len(new_block) < t:
        return None
    
    new_block.sort()
    if new_block == old_block:
        return None
    
    return new_block


def construct(v, k, t):
    """Construct a covering design using genetic algorithm with simulated annealing."""
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for 120s limit
    
    # Initial population: multiple randomized greedy solutions
    pop_size = min(8, max(3, 60 // max(1, v // 10)))
    population = []
    
    for _ in range(pop_size):
        blocks = _greedy_covering(v, k, t, rng, max_blocks=200)
        population.append(blocks)
    
    # Track best solution
    best_blocks = min(population, key=len)
    best_len = len(best_blocks)
    
    # Main evolution loop
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while time.time() - start_time < time_limit:
        # Pick a random individual to evolve
        idx = rng.randrange(len(population))
        current_blocks = population[idx]
        
        # Choose mutation strategy
        strategy = rng.random()
        
        if strategy < 0.5:
            # Add a new block covering some uncovered t-subsets
            uncovered = set()
            covered = set()
            for b in current_blocks:
                for ts in itertools.combinations(b, t):
                    covered.add(frozenset(ts))
            
            all_tsubs = set(frozenset(c) for c in itertools.combinations(range(v), t))
            uncovered = all_tsubs - covered
            
            if uncovered:
                seed_ts = rng.choice(list(uncovered))
                new_block = list(seed_ts)
                rng.shuffle(new_block)
                
                while len(new_block) < k:
                    best_x = None
                    best_gain = -1
                    for x in range(v):
                        if x in new_block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(new_block, t - 1):
                            new_ts = frozenset(rest + (x,))
                            if new_ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                        elif gain == best_gain and best_x is not None and rng.random() < 0.5:
                            best_x = x
                    
                    if best_x is None or best_gain <= 0:
                        break
                    new_block.append(best_x)
                
                new_block.sort()
                candidate = current_blocks + [new_block]
            else:
                candidate = current_blocks
                
        elif strategy < 0.8:
            # Try to replace a block
            if not current_blocks:
                candidate = current_blocks
            else:
                replace_idx = rng.randrange(len(current_blocks))
                # Build uncovered set for efficiency
                covered = set()
                for b in current_blocks:
                    for ts in itertools.combinations(b, t):
                        covered.add(frozenset(ts))
                all_tsubs = set(frozenset(c) for c in itertools.combinations(range(v), t))
                uncovered_set = all_tsubs - covered
                
                new_block = _try_replace_block(
                    current_blocks, replace_idx, v, k, t, uncovered_set, rng, time_limit
                )
                
                if new_block is None:
                    # Remove the block
                    candidate = [b for i, b in enumerate(current_blocks) if i != replace_idx]
                else:
                    candidate = current_blocks.copy()
                    candidate[replace_idx] = new_block
                    
        else:
            # Crossover: combine parts of two random solutions
            if len(population) < 2:
                candidate = current_blocks
            else:
                other_idx = rng.randrange(len(population))
                if other_idx == idx:
                    other_idx = (other_idx + 1) % len(population)
                other_blocks = population[other_idx]
                
                # Take a random subset of blocks from each, merge, then greedily complete
                n1 = rng.randrange(max(1, len(current_blocks) // 2 + 1))
                n2 = rng.randrange(max(1, len(other_blocks) // 2 + 1))
                
                merged = rng.sample(current_blocks, min(n1, len(current_blocks))) + \
                         rng.sample(other_blocks, min(n2, len(other_blocks)))
                
                # Remove duplicates
                seen = set()
                unique_merged = []
                for b in merged:
                    key = tuple(b)
                    if key not in seen:
                        seen.add(key)
                        unique_merged.append(b)
                
                # Greedily add blocks to cover remaining
                covered = set()
                for b in unique_merged:
                    for ts in itertools.combinations(b, t):
                        covered.add(frozenset(ts))
                all_tsubs = set(frozenset(c) for c in itertools.combinations(range(v), t))
                uncovered = all_tsubs - covered
                
                candidate = unique_merged
                while uncovered:
                    seed_ts = rng.choice(list(uncovered))
                    new_block = list(seed_ts)
                    rng.shuffle(new_block)
                    
                    while len(new_block) < k:
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in new_block:
                                continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t - 1):
                                new_ts = frozenset(rest + (x,))
                                if new_ts in uncovered:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                            elif gain == best_gain and best_x is not None and rng.random() < 0.5:
                                best_x = x
                        
                        if best_x is None or best_gain <= 0:
                            break
                        new_block.append(best_x)
                    
                    new_block.sort()
                    candidate.append(new_block)
                    
                    for ts in itertools.combinations(new_block, t):
                        uncovered.discard(frozenset(ts))
        
        # Evaluate candidate
        cand_len = len(candidate)
        cur_len = len(current_blocks)
        
        # Simulated annealing acceptance
        if cand_len < cur_len:
            accept = True
        elif cand_len == cur_len:
            accept = rng.random() < 0.5
        else:
            delta = cand_len - cur_len
            accept_prob = min(1.0, temperature / max(1, delta))
            accept = rng.random() < accept_prob
        
        if accept:
            population[idx] = candidate
            if cand_len < best_len:
                best_len = cand_len
                best_blocks = candidate
        
        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Occasionally re-inject a fresh greedy solution
        if rng.random() < 0.05 and time.time() - start_time < time_limit - 10:
            fresh = _greedy_covering(v, k, t, rng, max_blocks=150)
            worst_idx = max(range(len(population)), key=lambda i: len(population[i]))
            if len(fresh) < len(population[worst_idx]):
                population[worst_idx] = fresh
    
    # Final local search on best solution
    if time.time() - start_time < time_limit - 5:
        final_blocks = best_blocks.copy()
        improved = True
        while improved and time.time() - start_time < time_limit - 3:
            improved = False
            for i in range(len(final_blocks)):
                # Try removing block i if coverage remains
                test_blocks = [b for j, b in enumerate(final_blocks) if j != i]
                if not test_blocks:
                    continue
                covered = set()
                for b in test_blocks:
                    for ts in itertools.combinations(b, t):
                        covered.add(frozenset(ts))
                all_tsubs = set(frozenset(c) for c in itertools.combinations(range(v), t))
                if len(covered) == len(all_tsubs):
                    final_blocks = test_blocks
                    improved = True
                    break
    
    # Verify and return
    return final_blocks
# EVOLVE-BLOCK-END
