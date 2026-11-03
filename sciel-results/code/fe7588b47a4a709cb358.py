"""Genetic + local search for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))

def _block_t_subsets(block, t):
    return set(itertools.combinations(block, t))

def _greedy_construct(v, k, t, uncovered_set, rng=None):
    """Greedy: repeatedly pick block covering most uncovered t-subsets."""
    if rng is None:
        rng = random.Random()
    blocks = []
    uncovered = set(uncovered_set)
    while uncovered:
        best_block = None
        best_gain = -1
        # Try many random starting points
        for _ in range(min(200, len(uncovered))):
            # Pick a random uncovered t-subset as seed
            seed = rng.choice(list(uncovered)) if uncovered else None
            if seed is None:
                break
            block = set(seed)
            # Greedily add points that cover most uncovered t-subsets
            while len(block) < k:
                best_x = None
                best_g = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x is None or best_g <= 0:
                    break
                block.add(best_x)
            if len(block) == k:
                gain = len(_block_t_subsets(tuple(sorted(block)), t) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = tuple(sorted(block))
        if best_block is None:
            # Fallback: just take any uncovered subset and fill randomly
            seed = tuple(uncovered.pop())
            block = list(seed)
            remaining = [x for x in range(v) if x not in block]
            rng.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
            best_block = tuple(sorted(block))
        blocks.append(list(best_block))
        uncovered.difference_update(_block_t_subsets(best_block, t))
    return blocks

def _local_search(v, k, t, blocks, max_rounds=50, rng=None, deadline=None):
    """Try to remove blocks and add better ones."""
    if rng is None:
        rng = random.Random()
    blocks = [list(b) for b in blocks]
    all_ts = _all_t_subsets(v, t)
    
    for _ in range(max_rounds):
        if deadline and time.time() > deadline - 5:
            break
        improved = False
        # Try removing each block
        for i in range(len(blocks)):
            if deadline and time.time() > deadline - 5:
                break
            remaining = blocks[:i] + blocks[i+1:]
            # Check if all t-subsets still covered
            covered = set()
            for b in remaining:
                covered.update(_block_t_subsets(tuple(b), t))
            if len(covered) == len(all_ts):
                blocks = remaining
                improved = True
                break
        if improved:
            continue
        # Try adding a block that covers most uncovered
        covered = set()
        for b in blocks:
            covered.update(_block_t_subsets(tuple(b), t))
        uncovered = set(all_ts) - covered
        if not uncovered:
            break
        # Try to find a block that covers many uncovered
        best_block = None
        best_gain = 0
        for _ in range(50):
            seed = rng.choice(list(uncovered))
            block = set(seed)
            while len(block) < k:
                best_x = None
                best_g = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = sum(1 for rest in itertools.combinations(block, t-1)
                           if tuple(sorted(rest + (x,))) in uncovered)
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x is None:
                    break
                block.add(best_x)
            if len(block) == k:
                gain = len(_block_t_subsets(tuple(sorted(block)), t) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = tuple(sorted(block))
        if best_block and best_gain > 0:
            # Only add if it helps (we'll try to remove later)
            pass  # Don't add blocks in local search, only remove
    
    return blocks

def _crossover(v, k, t, parent1, parent2, rng):
    """Combine blocks from two parents, repair coverage."""
    # Randomly select blocks from each parent
    combined = list(parent1) + list(parent2)
    # Remove duplicates
    seen = set()
    unique = []
    for b in combined:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique.append(list(b))
    # Sort by coverage potential (heuristic: keep blocks that cover rare t-subsets)
    all_ts = _all_t_subsets(v, t)
    
    # Greedy repair: start empty, add blocks that cover most uncovered
    uncovered = set(all_ts)
    result = []
    available = list(unique)
    rng.shuffle(available)
    
    while uncovered and available:
        best_idx = -1
        best_gain = -1
        for i, b in enumerate(available):
            gain = len(_block_t_subsets(tuple(b), t) & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0 and best_gain > 0:
            result.append(available[best_idx])
            uncovered.difference_update(_block_t_subsets(tuple(available[best_idx]), t))
            available.pop(best_idx)
        else:
            break
    
    # If still uncovered, fill with greedy
    if uncovered:
        extra = _greedy_construct(v, k, t, uncovered, rng)
        result.extend(extra)
    
    return result

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    rng = random.Random(42)
    all_ts = _all_t_subsets(v, t)
    
    # Initial greedy solution
    best = _greedy_construct(v, k, t, all_ts, rng)
    best = _local_search(v, k, t, best, max_rounds=30, rng=rng, deadline=deadline)
    
    # Genetic algorithm
    population = [best]
    for i in range(1, min(5, len(all_ts))):
        if time.time() > deadline - 10:
            break
        pop = _greedy_construct(v, k, t, all_ts, rng)
        pop = _local_search(v, k, t, pop, max_rounds=10, rng=rng, deadline=deadline)
        population.append(pop)
    
    generations = 0
    while time.time() < deadline - 5:
        generations += 1
        new_pop = []
        
        # Elitism
        new_pop.append(best)
        
        # Crossover and mutation
        for i in range(len(population)):
            if time.time() > deadline - 5:
                break
            p1 = population[i]
            p2 = population[(i + 1) % len(population)]
            
            child = _crossover(v, k, t, p1, p2, rng)
            
            # Mutation: try local search
            child = _local_search(v, k, t, child, max_rounds=5, rng=rng, deadline=deadline)
            
            new_pop.append(child)
            
            # Update best
            if len(child) < len(best):
                best = child
        
        population = new_pop[:5]  # Keep top 5
    
    return best
# EVOLVE-BLOCK-END
