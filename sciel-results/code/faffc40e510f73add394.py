"""
Structural rewrite using a Greedy Set Cover initialization followed by 
an intensive Local Search with Exchange Moves (1-for-0, 1-for-1, 2-for-1).
This approaches the problem as a heuristic solution to the Integer Linear Program:
Minimize sum(x_b) subject to sum_{b in T} x_b >= 1 for all triples T, x_b in {0,1}.
The local search explores the neighborhood defined by these exchange moves.
"""

import itertools
import random
import time
import math

def _precompute_triples(v, t):
    """Precompute all t-subsets and map them to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triple_to_idx[combo] = idx
        triples.append(combo)
    return triples, triple_to_idx

def _get_block_triples(block, triple_to_idx):
    """Get the set of triple indices covered by a block."""
    idxs = set()
    for combo in itertools.combinations(block, 3):
        idx = triple_to_idx.get(combo, -1)
        if idx >= 0:
            idxs.add(idx)
    return idxs

def _greedy_cover(v, k, t, triples, triple_to_idx, rng):
    """
    Greedy Set Cover.
    Selects blocks that cover the most uncovered triples.
    """
    n_triples = len(triples)
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()
    
    # Precompute all possible blocks? No, too many (346k).
    # Instead, generate candidates on the fly or use a heuristic.
    # For speed, we will pick a seed triple and extend it greedily.
    
    while uncovered_count > 0:
        # Find the first uncovered triple to serve as a seed
        seed_idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                seed_idx = i
                break
        
        if seed_idx < 0:
            break
            
        seed_triple = triples[seed_idx]
        block = list(seed_triple)
        
        # Greedily extend the block to size k
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            # Calculate gain for each candidate point
            # Gain is the number of newly covered triples
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Check triples formed by x and pairs from current block
                for rest in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(new_triple, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain < 0:
                # If no gain, pick random
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            
            # Update uncovered
            for combo in itertools.combinations(b_tuple, 3):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1

    return blocks

def _local_search(blocks, v, k, t, triples, triple_to_idx, time_limit=100.0):
    """
    Local Search with Exchange Moves.
    1. Try to remove a block (1-for-0).
    2. Try to swap a block with another (1-for-1) to improve structure.
    3. Try to replace 2 blocks with 1 (2-for-1).
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(42)
    
    best_blocks = list(blocks)
    best_n = len(best_blocks)
    current_blocks = list(blocks)
    current_n = len(current_blocks)
    
    # Precompute coverage counts
    def compute_counts(blk_list):
        counts = [0] * len(triples)
        for b in blk_list:
            for combo in itertools.combinations(b, 3):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def is_valid_counts(counts):
        return all(c >= 1 for c in counts)

    current_counts = compute_counts(current_blocks)
    
    # Cache for block triples
    block_triples_cache = {}
    def get_block_triples(b):
        if b not in block_triples_cache:
            block_triples_cache[b] = _get_block_triples(b, triple_to_idx)
        return block_triples_cache[b]

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Move 1: Try to remove a block
        if rng.random() < 0.5:
            if current_n <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            b = current_blocks[i]
            b_triples = get_block_triples(b)
            
            removable = True
            for idx in b_triples:
                if current_counts[idx] <= 1:
                    removable = False
                    break
            
            if removable:
                new_blocks = current_blocks[:i] + current_blocks[i+1:]
                # Update counts efficiently
                for idx in b_triples:
                    current_counts[idx] -= 1
                current_blocks = new_blocks
                current_n -= 1
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
                    # Clear cache for removed block? It's still in cache, but that's fine.
                continue

        # Move 2: Try to swap a block (1-for-1)
        # We look for a block that covers the same "critical" triples as the old one
        elif rng.random() < 0.8:
            if current_n <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            old_b = current_blocks[i]
            old_triples = get_block_triples(old_b)
            
            # Find triples in old_b that are covered only by old_b (critical)
            critical = set()
            for idx in old_triples:
                if current_counts[idx] == 1:
                    critical.add(idx)
            
            if not critical:
                continue
            
            # Try to find a new block that covers all critical triples
            # and is different from old_b
            # Generate candidates by picking points from critical triples
            # For efficiency, we just try a few random blocks that cover the critical set
            
            # Heuristic: Pick one critical triple, generate random blocks containing it
            crit_list = list(critical)
            if not crit_list:
                continue
            seed_triple = triples[crit_list[0]]
            
            # Try a few random extensions
            found_swap = False
            for _ in range(10):
                new_block = list(seed_triple)
                while len(new_block) < k:
                    # Pick random point, prefer those in other critical triples
                    # Simplified: just random
                    candidates = [x for x in range(v) if x not in new_block]
                    x = rng.choice(candidates)
                    new_block.append(x)
                
                new_block.sort()
                new_b = tuple(new_block)
                
                if new_b == old_b:
                    continue
                if new_b in current_blocks:
                    continue
                
                # Check if new_b covers all critical triples
                new_triples = get_block_triples(new_b)
                if critical.issubset(new_triples):
                    # Check if swap is valid (it is, by construction of critical)
                    # But we must ensure no other triple becomes uncovered
                    # Actually, if we replace old_b with new_b, and new_b covers all criticals,
                    # and all other triples in old_b were covered by others, then it's valid.
                    # Wait, we need to check if any triple in old_b that was NOT critical
                    # becomes uncovered? No, if it was not critical, count >= 2.
                    # Removing old_b reduces count by 1. If count was 2, it becomes 1. Still valid.
                    # So validity is guaranteed if criticals are covered.
                    
                    # Update counts
                    for idx in old_triples:
                        current_counts[idx] -= 1
                    for idx in new_triples:
                        current_counts[idx] += 1
                        
                    current_blocks[i] = new_b
                    found_swap = True
                    break
            
            if not found_swap:
                # Try a different strategy: Just swap with a random block that covers criticals
                pass

        # Move 3: 2-for-1 exchange
        # Remove 2 blocks, add 1 block that covers the union of their critical triples
        else:
            if current_n <= 2:
                continue
            i, j = rng.sample(range(len(current_blocks)), 2)
            b1 = current_blocks[i]
            b2 = current_blocks[j]
            
            t1 = get_block_triples(b1)
            t2 = get_block_triples(b2)
            
            # Criticals for b1 and b2
            crit1 = set()
            crit2 = set()
            for idx in t1:
                if current_counts[idx] == 1:
                    crit1.add(idx)
            for idx in t2:
                if current_counts[idx] == 1:
                    crit2.add(idx)
            
            # If any triple is covered by both b1 and b2 and is critical for one,
            # it's complicated. Let's just look for a block that covers (crit1 U crit2)
            # and is not b1 or b2.
            # Actually, if a triple is in crit1, it's only covered by b1.
            # If it's also in b2, then count >= 2, so it wouldn't be in crit1.
            # So crit1 and crit2 are disjoint.
            
            target_crit = crit1.union(crit2)
            if not target_crit:
                continue
            
            # Try to find a block covering target_crit
            # Pick a seed from target_crit
            seed_idx = list(target_crit)[0]
            seed_triple = triples[seed_idx]
            
            found_2for1 = False
            for _ in range(10):
                new_block = list(seed_triple)
                while len(new_block) < k:
                    candidates = [x for x in range(v) if x not in new_block]
                    x = rng.choice(candidates)
                    new_block.append(x)
                
                new_block.sort()
                new_b = tuple(new_block)
                
                if new_b == b1 or new_b == b2:
                    continue
                if new_b in current_blocks:
                    continue
                
                new_triples = get_block_triples(new_b)
                if target_crit.issubset(new_triples):
                    # Perform exchange
                    # Remove b1, b2
                    # Add new_b
                    
                    # Update counts
                    for idx in t1:
                        current_counts[idx] -= 1
                    for idx in t2:
                        current_counts[idx] -= 1
                    for idx in new_triples:
                        current_counts[idx] += 1
                    
                    # Remove b1, b2 from list
                    # Be careful with indices
                    if i > j:
                        i, j = j, i
                    new_list = current_blocks[:j] + current_blocks[j+1:i] + current_blocks[i+1:]
                    new_list.append(new_b)
                    current_blocks = new_list
                    current_n -= 1
                    
                    if current_n < best_n:
                        best_n = current_n
                        best_blocks = list(current_blocks)
                    found_2for1 = True
                    break

        # Restart if stuck for a long time?
        if iteration % 10000 == 0 and current_n > best_n + 3:
            current_blocks = list(best_blocks)
            current_n = best_n
            current_counts = compute_counts(current_blocks)

    return best_blocks, best_n

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)

    # Step 1: Greedy Initialization
    # Run a few greedy restarts to find a good starting point
    best_greedy = None
    best_greedy_n = float('inf')
    
    greedy_deadline = start_time + 10.0
    
    for restart in range(10):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover(v, k, t, triples, triple_to_idx, r)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover(v, k, t, triples, triple_to_idx, rng)

    # Step 2: Local Search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        best_blocks, best_n = _local_search(best_greedy, v, k, t, triples, triple_to_idx, time_limit=remaining_time - 5.0)
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    return [list(b) for b in best_blocks]

if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    # Verify
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
