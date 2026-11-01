import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    
    # Precompute for each block (k-subset), which t-subsets it covers
    # But C(24,7) is 346104, and each has C(7,3)=35 triples, so 12M triples total - too much to precompute all
    
    # Instead, work with coverage counts
    # coverage_count[ts] = number of blocks covering triple ts
    
    def block_triples(block):
        return set(itertools.combinations(block, t))
    
    # Phase 1: Randomized greedy to get initial covering
    random.seed(42)
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple to start
        start_triple = random.choice(list(uncovered))
        block = list(start_triple)
        random.shuffle(block)
        
        while len(block) < k:
            # Find point that covers most uncovered triples
            best_x = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            for x in candidates:
                # Count how many uncovered triples would be covered
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(block_triples(block))
        
        if time.time() - start_time > time_limit * 0.3:
            break
    
    # If greedy didn't finish, finish it deterministically
    if uncovered:
        while uncovered:
            start_triple = min(uncovered)
            block = list(start_triple)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(block_triples(block))
    
    # Phase 2: Local search to minimize blocks
    # Maintain coverage counts
    coverage = {}
    for ts in all_tsubs:
        coverage[ts] = 0
    for block in blocks:
        for ts in block_triples(block):
            coverage[ts] += 1
    
    def is_valid():
        return all(coverage[ts] >= 1 for ts in all_tsubs)
    
    def try_delete_block(idx):
        """Try to delete block at index idx. Returns True if successful."""
        block = blocks[idx]
        # Check if all triples in this block are covered by other blocks
        for ts in block_triples(block):
            if coverage[ts] <= 1:
                return False
        # Delete it
        for ts in block_triples(block):
            coverage[ts] -= 1
        blocks.pop(idx)
        return True
    
    def try_swap_point(idx, old_x, new_x):
        """Try to swap old_x with new_x in block idx. Returns True if successful."""
        if new_x in blocks[idx]:
            return False
        block = blocks[idx]
        # Compute coverage change
        # Triples lost: those containing old_x but not new_x
        # Triples gained: those containing new_x but not old_x
        other_points = [p for p in block if p != old_x]
        
        lost = 0
        gained = 0
        # Triples with old_x: choose t-1 from other_points
        for rest in itertools.combinations(other_points, t-1):
            ts_old = tuple(sorted(rest + (old_x,)))
            ts_new = tuple(sorted(rest + (new_x,)))
            if coverage[ts_old] <= 1:
                return False  # Would leave ts_old uncovered
            if ts_new not in coverage or coverage[ts_new] == 0:
                gained += 1
        
        # Actually, we need to be more careful. Let's just simulate.
        # Remove old block's contribution, add new block's contribution
        old_block_triples = block_triples(block)
        new_block = sorted([new_x] + other_points)
        new_block_triples = block_triples(new_block)
        
        # Check all triples that would lose coverage
        for ts in old_block_triples:
            if coverage[ts] <= 1 and ts not in new_block_triples:
                return False
        
        # Apply swap
        for ts in old_block_triples:
            coverage[ts] -= 1
        for ts in new_block_triples:
            coverage[ts] += 1
        blocks[idx] = new_block
        return True
    
    # Local search loop
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to delete blocks in random order
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            if try_delete_block(idx):
                improved = True
                best_count = len(blocks)
                best_blocks = list(blocks)
        
        # If no deletion worked, try perturbations
        if not improved and len(blocks) > 0:
            # Try swapping a point in a random block
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                idx = random.randint(0, len(blocks)-1)
                old_x = random.choice(blocks[idx])
                new_x = random.randint(0, v-1)
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                if try_swap_point(idx, old_x, new_x):
                    # After swap, try to delete some blocks
                    indices2 = list(range(len(blocks)))
                    random.shuffle(indices2)
                    for idx2 in indices2:
                        if try_delete_block(idx2):
                            improved = True
                            best_count = len(blocks)
                            best_blocks = list(blocks)
                            break
    
    return best_blocks
