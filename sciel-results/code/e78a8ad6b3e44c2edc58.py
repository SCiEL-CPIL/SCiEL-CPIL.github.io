"""Randomized local search with block-intersection graph pruning for C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict
import heapq


def _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(tsubset_indices)))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = tsubset_indices[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_block_idx.get(fs)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        blocks.append(b_tuple)
        for c in itertools.combinations(block, t):
            uncovered.discard(subset_to_block_idx[frozenset(c)])
                
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all t-subsets and map frozenset -> index
    tsubset_indices = []
    subset_to_block_idx = {}
    for c in itertools.combinations(range(v), t):
        fs = frozenset(c)
        subset_to_block_idx[fs] = len(tsubset_indices)
        tsubset_indices.append(c)
    
    total_t = len(tsubset_indices)
    
    # Precompute triple indices for all possible k-subsets (blocks)
    block_triple_indices = {}
    for b in itertools.combinations(range(v), k):
        block_triple_indices[b] = [subset_to_block_idx[frozenset(c)] for c in itertools.combinations(b, t)]
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx)
    
    # Initialize counts array
    counts = [0] * total_t
    for b in blocks:
        for idx in block_triple_indices[tuple(b)]:
            counts[idx] += 1
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Block intersection graph: adj[i] = set of block indices sharing >=1 point with block i
    # We maintain this incrementally as blocks are added/removed/swapped.
    # For efficiency, store as list of sets.
    
    def build_intersection_graph():
        """Build the block intersection graph from scratch."""
        n = len(blocks)
        adj = [set() for _ in range(n)]
        # Point -> list of block indices containing that point
        point_to_blocks = defaultdict(list)
        for i, b in enumerate(blocks):
            for p in b:
                point_to_blocks[p].append(i)
        for p in point_to_blocks:
            blks = point_to_blocks[p]
            for a in range(len(blks)):
                for b_idx in range(a+1, len(blks)):
                    adj[blks[a]].add(blks[b_idx])
                    adj[blks[b_idx]].add(blks[a])
        return adj
    
    adj = build_intersection_graph()
    
    def get_degree(i):
        return len(adj[i])
    
    def try_remove_block(blk_idx):
        """
        Try to remove block at blk_idx.
        Returns True if successful, False otherwise.
        """
        b = blocks[blk_idx]
        for idx in block_triple_indices[tuple(b)]:
            if counts[idx] <= 1:
                return False
        
        # It is valid. Perform removal.
        for idx in block_triple_indices[tuple(b)]:
            counts[idx] -= 1
        
        # Update intersection graph: remove blk_idx from neighbors
        n = len(blocks)
        neighbors = list(adj[blk_idx])
        for nb in neighbors:
            adj[nb].discard(blk_idx)
        adj[blk_idx] = set()
        
        blocks.pop(blk_idx)
        # Reindex adj: shift indices > blk_idx down by 1
        # This is expensive but necessary for correctness.
        # Alternative: use a different data structure. For now, rebuild if needed.
        # Actually, let's just rebuild the graph after structural changes for safety.
        # But that's O(n^2). Let's do incremental removal properly.
        # Since we're popping from the middle, we need to reindex.
        # Simpler: just rebuild the graph.
        adj[:] = build_intersection_graph()
        return True
    
    def try_merge_two_blocks(i, j):
        """
        Try to replace blocks[i] and blocks[j] with a single new block.
        Returns True if successful, False otherwise.
        """
        b1 = blocks[i]
        b2 = blocks[j]
        
        # The new block must cover all triples that are uniquely covered by b1 or b2
        critical_points = set()
        for idx in block_triple_indices[tuple(b1)]:
            if counts[idx] == 1:
                critical_points.update(tsubset_indices[idx])
        for idx in block_triple_indices[tuple(b2)]:
            if counts[idx] == 1:
                critical_points.update(tsubset_indices[idx])
        
        if len(critical_points) > k:
            return False
        
        # Try to form a new block containing critical_points
        new_block_set = set(critical_points)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        # Greedy fill to maximize coverage of single-covered triples
        for x in cands:
            if len(new_block_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_block_set, t-1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_block_idx.get(fs)
                if idx is not None and counts[idx] == 1:
                    gain += 1
            if gain > 0:
                new_block_set.add(x)
            elif len(new_block_set) < k:
                new_block_set.add(x)
        
        if len(new_block_set) != k:
            return False
        
        new_block = tuple(sorted(new_block_set))
        
        # Check if new_block is same as either old block
        if new_block == tuple(b1) or new_block == tuple(b2):
            return False
        
        # Validate: after removing b1 and b2, adding new_block must keep all counts >= 1
        new_block_idx_set = set(block_triple_indices[new_block])
        valid = True
        for idx in block_triple_indices[tuple(b1)]:
            if counts[idx] == 1 and idx not in new_block_idx_set:
                valid = False
                break
        if valid:
            for idx in block_triple_indices[tuple(b2)]:
                if counts[idx] == 1 and idx not in new_block_idx_set:
                    valid = False
                    break
        
        if not valid:
            return False
        
        # Accept merge
        for idx in block_triple_indices[tuple(b1)]:
            counts[idx] -= 1
        for idx in block_triple_indices[tuple(b2)]:
            counts[idx] -= 1
        for idx in block_triple_indices[new_block]:
            counts[idx] += 1
        
        # Remove the later index first to avoid shifting
        if i > j:
            i, j = j, i
        blocks.pop(j)
        blocks.pop(i)
        blocks.append(list(new_block))
        adj[:] = build_intersection_graph()
        return True
    
    # Simulated Annealing / Local Search Loop
    iteration = 0
    no_improve = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block, prioritizing low-degree (peripheral) blocks
        removed = False
        n_blocks = len(blocks)
        if n_blocks > 0:
            # Create priority list: sort by degree (ascending) with some randomness
            # Low-degree blocks are more likely to be removable without breaking coverage
            degree_list = [(get_degree(i), i) for i in range(n_blocks)]
            rng.shuffle(degree_list)
            degree_list.sort(key=lambda x: x[0])  # sort by degree ascending
            # Try the lowest-degree blocks first, but with some randomization
            for deg, i in degree_list:
                if try_remove_block(i):
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                        no_improve = 0
                    else:
                        no_improve += 1
                    removed = True
                    break
                # Early exit if too much time spent
                if time.time() - start_time > time_limit - 5:
                    break
        
        if removed:
            continue
        
        # 2. Try to merge two blocks into one, prioritizing pairs with low combined degree
        merged = False
        if len(blocks) >= 2:
            n_blocks = len(blocks)
            # Generate candidate pairs weighted by low degree
            # Prefer pairs where both blocks have low degree
            degree_list = [(get_degree(i), i) for i in range(n_blocks)]
            degree_list.sort(key=lambda x: x[0])
            # Try pairs from the low-degree end first
            low_degree_blocks = [i for deg, i in degree_list[:min(15, n_blocks)]]
            rng.shuffle(low_degree_blocks)
            
            pairs_tried = 0
            for _ in range(50):
                if pairs_tried >= 50:
                    break
                if len(low_degree_blocks) >= 2:
                    i, j = rng.sample(low_degree_blocks, 2)
                else:
                    i, j = rng.sample(range(n_blocks), 2)
                pairs_tried += 1
                if try_merge_two_blocks(i, j):
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                        no_improve = 0
                    else:
                        no_improve += 1
                    merged = True
                    break
        
        if merged:
            continue
        
        # 3. Try to swap a block, prioritizing low-degree blocks
        if len(blocks) > 0:
            n_blocks = len(blocks)
            # Pick a low-degree block to swap
            degree_list = [(get_degree(i), i) for i in range(n_blocks)]
            rng.shuffle(degree_list)
            degree_list.sort(key=lambda x: x[0])
            i = degree_list[0][1]  # lowest degree block
            old_block = blocks[i]
            
            critical_indices = []
            for idx in block_triple_indices[tuple(old_block)]:
                if counts[idx] == 1:
                    critical_indices.append(idx)
            
            points_needed = set()
            for idx in critical_indices:
                points_needed.update(tsubset_indices[idx])
                
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = subset_to_block_idx.get(fs)
                        if idx is not None:
                            c = counts[idx]
                            if c == 0:
                                gain += 10
                            elif c == 1:
                                gain += 1
                    if gain > 0:
                        new_block_set.add(x)
                    elif len(new_block_set) < k:
                        new_block_set.add(x)
                        
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    
                    if new_block == tuple(old_block):
                        continue
                        
                    old_block_indices = block_triple_indices[tuple(old_block)]
                    new_block_indices = block_triple_indices[new_block]
                    
                    new_block_idx_set = set(new_block_indices)
                    valid = True
                    for idx in old_block_indices:
                        if counts[idx] == 1 and idx not in new_block_idx_set:
                            valid = False
                            break
                    
                    if valid:
                        for idx in old_block_indices:
                            counts[idx] -= 1
                        for idx in new_block_indices:
                            counts[idx] += 1
                        blocks[i] = list(new_block)
                        adj[:] = build_intersection_graph()
                        no_improve = 0
        
        # 4. Periodic greedy restart if stuck
        if no_improve > 500 or iteration % 200 == 0:
            new_cover = _greedy_cover(v, k, t, rng, tsubset_indices, subset_to_block_idx)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = [0] * total_t
                for b in blocks:
                    for idx in block_triple_indices[tuple(b)]:
                        counts[idx] += 1
                adj[:] = build_intersection_graph()
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    no_improve = 0
                else:
                    no_improve = 0
            else:
                no_improve += 100
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            no_improve = 0
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
