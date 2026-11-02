import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed: idx_3d[a][b][c] = index of triple (a,b,c)
    # -1 if not a valid triple (i.e., a==b etc, though we ensure distinct in logic)
    idx_3d = [[[-1]*v for _ in range(v)] for _ in range(v)]
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a][b][c] = triple_to_idx[(t1, t2, t3)]
        
    def get_triple_idx(a, b, c):
        return idx_3d[a][b][c]
    
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            # Pick a random uncovered triple to break symmetry
            # Find all uncovered indices
            uncovered_indices = [i for i in range(num_triples) if uncovered[i]]
            if not uncovered_indices:
                break
            
            target_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            target_triple = triples[target_idx]
            block = list(target_triple)
            
            # Mark this triple as covered
            uncovered[target_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Count triples (x, u, v) where u, v in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            idx = get_triple_idx(x, p1, p2)
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    
                if best_point == -1:
                    break
                
                block.append(best_point)
                
                # Update uncovered status
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def probabilistic_cover(random_state):
        """
        Uses the Alter-Broder-Frieze-Karlin-Peierls derandomization approach:
        1. Sample random blocks with probability p
        2. Deterministically prune to maintain a valid covering
        3. Minimize expected block count
        """
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        
        # Calculate optimal sampling probability
        # Each triple is in C(v-t, k-t) / C(v, k) blocks
        # We want each triple to be covered with high probability
        # Use p = (ln(num_triples) + c) / C(v-t, k-t)
        from math import log
        blocks_per_triple = len(list(itertools.combinations(range(v), k)))  # This is wrong, need C(v-t, k-t)
        # Actually, for a fixed triple, number of blocks containing it = C(v-t, k-t)
        blocks_per_triple = 1
        # Calculate C(v-t, k-t)
        vt = v - t
        kt = k - t
        if vt >= kt >= 0:
            # C(vt, kt)
            blocks_per_triple = 1
            for i in range(kt):
                blocks_per_triple *= (vt - i)
                blocks_per_triple //= (i + 1)
        
        # Optimal p for derandomization
        p = (log(num_triples) + 1.0) / blocks_per_triple
        p = max(p, 0.01)  # Ensure minimum probability
        p = min(p, 1.0)
        
        # Phase 1: Random sampling
        all_blocks = list(itertools.combinations(range(v), k))
        rng.shuffle(all_blocks)
        
        for block in all_blocks:
            if rng.random() < p:
                blocks.append(tuple(sorted(block)))
        
        # Phase 2: Deterministic pruning
        # Count coverage
        triple_counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                triple_counts[idx] += 1
        
        # Check if we have a valid covering
        uncovered_indices = [i for i in range(num_triples) if triple_counts[i] == 0]
        
        # If not valid, add blocks greedily to cover uncovered triples
        while uncovered_indices:
            # Pick a random uncovered triple
            target_idx = uncovered_indices[0]
            target_triple = triples[target_idx]
            
            # Find a block containing this triple
            # Sample random blocks until we find one
            found = False
            for _ in range(1000):  # Limit attempts
                candidate = tuple(sorted(rng.sample(range(v), k)))
                if all(x in candidate for x in target_triple):
                    blocks.append(candidate)
                    found = True
                    break
            
            if not found:
                # Fallback: construct a block containing the triple
                block = list(target_triple)
                remaining = [x for x in range(v) if x not in block]
                rng.shuffle(remaining)
                block.extend(remaining[:k-t])
                blocks.append(tuple(sorted(block)))
            
            # Update counts
            for triple in itertools.combinations(blocks[-1], t):
                idx = triple_to_idx[triple]
                triple_counts[idx] += 1
            
            # Recalculate uncovered
            uncovered_indices = [i for i in range(num_triples) if triple_counts[i] == 0]
        
        # Phase 3: Trim redundant blocks
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active[i]:
                    continue
                # Check if any triple in this block has count 1
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if triple_counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        triple_counts[idx] -= 1
        
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def trim_blocks(blocks):
        # Count frequency of each triple
        counts = [0] * num_triples
        
        for block in blocks:
            # Iterate all triples in the block
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                # Check if any triple in this block has count 1
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Run a mix of greedy and probabilistic methods
    seed_counter = 0
    method_idx = 0
    
    while time.time() < deadline:
        # Alternate between methods
        if method_idx % 3 == 0:
            blocks = greedy_cover(seed_counter)
        else:
            blocks = probabilistic_cover(seed_counter)
        
        blocks = trim_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Optional: If we hit the lower bound, stop?
        # Lower bound is 69.
        if best_len == 69:
            break
            
        seed_counter += 1
        method_idx += 1
    
    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return empty or a trivial covering?
        # Should not happen.
        return []
