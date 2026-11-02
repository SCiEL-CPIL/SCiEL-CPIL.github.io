import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    total_tsubs = len(all_tsubs)
    
    def block_tsubs(block):
        return set(itertools.combinations(block, t))
    
    def compute_coverage(blocks):
        """Compute coverage counts for all t-subsets."""
        cov = defaultdict(int)
        for b in blocks:
            for tri in itertools.combinations(b, t):
                cov[tri] += 1
        return cov
    
    def is_valid(blocks, cov=None):
        """Check if all t-subsets are covered."""
        if cov is None:
            cov = compute_coverage(blocks)
        for tri in all_tsubs:
            if cov.get(tri, 0) == 0:
                return False
        return True
    
    def remove_redundant(blocks, cov=None):
        """Remove blocks that don't contribute unique coverage."""
        if cov is None:
            cov = compute_coverage(blocks)
        
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i has any uniquely covered t-subset
                unique = False
                for tri in itertools.combinations(blocks[i], t):
                    if cov[tri] == 1:
                        unique = True
                        break
                if not unique:
                    # Remove this block
                    for tri in itertools.combinations(blocks[i], t):
                        cov[tri] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks, cov
    
    def greedy_construct(seed_triple=None, max_blocks=200):
        """Greedy construction starting from an optional seed triple."""
        uncovered = set(all_tsubs_set)
        blocks = []
        
        if seed_triple is not None:
            # Start with a block containing the seed triple
            pivot_points = set(seed_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    uncovered.discard(tri)
        
        while uncovered and len(blocks) < max_blocks:
            # Pick a random uncovered triple as pivot
            pivot_triple = random.choice(list(uncovered))
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    uncovered.discard(tri)
            else:
                break
        
        return blocks
    
    def local_search(blocks, max_time=30.0):
        """Local search to minimize block count."""
        if time.time() > start_time + max_time:
            return blocks
        
        blocks = [list(b) for b in blocks]
        cov = compute_coverage(blocks)
        
        # Initial redundancy removal
        blocks, cov = remove_redundant(blocks, cov)
        
        T = 30.0  # Initial temperature
        cooling = 0.96
        best_blocks = [list(b) for b in blocks]
        best_n = len(blocks)
        
        while time.time() < start_time + max_time and T > 0.5:
            if len(blocks) < best_n:
                best_n = len(blocks)
                best_blocks = [list(b) for b in blocks]
            
            n = len(blocks)
            if n == 0:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Try removing a block
                i = random.randint(0, n - 1)
                removable = True
                for tri in itertools.combinations(blocks[i], t):
                    if cov[tri] == 1:
                        removable = False
                        break
                
                if removable:
                    for tri in itertools.combinations(blocks[i], t):
                        cov[tri] -= 1
                    blocks.pop(i)
                    n -= 1
                else:
                    # Try to find a replacement for block i
                    only_tris = [tri for tri in itertools.combinations(blocks[i], t) if cov[tri] == 1]
                    
                    if not only_tris:
                        # Block is redundant
                        for tri in itertools.combinations(blocks[i], t):
                            cov[tri] -= 1
                        blocks.pop(i)
                        n -= 1
                    else:
                        # Must cover all only_tris
                        must_include = set()
                        for tri in only_tris:
                            must_include.update(tri)
                        
                        if len(must_include) <= k:
                            remaining_points = [p for p in range(v) if p not in must_include]
                            need = k - len(must_include)
                            
                            if need >= 0 and need <= len(remaining_points):
                                found = False
                                for _ in range(15):
                                    if need == 0:
                                        candidate = tuple(sorted(must_include))
                                    else:
                                        extra = random.sample(remaining_points, need)
                                        candidate = tuple(sorted(must_include | set(extra)))
                                    
                                    cand_tris = set(itertools.combinations(candidate, t))
                                    covers_all = all(tri in cand_tris for tri in only_tris)
                                    
                                    if covers_all:
                                        # Calculate delta
                                        old_tris = set(itertools.combinations(blocks[i], t))
                                        delta = 0
                                        for tri in cand_tris - old_tris:
                                            if cov.get(tri, 0) == 0:
                                                delta += 1
                                        for tri in old_tris - cand_tris:
                                            if cov[tri] == 1:
                                                delta -= 1
                                        
                                        if delta >= 0 or random.random() < np.exp(-delta / T):
                                            # Apply swap
                                            for tri in old_tris:
                                                cov[tri] -= 1
                                            for tri in cand_tris:
                                                cov[tri] += 1
                                            blocks[i] = list(candidate)
                                            found = True
                                            break
                                
                                if not found:
                                    pass  # No change
            
            elif move_type < 0.7:
                # Try adding a block then removing a redundant one
                if n < 150:
                    # Find a triple with low coverage
                    low_cov_tris = [tri for tri in all_tsubs if cov.get(tri, 0) <= 2]
                    if low_cov_tris:
                        target_tri = random.choice(low_cov_tris)
                        target_points = set(target_tri)
                        other_points = [p for p in range(v) if p not in target_points]
                        
                        best_block = None
                        best_gain = -1
                        for _ in range(30):
                            combo = random.sample(other_points, k - t)
                            block = tuple(sorted(target_points | set(combo)))
                            gain = sum(1 for tri in itertools.combinations(block, t) if cov.get(tri, 0) <= 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
                        
                        if best_block is not None and best_gain > 0:
                            new_tris = set(itertools.combinations(best_block, t))
                            for tri in new_tris:
                                cov[tri] += 1
                            blocks.append(list(best_block))
                            n += 1
                            
                            # Try to remove a redundant block
                            for i in range(n - 1, -1, -1):
                                removable = True
                                for tri in itertools.combinations(blocks[i], t):
                                    if cov[tri] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tri in itertools.combinations(blocks[i], t):
                                        cov[tri] -= 1
                                    blocks.pop(i)
                                    n -= 1
                                    break
            
            T *= cooling
        
        # Final redundancy removal
        blocks, _ = remove_redundant(blocks)
        
        if len(blocks) < best_n:
            best_blocks = blocks
        
        return best_blocks
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # First, do a quick greedy to get a valid solution
    initial_blocks = greedy_construct()
    initial_blocks, _ = remove_redundant(initial_blocks)
    if is_valid(initial_blocks):
        best_blocks = list(initial_blocks)
        best_len = len(initial_blocks)
    
    # Multiple random restarts with local search
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        
        # Generate a new random starting point
        seed_triple = random.choice(all_tsubs)
        blocks = greedy_construct(seed_triple)
        
        if not blocks:
            continue
        
        # Check validity
        if not is_valid(blocks):
            continue
        
        # Apply local search
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            blocks = local_search(blocks, max_time=min(25, remaining_time))
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        # If we've made good progress, spend more time polishing
        if iteration >= 8 and time.time() < deadline - 15:
            if best_blocks is not None:
                remaining_time = deadline - time.time() - 5
                if remaining_time > 10:
                    best_blocks = local_search(best_blocks, max_time=remaining_time)
                    if len(best_blocks) < best_len:
                        best_len = len(best_blocks)
                break
    
    # Final polish if we have time
    if best_blocks is not None and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 3
        if remaining_time > 5:
            best_blocks = local_search(best_blocks, max_time=remaining_time)
    
    # Ensure we return a valid solution
    if best_blocks is None or not is_valid(best_blocks):
        # Fallback to a valid greedy solution
        blocks = greedy_construct()
        blocks, _ = remove_redundant(blocks)
        best_blocks = blocks
    
    return best_blocks
