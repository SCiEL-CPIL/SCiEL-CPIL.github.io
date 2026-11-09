"""
Combined Snake Construction for Q_9:
- Bitmask-based validity checks (from Inspiration)
- Bidirectional Warnsdorff Growth with lookahead (from Current Program)
- Local Insertion Search for gap-filling (from Current Program)
- Multi-start with diverse randomization
"""
import time
import random

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START

# Precompute neighbors for speed
NEIGHBORS = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute adjacency masks for O(1) conflict checking
# adj_mask[v] has bit i set iff v and i are adjacent
ADJ_MASK = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in NEIGHBORS[v]:
        m |= (1 << nb)
    ADJ_MASK[v] = m


def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Strategy:
    1. Multi-start with diverse random seeds.
    2. Bidirectional growth using Warnsdorff's rule with bitmask optimizations.
    3. Local insertion search to fill gaps.
    4. Backtracking when stuck.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_path = None
    best_len = 0
    
    def is_valid_extension(v, path, used_mask, end_index):
        """
        Check if v can be added at end_index (0 for head, len(path) for tail).
        end_index: -1 means append to tail, 0 means prepend to head.
        """
        if used_mask & (1 << v):
            return False
        
        # v must not be adjacent to any vertex in path except the one it connects to
        if end_index == -1:  # appending to tail
            # v connects to path[-1], so check against path[:-1]
            other_mask = used_mask ^ (1 << path[-1])
        else:  # prepending to head
            # v connects to path[0], so check against path[1:]
            other_mask = used_mask ^ (1 << path[0])
        
        # Check if v is adjacent to any vertex in other_mask
        if ADJ_MASK[v] & other_mask:
            return False
        
        return True

    def count_onward_moves(v, path, used_mask, end_index):
        """
        Count how many valid moves are available from v if v were added.
        end_index: -1 means v becomes new tail, 0 means v becomes new head.
        """
        new_used = used_mask | (1 << v)
        count = 0
        
        if end_index == -1:  # v is new tail
            # Next move w from v: w must not be in new_used, w not adj to path[:-1]
            # w is adj to v. v is adj to path[-1]. So w is at dist 2 from path[-1].
            # Check w against path[:-1] (which is path without the last element)
            other_mask = used_mask  # all old vertices
            for w in NEIGHBORS[v]:
                if new_used & (1 << w):
                    continue
                # w must not be adjacent to any vertex in path except v (which is new)
                # Since v is new tail, the "other" vertices are the old path
                # But w is adj to v, so w is at distance 2 from path[-1]
                # We need to check w against path[:-1]
                # Actually, simpler: w must not be adj to any vertex in old path except possibly path[-1]
                # But w is adj to v, and v is adj to path[-1], so w could be adj to path[-1] only if w==path[-1] which is in used
                # So just check against all old path vertices
                if ADJ_MASK[w] & (used_mask ^ (1 << path[-1])):
                    # w is adj to some vertex in path[:-1], invalid
                    continue
                count += 1
        else:  # v is new head
            other_mask = used_mask  # all old vertices
            for w in NEIGHBORS[v]:
                if new_used & (1 << w):
                    continue
                # w must not be adj to any vertex in path[1:]
                if ADJ_MASK[w] & (used_mask ^ (1 << path[0])):
                    continue
                count += 1
        
        return count

    def warnsdorff_grow(path, used_mask, rng, max_steps=500):
        """
        Bidirectional growth with Warnsdorff's rule.
        """
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            # Get valid moves for tail
            tail_moves = []
            for v in NEIGHBORS[path[-1]]:
                if used_mask & (1 << v):
                    continue
                # v connects to path[-1], check against path[:-1]
                other_mask = used_mask ^ (1 << path[-1])
                if ADJ_MASK[v] & other_mask:
                    continue
                # Count onward moves
                sub_count = count_onward_moves(v, path, used_mask, -1)
                tail_moves.append((sub_count, v))

            # Get valid moves for head
            head_moves = []
            for v in NEIGHBORS[path[0]]:
                if used_mask & (1 << v):
                    continue
                # v connects to path[0], check against path[1:]
                other_mask = used_mask ^ (1 << path[0])
                if ADJ_MASK[v] & other_mask:
                    continue
                # Count onward moves
                sub_count = count_onward_moves(v, path, used_mask, 0)
                head_moves.append((sub_count, v))

            if not tail_moves and not head_moves:
                break

            # Decide which end to grow
            if not tail_moves:
                choose_head = True
            elif not head_moves:
                choose_head = False
            else:
                # Warnsdorff global: pick end with fewer options
                if len(head_moves) < len(tail_moves):
                    choose_head = True
                elif len(tail_moves) < len(head_moves):
                    choose_head = False
                else:
                    choose_head = rng.random() < 0.5

            if choose_head:
                # Pick move with min sub_count
                min_sub = min(m[0] for m in head_moves)
                candidates = [m[1] for m in head_moves if m[0] == min_sub]
                v = rng.choice(candidates)
                path.insert(0, v)
                used_mask |= (1 << v)
            else:
                min_sub = min(m[0] for m in tail_moves)
                candidates = [m[1] for m in tail_moves if m[0] == min_sub]
                v = rng.choice(candidates)
                path.append(v)
                used_mask |= (1 << v)
            
            steps += 1
            
        return path, used_mask

    def local_insert_search(path, used_mask, rng, max_iters=30):
        """
        Try to insert vertices into gaps in the path to extend it.
        """
        current_path = path[:]
        current_used = used_mask
        improved = False
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            if len(current_path) < 2:
                break
                
            # Pick a random gap
            pos = rng.randint(1, len(current_path) - 1)
            a = current_path[pos - 1]
            c = current_path[pos]
            
            # Find v such that v~a, v~c, v not in used, v not adj to others
            candidates = []
            for v in NEIGHBORS[a]:
                if v == c or (current_used & (1 << v)):
                    continue
                if (v ^ c).bit_count() != 1:
                    continue
                # Check validity against rest of path
                # v connects to a and c, so check against path excluding a and c
                other_mask = current_used ^ (1 << a) ^ (1 << c)
                if ADJ_MASK[v] & other_mask:
                    continue
                candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                current_path.insert(pos, v)
                current_used |= (1 << v)
                improved = True
        
        return current_path, current_used if improved else (path, used_mask)

    def aggressive_local_search(path, used_mask, rng):
        """
        More aggressive local improvement: try all possible single insertions
        and extensions, then re-run growth.
        """
        current_path = path[:]
        current_used = used_mask
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try all gaps for insertion
            for pos in range(1, len(current_path)):
                if time.time() > deadline:
                    break
                a = current_path[pos - 1]
                c = current_path[pos]
                
                candidates = []
                for v in NEIGHBORS[a]:
                    if v == c or (current_used & (1 << v)):
                        continue
                    if (v ^ c).bit_count() != 1:
                        continue
                    other_mask = current_used ^ (1 << a) ^ (1 << c)
                    if ADJ_MASK[v] & other_mask:
                        continue
                    candidates.append(v)
                
                if candidates:
                    v = candidates[0]
                    current_path.insert(pos, v)
                    current_used |= (1 << v)
                    improved = True
                    break
            
            # Try extending ends
            if not improved:
                # Try tail extension
                for v in NEIGHBORS[current_path[-1]]:
                    if current_used & (1 << v):
                        continue
                    other_mask = current_used ^ (1 << current_path[-1])
                    if ADJ_MASK[v] & other_mask:
                        continue
                    current_path.append(v)
                    current_used |= (1 << v)
                    improved = True
                    break
                
                # Try head extension
                if not improved:
                    for v in NEIGHBORS[current_path[0]]:
                        if current_used & (1 << v):
                            continue
                        other_mask = current_used ^ (1 << current_path[0])
                        if ADJ_MASK[v] & other_mask:
                            continue
                        current_path.insert(0, v)
                        current_used |= (1 << v)
                        improved = True
                        break
        
        return current_path, current_used

    def backtrack_retry(path, used_mask, rng, depth=5):
        """
        When stuck, try backtracking to a previous branching point.
        """
        if len(path) < 3:
            return path, used_mask, False
        
        for i in range(min(depth, len(path) - 2)):
            # Remove last i+1 vertices
            remove_count = i + 1
            body = path[:-remove_count]
            body_used = 0
            for v in body:
                body_used |= (1 << v)
            
            end_vertex = body[-1]
            
            # Get valid moves from end_vertex
            valid = []
            for v in NEIGHBORS[end_vertex]:
                if body_used & (1 << v):
                    continue
                other_mask = body_used ^ (1 << end_vertex)
                if ADJ_MASK[v] & other_mask:
                    continue
                valid.append(v)
            
            if valid:
                # Pick a random valid move
                rng.shuffle(valid)
                v = valid[0]
                new_path = body + [v]
                new_used = body_used | (1 << v)
                # Try to regrow
                new_path, new_used = warnsdorff_grow(new_path, new_used, rng, max_steps=50)
                return new_path, new_used, True
        
        return path, used_mask, False

    def run_one_attempt(rng):
        """Execute one full construction attempt."""
        # Initialize with a random starting edge
        v0 = rng.randint(0, MAX_VERTS - 1)
        path = [v0]
        used_mask = 1 << v0
        
        v1 = rng.choice(NEIGHBORS[v0])
        path.append(v1)
        used_mask |= (1 << v1)
        
        # Bidirectional Warnsdorff Growth
        path, used_mask = warnsdorff_grow(path, used_mask, rng, max_steps=300)
        
        # Aggressive local search
        path, used_mask = aggressive_local_search(path, used_mask, rng)
        
        # If still stuck, try backtracking
        if len(path) < 200:
            path, used_mask, success = backtrack_retry(path, used_mask, rng, depth=8)
            if success:
                path, used_mask = aggressive_local_search(path, used_mask, rng)
        
        return path

    # Main Loop: Multiple Restarts with adaptive strategy
    restarts = 0
    consecutive_no_improve = 0
    while time.time() < deadline:
        restarts += 1
        seed = restarts * 104729 + 7
        rng = random.Random(seed)
        
        # Adaptive: if we're not improving, try more aggressive strategies
        if consecutive_no_improve > 5:
            # Use a different random seed pattern for more diversity
            seed = restarts * 31337 + 13
            rng = random.Random(seed)
        
        path = run_one_attempt(rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # If we've hit a very high score, continue to see if we can do better
        if best_len > 200:
            # Reduce max steps for subsequent attempts to allow more restarts
            pass

    if best_path is None:
        # Fallback: Simple greedy
        path = [0]
        used_mask = 1
        while True:
            found = False
            for v in NEIGHBORS[path[-1]]:
                if used_mask & (1 << v):
                    continue
                other_mask = used_mask ^ (1 << path[-1])
                if ADJ_MASK[v] & other_mask:
                    continue
                path.append(v)
                used_mask |= (1 << v)
                found = True
                break
            if not found:
                break
        return path

    # Final: try to improve best_path with aggressive local search
    if best_path is not None and time.time() < deadline - 5:
        # Reconstruct used_mask
        used_mask = 0
        for v in best_path:
            used_mask |= (1 << v)
        
        improved_path, improved_mask = aggressive_local_search(best_path, used_mask, random.Random(42))
        if len(improved_path) > best_len:
            best_len = len(improved_path)
            best_path = improved_path[:]

    return best_path

# EVOLVE-BLOCK-END
