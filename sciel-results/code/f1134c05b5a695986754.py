"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _bfs_threshold(ta, tb):
    """
    Compute the exact shortest reset word length via BFS on the subset automaton.
    ta, tb: tuples of length N (destination states for each state).
    Returns (threshold, is_synchronizing).
    Uses precomputed bitmask transitions for O(1) subset updates.
    """
    # Precompute transition for all 2^N subsets for letter 'a' and 'b'
    # trans_a[mask] = new_mask after applying 'a'
    # trans_b[mask] = new_mask after applying 'b'
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    # Optimization: Precompute for all masks
    # For each state s, let bit_s = 1 << s
    # Let dest_a_s = 1 << ta[s]
    # Let dest_b_s = 1 << tb[s]
    # We can compute trans_a[mask] by iterating bits? No, that's O(N) per mask.
    # Better: DP or simple iteration for each mask is O(N * 2^N) which is 12 * 4096 ~ 50k ops. Very fast.
    
    for mask in range(size):
        # Calculate image under 'a'
        img_a = 0
        # We can iterate through set bits of mask
        # Or simply: for s in range(N): if mask & (1<<s): img_a |= 1 << ta[s]
        # This is 12 iterations per mask. 4096 * 12 = 49152 iterations.
        # Let's do it efficiently.
        m = mask
        while m:
            lsb = m & -m
            s = lsb.bit_length() - 1
            img_a |= 1 << ta[s]
            m ^= lsb
        trans_a[mask] = img_a
        
        img_b = 0
        m = mask
        while m:
            lsb = m & -m
            s = lsb.bit_length() - 1
            img_b |= 1 << tb[s]
            m ^= lsb
        trans_b[mask] = img_b

    full_mask = (1 << N) - 1
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append(full_mask)
    depth = 0
    
    while queue:
        size_q = len(queue)
        for _ in range(size_q):
            subset = queue.popleft()
            if subset & (subset - 1) == 0: # Power of 2 check (singleton)
                return depth, True
            # Apply 'a'
            next_a = trans_a[subset]
            if next_a not in visited:
                visited.add(next_a)
                queue.append(next_a)
            # Apply 'b'
            next_b = trans_b[subset]
            if next_b not in visited:
                visited.add(next_b)
                queue.append(next_b)
        depth += 1
        # Safety check: if depth > N*(N-1), it's not synchronizing?
        # Max threshold is N^2 - 3N + 3. For N=12, 123.
        if depth > 150:
            break
            
    return None, False

def _generate_pawlik():
    """Generate the standard Pawlik (Černý) automaton for N states."""
    # sigma0: i -> i+1 mod N
    # sigma1: 0 -> 0, i -> i-1 for i>0
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    return [tuple(sigma0), tuple(sigma1)]

def _local_search(ta, tb, rng, max_steps=500, deadline=None):
    """
    Perform local search to improve the threshold.
    Mutate one transition at a time and keep improvements.
    """
    ta = list(ta)
    tb = list(tb)
    
    current_thresh, _ = _bfs_threshold(tuple(ta), tuple(tb))
    if current_thresh is None:
        return None, None
    
    for _ in range(max_steps):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random position to mutate
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        
        if letter == 0:
            old_dest = ta[state]
            new_dest = rng.randint(0, N - 1)
            if new_dest == old_dest: continue
            ta[state] = new_dest
            t, s = _bfs_threshold(tuple(ta), tuple(tb))
            if s and (t is not None and t > current_thresh):
                current_thresh = t
            else:
                ta[state] = old_dest
        else:
            old_dest = tb[state]
            new_dest = rng.randint(0, N - 1)
            if new_dest == old_dest: continue
            tb[state] = new_dest
            t, s = _bfs_threshold(tuple(ta), tuple(tb))
            if s and (t is not None and t > current_thresh):
                current_thresh = t
            else:
                tb[state] = old_dest
                
    return tuple(ta), tuple(tb)

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Start with the Pawlik automaton as baseline
    best_ta, best_tb = _generate_pawlik()
    best_thresh, _ = _bfs_threshold(best_ta, best_tb)
    
    if best_thresh is None:
        # Fallback
        best_ta = tuple([(i + 1) % N for i in range(N)])
        best_tb = tuple([0] * N)
        best_thresh = 1

    rng = random.Random(42)
    
    # 1. Local search from Pawlik
    if time.time() < deadline - 10:
        improved_ta, improved_tb = _local_search(best_ta, best_tb, rng, max_steps=1000, deadline=deadline - 5)
        if improved_ta:
            imp_thresh, _ = _bfs_threshold(improved_ta, improved_tb)
            if imp_thresh is not None and imp_thresh > best_thresh:
                best_ta, best_tb = improved_ta, improved_tb
                best_thresh = imp_thresh

    # 2. Try structured variants
    # Variant A: Perturb Pawlik
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5: break
            ta, tb = _generate_pawlik()
            ta = list(ta)
            tb = list(tb)
            # Random perturbations
            for _ in range(rng.randint(1, 5)):
                l = rng.randint(0, 1)
                s = rng.randint(0, N-1)
                if l == 0: ta[s] = rng.randint(0, N-1)
                else: tb[s] = rng.randint(0, N-1)
            
            thresh, sync = _bfs_threshold(tuple(ta), tuple(tb))
            if sync and thresh > best_thresh:
                best_thresh = thresh
                best_ta, best_tb = tuple(ta), tuple(tb)
                # Local search from this new best
                if time.time() < deadline - 10:
                    improved_ta, improved_tb = _local_search(best_ta, best_tb, rng, max_steps=200, deadline=deadline - 5)
                    if improved_ta:
                        imp_thresh, _ = _bfs_threshold(improved_ta, improved_tb)
                        if imp_thresh is not None and imp_thresh > best_thresh:
                            best_ta, best_tb = improved_ta, improved_tb
                            best_thresh = imp_thresh

    # 3. Random search with local search
    if time.time() < deadline - 15:
        for _ in range(100):
            if time.time() > deadline - 5: break
            ta = tuple(rng.randint(0, N-1) for _ in range(N))
            tb = tuple(rng.randint(0, N-1) for _ in range(N))
            
            thresh, sync = _bfs_threshold(ta, tb)
            if sync:
                if thresh > best_thresh:
                    best_thresh = thresh
                    best_ta, best_tb = ta, tb
                    
                # Try local search even if not best, might find a local peak?
                # Usually only local search from good candidates.
                if thresh > best_thresh - 5:
                    if time.time() < deadline - 10:
                        improved_ta, improved_tb = _local_search(ta, tb, rng, max_steps=100, deadline=deadline - 5)
                        if improved_ta:
                            imp_thresh, _ = _bfs_threshold(improved_ta, improved_tb)
                            if imp_thresh is not None and imp_thresh > best_thresh:
                                best_ta, best_tb = improved_ta, improved_tb
                                best_thresh = imp_thresh

    return [list(best_ta), list(best_tb)]

# EVOLVE-BLOCK-END
