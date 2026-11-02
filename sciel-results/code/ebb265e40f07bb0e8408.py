"""Rewritten program for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

def _precompute_ball_indices(q=3, n=8):
    """Precompute for each word index, the list of indices it covers (Hamming dist <= 1)."""
    total = q ** n
    balls = np.zeros((total, 1 + n * (q - 1)), dtype=np.int32)
    for w in range(total):
        # Convert to word
        word = [0] * n
        tmp = w
        for i in range(n - 1, -1, -1):
            word[i] = tmp % q
            tmp //= q
        # Self
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            for v in range(q):
                if v != word[pos]:
                    # Compute new index
                    digit_val = word[pos]
                    new_word = word[:]
                    new_word[pos] = v
                    new_idx = 0
                    for d in new_word:
                        new_idx = new_idx * q + d
                    balls[w, idx] = new_idx
                    idx += 1
    return balls

def _construct_ternary_8_1():
    """Construct ternary covering code K_3(8,1) using vectorized local search."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    q, n, radius = 3, 8, 1
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17
    
    # Precompute ball membership: ball_members[i] = list of indices covered by codeword i
    # Actually, we need: for each position in the space, which codewords cover it
    # Better: precompute for each word index, the 17 indices it covers
    print("Precomputing balls...", flush=True)
    balls = _precompute_ball_indices(q, n)
    print("Balls computed.", flush=True)
    
    # Initial solution: direct sum of tetracode and 3^4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            # Convert to index
            idx = 0
            for d in cw:
                idx = idx * q + d
            codewords.append(idx)
    
    # coverage[i] = number of codewords covering space position i
    coverage = np.zeros(total, dtype=np.int32)
    
    # Initialize coverage
    for cw_idx in codewords:
        for j in range(ball_size):
            coverage[balls[cw_idx, j]] += 1
    
    # Verify initial solution covers everything
    if np.any(coverage < 1):
        print("WARNING: initial solution has uncovered words!", flush=True)
    
    print(f"Initial: {len(codewords)} codewords", flush=True)
    
    best_count = len(codewords)
    best_codewords = codewords[:]
    
    # Local search: 1-exchange (remove) and 2-exchange (remove + add)
    # Use vectorized operations where possible
    
    def try_remove(cw_idx):
        """Try removing codeword at index cw_idx. Returns True if successful."""
        # Check if any position is only covered by this codeword
        pos = balls[codewords[cw_idx], :]
        if np.any(coverage[pos] == 1):
            return False
        # Remove it
        coverage[pos] -= 1
        codewords.pop(cw_idx)
        return True
    
    def try_add(space_idx):
        """Add a codeword at space_idx. Returns True if it was added."""
        pos = balls[space_idx, :]
        coverage[pos] += 1
        codewords.append(space_idx)
        return True
    
    def try_remove_add(remove_idx, add_idx):
        """Try removing codeword at remove_idx and adding space_idx as new codeword."""
        if remove_idx >= len(codewords):
            return False
        pos_rem = balls[codewords[remove_idx], :]
        pos_add = balls[add_idx, :]
        
        # Check: after removal, are all positions still covered by at least one other codeword?
        # A position is critically covered if coverage[pos] == 1 AND it's in pos_rem
        # After removal, coverage[pos] would be 0 for those positions
        # But if add covers them, that's fine
        # So: for each pos in pos_rem where coverage==1, it must be in pos_add
        critical = (coverage[pos_rem] == 1)
        if np.any(critical):
            # Check if all critical positions are covered by the new codeword
            critical_positions = pos_rem[critical]
            # Check if each critical position is in pos_add
            add_set = set(pos_add.tolist())
            for cp in critical_positions:
                if cp not in add_set:
                    return False
        
        # Perform the exchange
        coverage[pos_rem] -= 1
        codewords.pop(remove_idx)
        coverage[pos_add] += 1
        codewords.append(add_idx)
        return True
    
    # Main local search loop
    improved = True
    no_improve_rounds = 0
    max_no_improve = 3
    
    while improved and (time.time() - start_time) < deadline:
        improved = False
        no_improve_rounds = 0
        
        # Phase 1: Try to remove codewords (1-exchange: remove only)
        # Go in reverse to avoid index issues
        for i in range(len(codewords) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            if try_remove(i):
                improved = True
                if len(codewords) < best_count:
                    best_count = len(codewords)
                    best_codewords = codewords[:]
        
        # Phase 2: Try 2-exchanges (remove one, add one)
        if (time.time() - start_time) < deadline:
            # Sample candidate positions to add
            # Focus on positions that are covered only once (critical positions)
            # Or just sample randomly
            n_samples = 200
            for _ in range(n_samples):
                if (time.time() - start_time) > deadline:
                    break
                
                # Pick a random codeword to potentially remove
                rem_idx = random.randint(0, len(codewords) - 1)
                
                # Pick a random position to add
                add_pos = random.randint(0, total - 1)
                
                if try_remove_add(rem_idx, add_pos):
                    improved = True
                    if len(codewords) < best_count:
                        best_count = len(codewords)
                        best_codewords = codewords[:]
                
                # Also try adding without removing (if we have room, but we're minimizing)
                # Actually for minimization, 2-exchange is the key
        
        # Phase 3: Targeted 2-exchanges focused on critical positions
        if (time.time() - start_time) < deadline:
            # Find positions with coverage == 1
            critical_positions = np.where(coverage == 1)[0]
            if len(critical_positions) > 0:
                # For each critical position, find which codeword covers it
                # Then try to swap that codeword with one that also covers this position
                n_targeted = min(50, len(critical_positions))
                crit_sample = random.sample(critical_positions.tolist(), n_targeted)
                
                for cp in crit_sample:
                    if (time.time() - start_time) > deadline:
                        break
                    
                    # Find which codeword covers cp
                    # We need to find i such that cp is in balls[codewords[i], :]
                    # This is expensive, so let's build a reverse map periodically
                    # For now, just try random removals
                    for _ in range(10):
                        rem_idx = random.randint(0, len(codewords) - 1)
                        # Check if this codeword covers cp
                        if cp in balls[codewords[rem_idx], :].tolist():
                            # Try to find a replacement that also covers cp
                            # Sample some positions
                            for _ in range(5):
                                add_pos = random.randint(0, total - 1)
                                if try_remove_add(rem_idx, add_pos):
                                    improved = True
                                    if len(codewords) < best_count:
                                        best_count = len(codewords)
                                        best_codewords = codewords[:]
                                    break
    
    # Convert best codewords back to list format
    result = []
    for idx in best_codewords:
        word = [0] * n
        tmp = idx
        for i in range(n - 1, -1, -1):
            word[i] = tmp % q
            tmp //= q
        result.append(word)
    
    return result

def construct(q, n, radius):
    """Construct a covering code for the given parameters."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    else:
        # Fallback: greedy construction
        codewords = []
        covered = np.zeros(q ** n, dtype=bool)
        total = q ** n
        ball_size = 1 + n * (q - 1)
        
        # Precompute balls
        balls = _precompute_ball_indices(q, n)
        
        remaining = set(range(total))
        while remaining and (time.time() - start_time) < deadline:
            # Find word covering most uncovered
            best_w = None
            best_count = -1
            sample = random.sample(list(remaining), min(200, len(remaining)))
            for w in sample:
                pos = balls[w, :]
                count = sum(1 for p in pos if p in remaining)
                if count > best_count:
                    best_count = count
                    best_w = w
            if best_w is None:
                break
            codewords.append(best_w)
            for p in balls[best_w, :]:
                remaining.discard(int(p))
        
        result = []
        for idx in codewords:
            word = [0] * n
            tmp = idx
            for i in range(n - 1, -1, -1):
                word[i] = tmp % q
                tmp //= q
            result.append(word)
        return result
# EVOLVE-BLOCK-END
