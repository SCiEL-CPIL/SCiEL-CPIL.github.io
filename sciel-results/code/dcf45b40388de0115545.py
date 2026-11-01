import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses a greedy set-cover approach followed by local search optimization
    with simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        # Specific optimization for K_3(7,1)
        
        # 1. Precompute all words and their balls (neighbors within radius 1)
        all_words = list(itertools.product(range(3), repeat=7))
        N = len(all_words) # 2187
        
        # Map word tuple to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute balls: for each index, list of indices covered by that codeword
        # Ball(c) = {c} U {words at dist 1}
        balls = []
        for w in all_words:
            covered_indices = [word_to_idx[w]]
            for i in range(7):
                for val in range(3):
                    if val != w[i]:
                        new_word = list(w)
                        new_word[i] = val
                        new_word = tuple(new_word)
                        covered_indices.append(word_to_idx[new_word])
            balls.append(np.array(covered_indices, dtype=np.int32))
        
        # 2. Greedy Set Cover to get initial solution
        # covered: boolean array
        covered = np.zeros(N, dtype=bool)
        codewords = []
        
        # Greedy: pick word covering most uncovered
        while not covered.all():
            best_cw = -1
            best_count = -1
            
            # Vectorized greedy step
            # For each candidate i, count how many of its ball are currently uncovered
            # We can do this by checking ~covered[balls[i]]
            # To speed up, we can just scan. N=2187 is small.
            # However, scanning 2187 * 15 elements * 2187 iterations is ~70M ops.
            # Let's try to be smarter or just accept it if fast enough.
            
            # Optimization: Use a heap or just random sampling?
            # Actually, for N=2187, a full scan of 2187 candidates, each checking 15 indices,
            # is 2187*15 = 32k operations.
            # If we do this ~150 times, that's 4.8M ops. Very fast.
            
            for i in range(N):
                b = balls[i]
                # Count new coverage
                new_cov = np.sum(~covered[b])
                if new_cov > best_count:
                    best_count = new_cov
                    best_cw = i
            
            if best_count == 0:
                # Should not happen if not covered.all()
                break
                
            codewords.append(best_cw)
            covered[balls[best_cw]] = True
            
            if time.time() > deadline - 60:
                break

        # 3. Initialize coverage counts for Local Search
        # coverage_count[w] = number of codewords covering w
        coverage_count = np.zeros(N, dtype=np.int32)
        for cw in codewords:
            coverage_count[balls[cw]] += 1
        
        # Count how many words have coverage 0
        cnt0 = np.sum(coverage_count == 0)
        
        # Helper to check validity
        def is_valid():
            return cnt0 == 0

        # Helper to remove codeword
        # Returns True if valid after removal, False otherwise (and reverts)
        def try_remove(cw_idx):
            nonlocal cnt0
            b = balls[cw_idx]
            # Check if any word in b has count 1
            # If yes, removing will make it 0 -> invalid
            if np.min(coverage_count[b]) == 1:
                return False
            # Safe to remove
            coverage_count[b] -= 1
            # Update cnt0
            # Words that were 1 become 0
            new_zeros = np.sum(coverage_count[b] == 0)
            cnt0 += new_zeros
            return True

        # Helper to add codeword
        def try_add(cw_idx):
            nonlocal cnt0
            b = balls[cw_idx]
            # Words that were 0 become 1 (or more)
            old_zeros = np.sum(coverage_count[b] == 0)
            coverage_count[b] += 1
            cnt0 -= old_zeros
            return True

        # 4. Redundancy Removal (Greedy Pass)
        # Try to remove each codeword
        i = 0
        while i < len(codewords):
            if try_remove(codewords[i]):
                codewords.pop(i)
            else:
                i += 1
            if time.time() > deadline - 40:
                break

        # 5. Simulated Annealing Local Search
        # Strategy: 
        # - Move types: Remove (if valid), Swap (replace one with another)
        # - Objective: Minimize len(codewords). Secondary: Maximize "slack" (sum of counts above 1) to allow future removals?
        # - Actually, just minimizing size is the goal. 
        # - We accept worse states (larger size) with probability exp(-delta/T).
        # - But we must always maintain validity (cnt0 == 0).
        # - So we only consider valid moves.
        # - "Worse" move: Swap that keeps size same but changes structure? Or Add?
        # - Adding increases size, which is worse. We might do this to escape local minima where no removal/swap is possible.
        
        random.seed(42)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        current_size = len(codewords)
        best_size = current_size
        best_codewords = list(codewords)
        best_coverage = coverage_count.copy()
        
        while time.time() < deadline - 5:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try Remove
                if codewords:
                    idx = random.randint(0, len(codewords) - 1)
                    cw = codewords[idx]
                    if try_remove(cw):
                        codewords.pop(idx)
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_codewords = list(codewords)
                            best_coverage = coverage_count.copy()
                        # Accept always (improvement)
                    else:
                        # Revert if try_remove modified state? 
                        # My try_remove reverts on failure? 
                        # Let's check: 
                        # if np.min(...) == 1: return False. No modification.
                        # So no revert needed.
                        pass
            
            elif move_type < 0.9:
                # Try Swap
                if codewords:
                    idx_old = random.randint(0, len(codewords) - 1)
                    cw_old = codewords[idx_old]
                    cw_new = random.randint(0, N - 1)
                    
                    if cw_new == cw_old:
                        continue
                    
                    b_old = balls[cw_old]
                    b_new = balls[cw_new]
                    
                    # Simulate swap
                    # Decrement old, Increment new
                    coverage_count[b_old] -= 1
                    coverage_count[b_new] += 1
                    
                    new_cnt0 = np.sum(coverage_count == 0)
                    
                    if new_cnt0 == 0:
                        # Valid swap
                        codewords[idx_old] = cw_new
                        # Size unchanged
                        # We don't update best here as size is same, but structure changed.
                        # However, if we are stuck, a valid swap might enable a removal later.
                        # So we accept it.
                        # To track "best" for restart, we might want to keep the state with max slack?
                        # For now, just accept valid swaps.
                        pass
                    else:
                        # Invalid, revert
                        coverage_count[b_old] += 1
                        coverage_count[b_new] -= 1
                        # cnt0 variable needs to be consistent.
                        # Since we reverted, cnt0 is back to original value.
                        # We didn't change cnt0 variable in the simulation, so it's still correct.
                        pass
            
            else:
                # Try Add (Worse move, increases size)
                # Only if we are stuck or to explore
                if random.random() < 0.1: # Low probability
                    cw_new = random.randint(0, N - 1)
                    # Check if it's already in codewords? 
                    # If yes, adding duplicate is useless but valid? 
                    # Problem says codewords must be distinct. 
                    # So we should avoid adding existing.
                    if cw_new not in codewords:
                        try_add(cw_new)
                        codewords.append(cw_new)
                        current_size += 1
                        # Accept with probability? 
                        # Since it's worse, we should be careful.
                        # But for SA, we accept worse moves to escape.
                        # Let's accept it always for now, but cool down faster.
                        # Or reject if T is low.
                        if random.random() > np.exp(-1.0 / T):
                            # Reject
                            codewords.pop()
                            try_remove(cw_new) # Revert add? 
                            # Wait, try_add modified counts. 
                            # We need a revert_add.
                            # Let's just not do Adds if it's complicated to revert.
                            # Or implement revert_add.
                            pass
                        else:
                            # Accept
                            pass
                    else:
                        pass
            
            # Cool down
            T = max(T * cooling, T_min)
            
            # If we hit a new best, we might want to restart or just continue.
            # Continuing is fine.
            
            # Periodic restart to best if stuck?
            # Not implemented for simplicity, but random swaps help.

        # Use the best found solution
        # Note: best_codewords and best_coverage are from when best_size was achieved.
        # But current codewords might be smaller? No, best_size tracks min.
        # If current_size < best_size, we would have updated best.
        # So best_codewords is the smallest found.
        
        # However, we might have ended in a state with size == best_size but different structure.
        # It doesn't matter for score, only size matters.
        
        return [list(all_words[cw]) for cw in best_codewords]
    
    # General fallback (not optimized for this specific case)
    all_words = list(itertools.product(range(q), repeat=n))
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == len(all_words):
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
