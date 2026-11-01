"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill (allow for overhead)
    
    if n <= 1:
        return "1"
    
    # 1. Generate standard superpermutation for n-1
    # This provides a valid sequence of permutations of 1..n-1
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # Extract the sequence of permutations of length n-1 from the generated word
    perms_seq = []
    seen = set()
    perm_len = n - 1
    for i in range(len(word) - perm_len + 1):
        window = tuple(word[i:i + perm_len])
        if len(set(window)) == perm_len and window not in seen:
            seen.add(window)
            perms_seq.append(window)
            
    # 2. Precompute overlap matrix
    # The construction of S_n is formed by concatenating blocks B_p = p + [n] + p.
    # The length of the final word is sum(len(B_p)) - sum(overlap(B_{p_i}, B_{p_{i+1}})).
    # Since len(B_p) is constant (2n-1), minimizing length is equivalent to maximizing
    # the sum of overlaps between consecutive blocks in the ordering.
    # overlap(B_i, B_j) is the largest k such that B_i[-k:] == B_j[:k].
    # Note: B_i ends with p_i. B_j starts with p_j.
    # So overlap depends only on the suffix of p_i and prefix of p_j.
    # Specifically, overlap(B_i, B_j) = max k such that p_i[-k:] == p_j[:k].
    # Max possible k is n-1.
    
    num_perms = len(perms_seq)
    # Map perm tuple to index for faster lookup if needed, but we work with list of indices
    # M[i][j] = overlap between perm i and perm j
    # p_i[-k:] == p_j[:k]
    
    M = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Precompute suffixes and prefixes?
    # Since n is small (7), k is small (<=6).
    # We can just loop k from n-1 down to 0.
    
    for i in range(num_perms):
        p_i = perms_seq[i]
        for j in range(num_perms):
            p_j = perms_seq[j]
            max_ov = 0
            # Check overlaps from max possible down to 0
            for k in range(n - 1, -1, -1):
                # p_i[-k:] == p_j[:k]
                # If k=0, empty string == empty string is True.
                if p_i[-k:] == p_j[:k]:
                    max_ov = k
                    break
            M[i][j] = max_ov
    
    # 3. Local Search on the ordering
    # We want to find a permutation of indices 0..num_perms-1 that maximizes sum M[order[k]][order[k+1]].
    # This is the Maximum Weight Hamiltonian Path problem.
    # We start with the standard order (indices 0..num_perms-1 in order).
    
    best_indices = list(range(num_perms))
    
    def calc_score(indices):
        total = 0
        for k in range(len(indices) - 1):
            total += M[indices[k]][indices[k + 1]]
        return total
    
    best_score = calc_score(best_indices)
    current_indices = best_indices[:]
    current_score = best_score
    
    # Local Search: 2-opt style moves (reverse segments) and swaps
    # Given the time limit, we should prioritize moves that are likely to improve the score.
    # However, random restarts or random perturbations might help escape local minima.
    # Let's try a simple hill-climbing with random restarts if stuck, or just extensive local search.
    
    # Since N=720 for n=7, the search space is huge.
    # A simple greedy construction might be a better starting point than the standard order?
    # Greedy: Start with any perm, then always pick the next perm that gives max overlap.
    # This is O(N^2) to build.
    
    # Let's try a few starting points:
    # 1. Standard order
    # 2. Greedy from each of the first 10 perms
    # 3. Random
    
    candidates = []
    
    # Standard order
    candidates.append(list(range(num_perms)))
    
    # Greedy constructions
    for start_idx in range(min(10, num_perms)):
        if time.time() > deadline - 20:
            break
        remaining = set(range(num_perms))
        remaining.remove(start_idx)
        order = [start_idx]
        curr = start_idx
        while remaining:
            best_next = -1
            best_ov = -1
            for nxt in remaining:
                ov = M[curr][nxt]
                if ov > best_ov:
                    best_ov = ov
                    best_next = nxt
            if best_next == -1:
                break
            order.append(best_next)
            remaining.remove(best_next)
            curr = best_next
        candidates.append(order)
        
    # Random candidates
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        rand_order = list(range(num_perms))
        random.shuffle(rand_order)
        candidates.append(rand_order)
    
    best_overall_indices = None
    best_overall_score = -1
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        
        # Local search on this candidate
        curr = cand[:]
        curr_score = calc_score(curr)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try swaps of adjacent elements? Or 2-opt (reverse segment)?
            # 2-opt is usually more effective for TSP-like problems.
            # Let's try 2-opt.
            # Find i, j such that reversing curr[i:j] improves score.
            # Change in score:
            # Old: ... a -> b ... c -> d ... (where a=curr[i-1], b=curr[i], c=curr[j], d=curr[j+1])
            # New: ... a -> d ... c -> b ...
            # Delta = M[a][d] + M[c][b] - M[a][b] - M[c][d]
            
            # To be efficient, we can precompute or just loop.
            # N=720. N^2 = 518k. Checking all pairs is expensive.
            # We can do a first-improvement or best-improvement search.
            # Given time constraints, let's try a limited 2-opt search.
            
            # Actually, simple swap of adjacent elements is O(N) per pass.
            # Let's try adjacent swaps first as it's cheaper.
            
            for i in range(num_perms - 1):
                # Swap i and i+1
                a = curr[i-1] if i > 0 else -1
                b = curr[i]
                c = curr[i+1]
                d = curr[i+2] if i+2 < num_perms else -1
                
                old_cost = 0
                if a != -1: old_cost += M[a][b]
                old_cost += M[b][c]
                if d != -1: old_cost += M[c][d]
                
                # New order: a, c, b, d
                new_cost = 0
                if a != -1: new_cost += M[a][c]
                new_cost += M[c][b]
                if d != -1: new_cost += M[b][d]
                
                if new_cost > old_cost:
                    curr[i], curr[i+1] = curr[i+1], curr[i]
                    curr_score += (new_cost - old_cost)
                    improved = True
                    if curr_score > best_overall_score:
                        best_overall_score = curr_score
                        best_overall_indices = curr[:]
                    # Restart scan? Or continue?
                    # Restarting scan (steepest ascent) is safer but slower.
                    # Let's just continue scanning from 0 to be safe and simple.
                    break
        
        if curr_score > best_overall_score:
            best_overall_score = curr_score
            best_overall_indices = curr[:]
            
    # 4. Construct the final word
    if best_overall_indices is None:
        best_overall_indices = list(range(num_perms))
        
    final_word = []
    for idx in best_overall_indices:
        p = perms_seq[idx]
        block = list(p) + [n] + list(p)
        
        # Overlap with current end of final_word
        # We need to find max overlap between final_word suffix and block prefix.
        # Since we optimized the overlaps between blocks, this should match M.
        # But to be safe, we calculate it.
        max_ov = 0
        # Max possible overlap is 2n-1? No, block length is 2n-1.
        # But suffix of previous block is p_prev (len n-1).
        # Prefix of current block is p_curr (len n-1).
        # So overlap is at most n-1.
        # Wait, the block is p + [n] + p.
        # Suffix of previous block is p_prev.
        # Prefix of current block is p_curr.
        # Overlap is max k such that p_prev[-k:] == p_curr[:k].
        # This is exactly M[prev_idx][curr_idx].
        
        # However, the first block has no previous.
        if not final_word:
            final_word = block
        else:
            # Calculate overlap
            # We know the previous perm index?
            # It's best_overall_indices[k-1]
            # Let's just compute it to be robust.
            # Since final_word is built by appending blocks, its suffix is the suffix of the last block.
            # Last block suffix is p_prev.
            # So we just check overlap between p_prev and p_curr.
            
            prev_idx = best_overall_indices[-1] # This is wrong if we are in the middle of construction
            # Actually, we are iterating. Let's keep track of prev_idx.
            
            # For the first iteration, prev_idx is undefined.
            # For subsequent, it's the previous index.
            
            # Let's restructure the loop to keep track.
            
            pass
            
    # Re-implement construction loop properly
    final_word = []
    prev_idx = -1
    for k, idx in enumerate(best_overall_indices):
        p = perms_seq[idx]
        block = list(p) + [n] + list(p)
        
        if k == 0:
            final_word = block
        else:
            # Overlap is M[prev_idx][idx]
            ov = M[prev_idx][idx]
            # Append block[ov:]
            final_word += block[ov:]
        prev_idx = idx
        
    # 5. Deletion pass
    # Try to remove redundant symbols from the end or middle.
    # The standard superpermutation has some redundancy.
    # We can try to remove symbols from the end if the last permutation is still covered.
    # Or more generally, try to remove any symbol that doesn't destroy a unique permutation window.
    
    current = list(final_word)
    
    def build_perm_map(w, n):
        perm_count = defaultdict(int)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                perm_count[window] += 1
        return perm_count
    
    target = 1
    for i in range(1, n + 1):
        target *= i
        
    perm_map = build_perm_map(current, n)
    if len(perm_map) < target:
        # Should not happen, but return anyway
        return "".join(map(str, current))
        
    # Fast deletion pass
    # We try to remove symbols from right to left.
    # If removing symbol at index i means some permutation window covering i has count 1, we can't remove it.
    # Windows covering i are [i-n+1, i].
    
    # To speed up, we can just try removing from the end first, as it's often safe.
    # Then try random positions or scan from end to start.
    
    # Let's scan from end to start.
    # This is O(L * n) per check. L ~ 6000, n=7. ~42k ops per removal attempt.
    # If we do 1000 attempts, 42M ops. Might be slow in Python.
    # Let's limit the number of attempts or use a smarter check.
    
    # Smarter check:
    # Maintain a list of counts for each perm.
    # When we remove a symbol, we decrement counts for affected windows.
    # If any count drops to 0, we revert.
    
    # However, implementing full rollback is complex.
    # Let's stick to a limited scan from the end.
    
    for i in range(len(current) - 1, -1, -1):
        if time.time() > deadline - 2:
            break
            
        # Check if removal is safe
        start_w = max(0, i - n + 1)
        end_w = min(i, len(current) - n)
        
        can_remove = True
        for w in range(start_w, end_w + 1):
            window = tuple(current[w:w + n])
            if len(set(window)) == n:
                if perm_map[window] == 1:
                    can_remove = False
                    break
        
        if can_remove:
            new_word = current[:i] + current[i + 1:]
            # Verify
            new_pm = build_perm_map(new_word, n)
            if len(new_pm) == target:
                current = new_word
                perm_map = new_pm
            else:
                # Should not happen if check was correct, but just in case
                pass
        else:
            # If we can't remove from the end, maybe we can skip some?
            # For now, just continue scanning.
            pass
            
    return "".join(map(str, current))
# EVOLVE-BLOCK-END
