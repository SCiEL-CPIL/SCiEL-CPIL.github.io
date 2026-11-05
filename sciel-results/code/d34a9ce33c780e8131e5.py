import time
import random
import sys
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = 150  # Safety margin for 120s limit
    
    if n <= 5:
        # Standard construction is optimal for n <= 5
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 6, use the TSP-based reordering of the (n-1)! groups
    # First, generate the standard S_{n-1} to get the list of (n-1)! permutations
    # and the standard order.
    
    # Generate S_{n-1}
    word_prev = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word_prev) - m + 2):
            window = tuple(word_prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_prev = new
        
    # Extract the (n-1)! permutations from S_{n-1} in the order they appear
    # These are the "groups"
    perms, seen = [], set()
    for i in range(len(word_prev) - (n-1) + 1):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # perms is a list of tuples, length (n-1)!
    # For n=7, len(perms) = 720
    # We want to find a permutation of perms that maximizes the sum of overlaps.
    
    if time.time() - start_time > deadline - 10:
        # Fallback to standard construction
        return _standard_construct(n)

    # Precompute overlap matrix
    # overlap[i][j] = longest k such that suffix of perms[i] of len k == prefix of perms[j] of len k
    N = len(perms)
    # For n=7, N=720. Matrix is 720x720 = 518,400 entries.
    # Storing as a list of lists of ints is fine.
    overlap = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = n - 1
            k = 0
            # Check k from max_k down to 0
            # Since max_k is small (6), this is fast
            for k in range(max_k, -1, -1):
                if pi[n-1-k:] == pj[:k]:
                    break
            overlap[i][j] = k

    # Initial order: standard order (0, 1, ..., N-1)
    order = list(range(N))
    
    # Calculate initial score (sum of overlaps)
    def calc_score(ord):
        total = 0
        for i in range(len(ord) - 1):
            total += overlap[ord[i]][ord[i+1]]
        return total

    current_score = calc_score(order)
    
    # 2-Opt Local Search
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            if time.time() - start_time > deadline - 5:
                break
            for j in range(i + 1, N):
                # Reverse segment i..j
                # Old edges: (i-1, i) and (j, j+1)
                # New edges: (i-1, j) and (i, j+1)
                
                # Get neighbors
                prev_i = order[i-1] if i > 0 else None
                next_j = order[j+1] if j < N - 1 else None
                
                old_cost = 0
                if prev_i is not None:
                    old_cost += overlap[prev_i][order[i]]
                if next_j is not None:
                    old_cost += overlap[order[j]][next_j]
                
                new_cost = 0
                if prev_i is not None:
                    new_cost += overlap[prev_i][order[j]]
                if next_j is not None:
                    new_cost += overlap[order[i]][next_j]
                
                # Internal edges of the reversed segment change direction
                # Overlap is not symmetric.
                # The edges inside the segment (i, i+1), ..., (j-1, j) become (j, j-1), ..., (i+1, i).
                # We need to calculate the change in internal cost.
                # This makes 2-opt O(N) per move if we precompute prefix sums?
                # Or we can just calculate the full segment cost change.
                # Since N=720, calculating the segment cost is O(N) worst case, but average is small.
                # However, doing this for all i,j is O(N^3) which is too slow (720^3 ~ 374M).
                
                # Optimization: Only try small segments or use a different heuristic?
                # Or precompute the cost of any segment?
                # Let's try a simpler heuristic: Or-opt (move one element).
                # Or just random swaps?
                
                # Let's switch to a simpler local search: Swap adjacent elements?
                # No, that's too slow to converge.
                
                # Let's try Or-opt: Move one element to a different position.
                # Or just do a few iterations of 2-opt with precomputed internal costs?
                
                # Given time constraints, let's do a limited 2-opt where we only check
                # if the boundary improvement is positive, and assume internal cost change is small?
                # No, that's unsafe.
                
                # Alternative: Use the fact that the graph is dense and overlaps are high.
                # Let's just do a simple "greedy" pass:
                # Iterate through the order, and for each position, try to swap with the next
                # if it improves the local cost (prev, curr, next).
                
                # Actually, let's just do a simple simulated annealing or random restarts
                # on the full order, but that's risky.
                
                # Let's stick to a simple 2-opt but optimize the internal cost calculation.
                # Precompute prefix sums of overlap costs?
                # Cost of path 0..k is sum of overlaps.
                # This doesn't help with reversal directly because overlaps are directional.
                
                # Let's try a different move: Swap two elements i and j.
                # This changes 4 edges.
                # O(1) per swap.
                # O(N^2) per pass.
                # 720^2 = 518k ops. Very fast.
                
                # Let's implement Swap-based local search.
                
                # Check swap i, j
                # Old: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # New: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
                # (Assuming i and j are not adjacent)
                
                # If i and j are adjacent, it's a swap of neighbors.
                
                # Let's just do this.
                pass # Skip 2-opt for now, implement swap search below

        # Implement Swap-based Local Search
        # We break out of the while loop if no improvement in a full pass
        # But the inner loop above was 2-opt which we skipped.
        # Let's restart the loop structure for Swap search.
        break

    # Re-implement local search with Swaps
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            if time.time() - start_time > deadline - 5:
                break
            for j in range(i + 1, N):
                if i + 1 == j:
                    # Adjacent swap
                    # Old: (i-1, i), (i, j), (j, j+1)
                    # New: (i-1, j), (j, i), (i, j+1)
                    prev_i = order[i-1] if i > 0 else None
                    next_j = order[j+1] if j < N - 1 else None
                    
                    old_cost = 0
                    if prev_i is not None: old_cost += overlap[prev_i][order[i]]
                    old_cost += overlap[order[i]][order[j]]
                    if next_j is not None: old_cost += overlap[order[j]][next_j]
                    
                    new_cost = 0
                    if prev_i is not None: new_cost += overlap[prev_i][order[j]]
                    new_cost += overlap[order[j]][order[i]]
                    if next_j is not None: new_cost += overlap[order[i]][next_j]
                    
                    if new_cost > old_cost:
                        order[i], order[j] = order[j], order[i]
                        current_score += (new_cost - old_cost)
                        improved = True
                else:
                    # Non-adjacent swap
                    # Edges affected:
                    # (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                    # Becomes:
                    # (i-1, j), (j, i+1), (j-1, i), (i, j+1)
                    
                    prev_i = order[i-1] if i > 0 else None
                    next_i = order[i+1]
                    prev_j = order[j-1]
                    next_j = order[j+1] if j < N - 1 else None
                    
                    old_cost = 0
                    if prev_i is not None: old_cost += overlap[prev_i][order[i]]
                    old_cost += overlap[order[i]][next_i]
                    old_cost += overlap[prev_j][order[j]]
                    if next_j is not None: old_cost += overlap[order[j]][next_j]
                    
                    new_cost = 0
                    if prev_i is not None: new_cost += overlap[prev_i][order[j]]
                    new_cost += overlap[order[j]][next_i]
                    new_cost += overlap[prev_j][order[i]]
                    if next_j is not None: new_cost += overlap[order[i]][next_j]
                    
                    if new_cost > old_cost:
                        order[i], order[j] = order[j], order[i]
                        current_score += (new_cost - old_cost)
                        improved = True
                        
            if time.time() - start_time > deadline - 5:
                break

    # Reconstruct the word using the new order
    # The standard construction logic:
    # word = S_{n-1} (but we only use the perms list, not the full word_prev for construction,
    # we rebuild S_n from scratch using the new order of perms)
    
    # Wait, the standard construction builds S_n from S_{n-1}.
    # The "perms" extracted are the windows of S_{n-1}.
    # The construction of S_n involves:
    # 1. Start with the first block corresponding to perms[order[0]]
    # 2. Append subsequent blocks with overlap.
    
    # Block for permutation p (length n-1) is: p + [n] + p
    # We need to concatenate these blocks in the order given by 'order'.
    
    blocks = []
    for idx in order:
        p = perms[idx]
        blocks.append(list(p) + [n] + list(p))
        
    # Now concatenate with maximum overlap
    word = []
    for b in blocks:
        if not word:
            word = b
            continue
        # Find max overlap between end of word and start of b
        max_overlap = min(len(word), len(b))
        found_overlap = 0
        for o in range(max_overlap, -1, -1):
            if word[-o:] == b[:o]:
                found_overlap = o
                break
        word += b[found_overlap:]
        
    # Verify validity?
    # The problem states "Valid when every permutation... occurs".
    # This construction guarantees it if the order is a permutation of all (n-1)! perms.
    # And the blocks cover all n! perms?
    # Yes, the standard construction property holds.
    
    return "".join(map(str, word))

def _standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
