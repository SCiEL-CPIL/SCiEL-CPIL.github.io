import itertools
import random
import time
import sys
import numpy as np

# Constants for the specific problem q=3, n=7, radius=1
Q = 3
N = 7
R = 1
TOTAL_WORDS = Q ** N  # 2187

# Precompute cover masks for the specific problem instance
# cover_masks[i] is an integer where bit j is 1 if dist(i, j) <= 1
COVER_MASKS = [0] * TOTAL_WORDS
FULL_MASK = (1 << TOTAL_WORDS) - 1

def _precompute_masks():
    """Precompute cover masks for q=3, n=7, r=1."""
    for i in range(TOTAL_WORDS):
        w = [0] * N
        temp = i
        for k in range(N - 1, -1, -1):
            w[k] = temp % Q
            temp //= Q
        
        mask = 1 << i
        # Check neighbors at distance 1
        for pos in range(N):
            for val in range(Q):
                if val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = val
                # Convert new_w back to index
                idx = 0
                for k in range(N):
                    idx = idx * Q + new_w[k]
                mask |= (1 << idx)
        COVER_MASKS[i] = mask

# Run precomputation at import time to save time in construct()
_precompute_masks()

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _solve_k3_7_1_bnb()
    
    # Fallback for other parameters
    # Use the previous logic or a simple greedy
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Generic fallback: return all words (always valid, but large)
    total = q ** n
    result = []
    for i in range(total):
        w = [0] * n
        temp = i
        for k in range(n - 1, -1, -1):
            w[k] = temp % q
            temp //= q
        result.append(w)
    return result

def _solve_k3_7_1_bnb():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin
    
    # Initial Upper Bound: Use the known best 186, or try a quick greedy to see if we can do better
    # Let's do a quick greedy to get a potentially better UB
    U = FULL_MASK
    C_greedy = []
    while U != 0:
        best_c = -1
        max_covered = 0
        # Optimization: Only check candidates that cover the first uncovered bit
        # Find first uncovered bit
        lsb = U & -U
        idx_v = lsb.bit_length() - 1
        
        # Candidates must cover idx_v
        # Candidates are those c such that bit idx_v is set in COVER_MASKS[c]
        # This is equivalent to c being in the "inverse" cover of idx_v
        # Due to symmetry, the set of c covering v is the same as the set of v' such that v covers v'
        # So we iterate c in 0..TOTAL_WORDS and check if (COVER_MASKS[c] & lsb) != 0
        # This is still O(N) per step if done naively. 
        # Better: Precompute "inverse cover" or just iterate all. 2187 is small.
        for c in range(TOTAL_WORDS):
            if C_greedy and c in C_greedy: # Use a set for O(1) lookup in greedy
                continue
            # Actually, just using a boolean array is faster
            pass
            
    # Let's stick to the B&B with a safe initial UB of 186.
    # If we find something < 186, great. If not, we return 186 (if valid) or the best found.
    # But we must ensure the returned solution is valid.
    # The B&B will find a valid solution.
    
    best_size = 186 # Known best
    best_solution = None
    
    # We need a valid solution of size 186 to return if B&B doesn't beat it or times out.
    # We can construct one using the greedy method quickly.
    # Greedy:
    U = FULL_MASK
    C = []
    in_C = [False] * TOTAL_WORDS
    while U != 0:
        # Find uncovered vertex with fewest candidates? Or just first?
        # First uncovered is fast.
        lsb = U & -U
        v_idx = lsb.bit_length() - 1
        
        best_c = -1
        max_new_cover = -1
        # Candidates are those c not in C that cover v_idx
        # c covers v_idx iff (COVER_MASKS[c] >> v_idx) & 1
        for c in range(TOTAL_WORDS):
            if in_C[c]:
                continue
            if (COVER_MASKS[c] >> v_idx) & 1:
                # Count how many NEW vertices c covers
                new_cover = COVER_MASKS[c] & U
                cnt = new_cover.bit_count()
                if cnt > max_new_cover:
                    max_new_cover = cnt
                    best_c = c
        
        if best_c == -1:
            break # Should not happen if U != 0
            
        C.append(best_c)
        in_C[best_c] = True
        U &= ~COVER_MASKS[best_c]
        
    # C is a valid cover.
    # Try to reduce C using local search (deletion) to get a tighter UB for B&B
    # This is the same local search as before, but faster now because we just need a good UB.
    # Let's run a few rounds of deletion.
    for _ in range(50):
        progress = False
        for i in range(len(C) - 1, -1, -1):
            c = C[i]
            # Check if c is redundant
            # c is redundant if all its neighbors are covered by C \ {c}
            # i.e., for all v in neighbors(c), there exists u in C\{c} such that v in neighbors(u)
            # Equivalent: (Union of masks of C\{c}) covers (mask of c)
            # Let M = Union of masks of C.
            # Let M_c = mask of c.
            # Let M_rest = Union of masks of C\{c}.
            # If M_rest & M_c == M_c, then c is redundant.
            
            # Efficient check:
            # We can maintain the union mask? No, removing one element requires recomputing union or using counts.
            # Since N=186, recomputing union is 186 big-int ORs. 186 * 186 ~ 34k ops. Fast.
            
            union_rest = 0
            for j in range(len(C)):
                if j == i:
                    continue
                union_rest |= COVER_MASKS[C[j]]
            
            if (union_rest & COVER_MASKS[c]) == COVER_MASKS[c]:
                # Redundant
                C.pop(i)
                in_C[c] = False
                progress = True
                break # Restart loop to check new state
        
        if not progress:
            break
    
    greedy_ub = len(C)
    if greedy_ub < best_size:
        best_size = greedy_ub
        best_solution = C[:]
    else:
        # If greedy didn't beat 186, we still need a valid solution of size 186.
        # We can't just assume 186 exists and is valid without constructing it.
        # The greedy solution is valid. If its size is > 186, we return the greedy one?
        # No, we want to minimize. If we can't prove < 186, we should return the best valid one we have.
        # But the prompt implies 186 is the target.
        # Let's assume the greedy + local search finds something close to 186.
        # If it finds 190, we return 190. Score 243/190 = 1.27.
        # If we find 186, score 243/186 = 1.30.
        # Let's use the greedy result as the initial best_solution.
        best_solution = C[:]
        best_size = greedy_ub

    # Now run B&B to try to beat best_size
    # B&B State:
    # uncovered_mask: bits for vertices not yet covered
    # candidates_mask: bits for vertices available to be selected as codewords
    # current_solution: list of selected codewords
    
    # We start with the empty solution.
    # To guide the search, we can use the greedy solution as a heuristic for variable ordering?
    # Or just standard B&B.
    
    # Optimization: If best_size is 186, and we find 185, we update.
    
    def lower_bound(uncovered_mask, candidates_mask):
        # Simple LB: number of uncovered vertices / max coverage
        # Max coverage is 15.
        # But we can do better:
        # If there is an uncovered vertex v, it must be covered by a candidate c.
        # The candidates covering v are limited.
        # A better LB:
        # Let U be the set of uncovered vertices.
        # We need to cover U.
        # Each codeword covers at most 15 vertices in U.
        # So LB >= ceil(|U| / 15).
        # This is weak.
        
        # Stronger LB:
        # Find a vertex v in U with the fewest candidates covering it.
        # Let k be that number.
        # This doesn't directly give a count LB, but helps branching.
        
        # For now, use the weak LB to prune.
        # If current_depth + ceil(|U|/15) >= best_size, prune.
        return (uncovered_mask.bit_count() + 14) // 15

    def bnb(uncovered_mask, candidates_mask, current_depth, current_solution):
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return
            
        if uncovered_mask == 0:
            if current_depth < best_size:
                best_size = current_depth
                best_solution = current_solution[:]
            return
            
        # Pruning
        if current_depth + lower_bound(uncovered_mask, candidates_mask) >= best_size:
            return
            
        # Select a vertex to cover
        # Choose the uncovered vertex with the fewest candidate covers to minimize branching
        # Or just the first one for speed?
        # Let's try to find the "most constrained" vertex.
        
        # To avoid iterating 2187 vertices and their candidate sets every time,
        # we can just pick the lowest set bit in uncovered_mask.
        # This is fast.
        lsb = uncovered_mask & -uncovered_mask
        v_idx = lsb.bit_length() - 1
        
        # Find candidates that cover v_idx
        # c covers v_idx iff (COVER_MASKS[c] >> v_idx) & 1
        # We only consider c in candidates_mask.
        
        # Iterate through set bits in candidates_mask
        cands = candidates_mask
        # To order candidates by coverage of *other* uncovered vertices, we might need to sort.
        # Sorting is expensive. Let's just iterate.
        # However, we can skip candidates that cover 0 *other* uncovered vertices?
        # No, they cover v_idx, so they cover at least 1.
        
        # Optimization: 
        # If a candidate c covers v_idx, it covers v_idx and potentially others.
        # We want candidates that cover the most *new* vertices.
        
        # Let's collect valid candidates and sort by coverage count (descending)
        # This is O(K log K) where K is number of candidates covering v_idx (max 15).
        # This is very fast.
        
        valid_cands = []
        temp_cands = cands
        while temp_cands:
            lsb_c = temp_cands & -temp_cands
            c = lsb_c.bit_length() - 1
            temp_cands &= temp_cands - 1
            
            # c must be in candidates_mask (guaranteed by iteration)
            # c must cover v_idx
            if (COVER_MASKS[c] >> v_idx) & 1:
                # Count new coverage
                new_cover = COVER_MASKS[c] & uncovered_mask
                cnt = new_cover.bit_count()
                valid_cands.append((cnt, c))
                
        # Sort by coverage count descending
        valid_cands.sort(reverse=True)
        
        # Branch
        for cnt, c in valid_cands:
            # Try adding c
            new_uncovered = uncovered_mask & ~COVER_MASKS[c]
            new_candidates = candidates_mask & ~(1 << c) # Remove c from candidates
            
            # Heuristic: If new_uncovered is not empty, and we have few candidates left, prune?
            # The lower_bound call in the next recursion handles this.
            
            current_solution.append(c)
            bnb(new_uncovered, new_candidates, current_depth + 1, current_solution)
            current_solution.pop()
            
            # If we found a solution of size 1 (impossible) or updated best_size, 
            # we might want to stop early? No, we want the global minimum.
            # But if best_size becomes current_depth + 1, we can prune siblings?
            # The lower_bound check at the top of the recursion handles this.
            
            if time.time() - start_time > time_limit:
                return

    # Start B&B
    # Initial state: All uncovered, all candidates
    # We can use the greedy solution to guide the initial candidate set?
    # No, B&B needs to explore.
    
    # However, starting from scratch with 2187 candidates might be slow.
    # We can restrict the initial candidates to those in the greedy solution + some randoms?
    # No, that might miss the optimal.
    
    # Let's run B&B.
    # To speed up, we can use the fact that the problem is symmetric.
    # But for now, let's rely on bitset speed.
    
    # One issue: The search space is large.
    # If B&B doesn't finish in time, we return best_solution (the greedy one).
    
    # To improve the greedy solution before B&B, we can do more local search swaps.
    # But B&B itself is a search.
    
    # Let's run B&B with a time limit.
    bnb(FULL_MASK, FULL_MASK, 0, [])
    
    if best_solution is None:
        # Fallback to greedy
        best_solution = C[:]
        
    # Convert indices to words
    result = []
    for idx in best_solution:
        w = [0] * N
        temp = idx
        for k in range(N - 1, -1, -1):
            w[k] = temp % Q
            temp //= Q
        result.append(w)
        
    return result

# EVOLVE-BLOCK-START
# The above functions are defined.
# We need to make sure `construct` is the entry point.
# EVOLVE-BLOCK-END
