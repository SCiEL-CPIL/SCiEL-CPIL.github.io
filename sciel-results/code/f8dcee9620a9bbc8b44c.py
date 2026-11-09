import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _get_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            sums.add(total)
    return sums

def _search_with_swap(max_val=256, deadline=None, initial_seq=None):
    """
    Search strategy:
    1. Start with a greedy sequence.
    2. Use a DFS-like approach but with a twist:
       - At each step, we don't just pick the smallest valid x.
       - We maintain a "beam" of the top K most promising states.
       - A state is (seq, sums, max_sum_so_far).
       - Promisingness is defined by:
         a) Length (primary)
         b) Low max_sum_so_far (secondary, to leave room)
         c) Low last element (tertiary, to allow more values)
    
    Actually, let's try a different structural approach:
    **Iterative Deepening with Conflict-Based Backtracking**
    
    We will attempt to build a sequence of length L.
    If we fail, we identify the conflict and backtrack.
    To handle the "local optimum" issue of greedy, we will use a 
    randomized restart strategy but with a specific heuristic:
    "Minimize the spread of sums".
    
    Let's implement a Beam Search with a wide beam that retains 
    diverse sum-sets, not just the single best path.
    """
    if deadline is None:
        deadline = time.time() + 50
        
    best_seq = []
    best_len = 0
    
    # State: (seq, sums_set, last_val)
    # We use a beam search to keep multiple paths alive.
    beam_width = 50
    
    # Initialize beam with empty sequence
    beam = [([], set(), 0)]
    
    # We will grow the beam step by step.
    # However, standard beam search on this problem is tricky because 
    # the "value" of a state depends on future potential.
    
    # Alternative: 
    # 1. Generate a strong initial guess using Greedy.
    # 2. Use a local search / hill climbing on the sequence itself?
    #    No, the sequence must be strictly increasing integers.
    
    # Let's stick to a modified DFS that uses a "Lookahead" heuristic.
    # Heuristic: For each candidate x, estimate the "potential length" 
    # remaining. Potential is roughly (max_val - x) / avg_step.
    # But more importantly, the sum space must not be "too crowded".
    
    # Let's try a different tactic: **Reverse Search**.
    # Start with a high value and work backwards? No, sums are forward.
    
    # Let's try **Simulated Annealing** on the sequence of indices?
    # No, the values are fixed 1..256. We are selecting a subset.
    
    # Let's go with **Beam Search with Diversity**.
    # We keep a list of states. At each depth, we expand each state 
    # with its top K valid children. We then prune the beam to the 
    # best beam_width states based on a score.
    
    def score_state(seq, sums, last):
        # We want long sequences.
        # We want small last value (more room).
        # We want small max sum (more room in sum space).
        # We want "sparse" sums? Hard to measure quickly.
        # Let's use: (len(seq), -last, -max(sums) if sums else 0)
        # But we are comparing states at the same depth in beam search.
        # So len is constant.
        return (-last, -max(sums) if sums else 0)

    current_beam = [([], set(), 0)]
    
    # We will iterate until we can't grow anymore or time runs out.
    # But beam search doesn't naturally "stop" at a local max length 
    # unless we track the best found.
    
    # Let's run a DFS with a specific pruning rule:
    # "If the number of available sum slots is less than the number of 
    # future elements we hope to add, prune."
    # Total sum space is roughly 1..sum(seq). 
    # Max sum is bounded by 256*256/2 ~ 32000.
    
    # Let's implement a **Greedy with Backtracking on Stalls**.
    # 1. Greedily build a sequence.
    # 2. If we get stuck (no valid next element), backtrack to the last 
    #    element and try the next smallest valid candidate for that position.
    # 3. If that fails, backtrack further.
    
    def greedy_with_backtrack(seq, sums, last, max_val=256, deadline=None):
        nonlocal best_seq, best_len
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        if last >= max_val:
            return
            
        # Find all valid candidates for the next element
        candidates = []
        for x in range(last + 1, max_val + 1):
            tails = [x]
            total = x
            valid = True
            for a in reversed(seq):
                total += a
                tails.append(total)
                if total in sums:
                    valid = False
                    break
            if x in sums:
                valid = False
            if valid:
                candidates.append((x, tails))
                
        if not candidates:
            return
            
        # Try candidates in order of increasing x (smallest first)
        # But if the smallest x leads to a dead end quickly, we want to 
        # try the next one.
        
        # To avoid exponential explosion, we limit the depth of backtracking 
        # or the number of candidates we try per level if time is tight.
        # However, for length < 20, the branching factor is small.
        
        for x, tails in candidates:
            if time.time() > deadline:
                return
                
            new_sums = set(sums)
            new_sums.update(tails)
            new_seq = seq + [x]
            
            # Recurse
            greedy_with_backtrack(new_seq, new_sums, x, max_val, deadline)
            
            # Optimization: If we found a solution of length L, and we are 
            # at depth L-1, and we've tried the smallest candidate, do we 
            # need to try larger ones?
            # Larger ones leave less room. So if the smallest candidate 
            # yielded a sequence of length L, a larger candidate is unlikely 
            # to yield > L. 
            # This is a strong heuristic for pruning.
            if len(seq) + 1 >= best_len:
                # We already found a sequence at least as long as the current depth + 1.
                # Trying a larger x will only make the sequence "heavier".
                # It is very unlikely to beat the best found if the best was 
                # found with a smaller x at this depth.
                # We can break here to save time.
                break

    # Run the main search
    start_time = time.time()
    greedy_with_backtrack([], set(), 0, max_val, deadline)
    
    return best_seq, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run the improved DFS with pruning
    # The pruning heuristic "if len(seq)+1 >= best_len, break" is the key structural change.
    # It prevents the search from wasting time on "heavier" branches that 
    # cannot possibly beat the current best length found via "lighter" branches.
    
    found_seq, found_len = _search_with_swap(max_val=256, deadline=deadline, initial_seq=None)
    if found_len > best_len:
        best = found_seq
        best_len = found_len
        
    # 3. If time remains, try a few randomized restarts with different seeds?
    # The deterministic DFS with pruning is likely very fast now.
    # Let's check time.
    
    if time.time() < deadline - 5:
        # Try a variant where we don't break as aggressively?
        # Or try a different initial sequence?
        # Let's try to extend the greedy sequence with a small randomized search.
        
        pass # The main DFS should be sufficient given the pruning.
        
    # Final validation
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
