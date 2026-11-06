import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Normalize triple to sorted tuple for deduplication if needed,
                    # but (a,b,c) structure is fixed by a < b.
                    # However, 1/a = 1/b + 1/c is not symmetric in b,c.
                    # But the set {a,b,c} is what matters for the constraint.
                    # The equation defines a specific relationship.
                    # Let's just store the set of elements.
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
        
    # For fast "can add" check, we need to know for each triple how many of its elements are in S.
    # We will maintain this count array.
    # counts[t] = number of elements of triple t currently in S.
    
    best_set = set()
    best_size = 0
    
    # Strategy: Multiple restarts of a randomized greedy + local search.
    # We will try different random seeds and different greedy heuristics.
    
    def greedy_removal(random_seed, strategy):
        """
        Strategy 0: Remove element with max conflict count (ties broken randomly).
        Strategy 1: Remove element with max conflict count, but prefer smaller numbers?
        Strategy 2: Random removal from high conflict set.
        """
        random.seed(random_seed)
        S = set(range(1, N + 1))
        counts = [0] * T
        
        # Initialize counts
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            # Initially all in S, so count is 3
            counts[t_idx] = 3
            
        # We need to efficiently find elements with high conflict.
        # Conflict of x = sum over t containing x of (1 if counts[t] == 3 else 0) ?
        # No, "active triple" means all 3 in S.
        # If we remove x, all triples containing x become inactive (count drops).
        # The "cost" of removing x is the number of active triples it hits.
        # We want to remove x that hits the most active triples.
        
        # Let's maintain active_triples count for each element.
        # active_count[x] = number of triples t containing x such that counts[t] == 3.
        
        active_count = [0] * (N + 1)
        # Initialize active_count
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
            
        # Loop until no active triples
        # Optimization: Instead of scanning all triples to find active ones,
        # we can just check if there's any active triple by checking if sum(active_count) > 0?
        # No, sum(active_count) = 3 * (# active triples).
        # So if sum(active_count) == 0, we are done.
        
        # However, updating active_count and counts is O(deg(x)) per removal.
        # Total removals <= N. Total cost O(N * max_deg).
        
        while sum(active_count) > 0:
            # Check time
            if time.time() > deadline - 5.0:
                return S
                
            # Find element with max active_count
            # We can just scan 1..N
            max_ac = 0
            candidates = []
            for x in range(1, N + 1):
                if x in S:
                    ac = active_count[x]
                    if ac > max_ac:
                        max_ac = ac
                        candidates = [x]
                    elif ac == max_ac and ac > 0:
                        candidates.append(x)
                        
            if not candidates:
                break # Should not happen if sum > 0
                
            # Choose one
            if strategy == 0:
                x_to_remove = random.choice(candidates)
            elif strategy == 1:
                # Prefer smaller number (might preserve larger numbers which have fewer triples?)
                # Actually, smaller numbers tend to be in more triples (a is small).
                # So removing small numbers might be better?
                # Let's stick to random for now, or try sorting.
                x_to_remove = min(candidates)
            else:
                x_to_remove = random.choice(candidates)
                
            S.remove(x_to_remove)
            
            # Update counts and active_count for triples containing x_to_remove
            for t_idx in elem_triples[x_to_remove]:
                if counts[t_idx] == 3:
                    # This triple was active, now it's not
                    a, b, c = triples[t_idx]
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    counts[t_idx] = 2
                elif counts[t_idx] == 2:
                    # Was already inactive (2 in S), now 1 in S
                    counts[t_idx] = 1
                # else counts[t_idx] == 1, now 0. No active count change.
                
        return S

    def local_search(S, random_seed):
        """
        Try to add back removed elements.
        """
        random.seed(random_seed)
        S = set(S)
        
        # Rebuild counts for current S
        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
                
        # Check if valid (should be)
        # For each triple, if counts[t] == 3, it's a violation.
        # But our greedy removal ensured no active triples.
        
        removed = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(removed)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            for x in removed:
                if x in S:
                    continue
                # Check if x can be added
                # x can be added if for all t containing x, counts[t] < 2.
                # Because if counts[t] == 2, adding x makes it 3 (violation).
                can_add = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    # Update counts
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                    # Remove x from removed list? 
                    # We are iterating over a static list. 
                    # It's fine, we just skip if x in S.
                    # But we should remove it from 'removed' to avoid re-checking?
                    # Not strictly necessary, but 'removed' is a list.
        
        return S

    def swap_local_search(S, random_seed):
        """
        Try to swap a removed element with a present element to improve size?
        No, we want to maximize size. Swapping 1 for 1 doesn't help size.
        We want to add elements.
        However, we might be stuck in a local optimum where no single addition is possible,
        but a swap (remove one, add two) is.
        Or, we can try to remove an element to allow adding two others.
        
        Let's try a "2-opt" style move:
        Remove x from S.
        Then try to add y, z from removed.
        If we can add 2, we net +1.
        """
        random.seed(random_seed)
        S = set(S)
        
        # Rebuild counts
        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
                
        removed = [x for x in range(1, N + 1) if x not in S]
        
        # This is expensive. Let's just do multiple passes of simple addition first.
        # The previous local_search does that.
        # Let's add a simple swap: 
        # Pick x in S. Remove it.
        # Try to add as many from removed as possible.
        # If count_added > 1, accept.
        
        # To keep it simple and fast, let's just rely on the greedy restarts.
        # The hypergraph is sparse enough that greedy + add-back might be near optimal.
        
        return S

    # Run multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Alternate strategies
        strategy = i % 2
        S_greedy = greedy_removal(i, strategy)
        S_final = local_search(S_greedy, i + 1000)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)
            
            # Optional: If we reach a very high size, we can try to verify or stop early?
            # Max possible is 180.
            # If best_size > 150, we are doing well.
            
    # Final validity check
    # If best_set is empty (should not happen), return a valid fallback
    if not best_set:
        # Fallback: just take large numbers
        best_set = set(range(90, 181))
        
    # Verify best_set
    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
            
    if not valid:
        # Fallback to a known valid set (e.g., all even numbers? No, 1/2=1/4+1/4 invalid distinct.
        # All numbers > 90?
        # 1/a = 1/b + 1/c. If a,b,c > 90, 1/a < 1/90, 1/b+1/c < 2/90 = 1/45.
        # Is it possible?
        # 1/91 = 1/92 + 1/184? 1/92 + 1/184 = 2/184 + 1/184 = 3/184 = 1/61.33. No.
        # Generally, for large a,b,c, 1/a approx 1/b + 1/c implies a approx b/2 or c/2.
        # If a,b,c > 90, a < b, c.
        # Min a=91. Max b,c=180.
        # 1/91 = 1/180 + 1/x => 1/x = 1/91 - 1/180 = (180-91)/(91*180) = 89/16380.
        # x = 16380/89 = 184.04. Not integer.
        # It is likely that {91..180} is valid.
        # Let's just remove the offending elements from best_set.
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                best_set.remove(c) # Remove the largest? Or random?
                
        # Re-verify
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                # Should be clean now
                pass

    return list(best_set)

# EVOLVE-BLOCK-END
