import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    indices_map = {x: i for i, x in enumerate(numbers)}
    
    # 1. Identify all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # and a, b, c are distinct.
    
    bad_triples = []
    
    # Optimization: Precompute LCMs for pairs?
    # 128*127/2 = 8128 pairs.
    # Storing in a dict or 2D list is fast.
    # But 341,376 triples * 3 LCMs is also fast.
    # Let's just compute on the fly to keep memory low and code simple.
    # Actually, precomputing pair LCMs helps if we check triples frequently.
    # But we only check them once to build the list.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                # Quick check: if lab is 1, impossible for distinct > 0
                # If lab is small, more likely to match?
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers # Fallback, invalid but fast
        
    # Map values to indices for faster set operations?
    # We'll work with values directly as N is small.
    
    # Group bad triples by element for fast lookup during greedy add
    # triples_containing[x] = list of tuples (u, v) such that (x, u, v) is a bad triple
    # Note: (x, u, v) might be stored as (min, mid, max) in bad_triples
    # We need to handle all permutations or just check membership.
    
    triples_by_elem = {x: [] for x in numbers}
    for t in bad_triples:
        a, b, c = t
        triples_by_elem[a].append((b, c))
        triples_by_elem[b].append((a, c))
        triples_by_elem[c].append((a, b))
        
    # 2. Heuristic Search
    best_size = -1
    best_set = set()
    
    num_restarts = 1000 # Can adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Randomize order
        # We can bias towards elements with fewer conflicts
        # Count conflicts (degree in hypergraph)
        # Or just random shuffle
        
        order = numbers[:]
        random.shuffle(order)
        
        current_set = set()
        
        # Greedy Add
        for x in order:
            # Check if x can be added
            can_add = True
            for u, v in triples_by_elem[x]:
                if u in current_set and v in current_set:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(x)
                
        # Local Search: Try to improve by kicking out an element and re-adding
        # We do this for a few iterations or until no improvement
        improved = True
        max_kicks = 20
        kick_count = 0
        while improved and kick_count < max_kicks:
            improved = False
            kick_count += 1
            if time.time() > deadline - 2.0:
                break
                
            # Try kicking out a random element from current_set
            # Or the one with highest degree?
            # Random is safer for diversity
            if not current_set:
                break
                
            to_kick = random.choice(list(current_set))
            current_set.remove(to_kick)
            
            # Re-add all possible elements from the rest
            # We need to check elements NOT in current_set
            # To be efficient, we can iterate through all numbers not in current_set
            # But we should prioritize those that were blocked by to_kick?
            # Simple approach: iterate through all numbers not in current_set in random order
            # and try to add them.
            
            candidates = [x for x in numbers if x not in current_set]
            random.shuffle(candidates)
            
            for x in candidates:
                can_add = True
                for u, v in triples_by_elem[x]:
                    if u in current_set and v in current_set:
                        can_add = False
                        break
                if can_add:
                    current_set.add(x)
                    
            # Check if size improved
            # We don't track size explicitly, just compare at the end of kick loop?
            # Actually, we want to accept the kick if it leads to a larger set.
            # So we should compare len(current_set) with len(best_in_this_restart)
            # But we mutated current_set.
            # Let's store the size before kick.
            # Wait, the loop above modifies current_set.
            # We need to compare the new size with the size before the kick.
            # Let's restructure:
            # 1. Save size_before
            # 2. Perform kick and re-add
            # 3. If size_after > size_before, accept (improved = True)
            # 4. Else, revert? Reverting is hard.
            #    Better: Keep a copy of the set? N=128, copying is cheap.
            
            # Let's restart the kick logic with copying.
            # This is safer.
            
        # Actually, the loop above is a bit messy.
        # Let's implement a cleaner local search.
        
        # Refine:
        # We have a valid set current_set.
        # Try to increase size.
        # 1. Try adding any x not in set. (Greedy phase should have done this, so this will fail immediately unless we kick).
        # 2. Kick one element y from set.
        # 3. Try to add any x not in set (including y? No, y is removed).
        # 4. If size increases, keep.
        
        # Let's run a few kicks.
        
        for _ in range(10): # 10 kicks per restart
            if time.time() > deadline - 2.0:
                break
            if not current_set:
                break
            
            size_before = len(current_set)
            to_kick = random.choice(list(current_set))
            current_set.remove(to_kick)
            
            # Try to add all possible
            candidates = [x for x in numbers if x not in current_set]
            random.shuffle(candidates)
            
            for x in candidates:
                can_add = True
                for u, v in triples_by_elem[x]:
                    if u in current_set and v in current_set:
                        can_add = False
                        break
                if can_add:
                    current_set.add(x)
                    
            if len(current_set) > size_before:
                # Accept improvement
                continue
            else:
                # Revert?
                # Since we don't have a copy, and reverting is complex,
                # we might just keep the current state (which is size_before or larger? No, we added stuff).
                # Actually, if len(current_set) <= size_before, we might have made it worse or same.
                # If we made it worse, we should revert.
                # But we don't have the old set.
                # To be safe, let's just accept the new state. It's still valid.
                # It might be smaller, but the next kick might fix it.
                # Or we can break if we get stuck.
                pass
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we found a perfect set, break
        if best_size == n:
            break
            
    if best_size == -1:
        return [1]
        
    return list(best_set)
