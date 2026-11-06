import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 140s to be safe.
    deadline = start_time + 140.0
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper to check if a candidate x can be added to the current set
    def can_add(x, a_set, sums_set):
        if x in a_set:
            return False
        # Check sums with existing elements
        for y in a_set:
            s = x + y
            if s in sums_set:
                return False
        # Check double sum
        if 2 * x in sums_set:
            return False
        return True

    def get_sums(a_list):
        s = set()
        for i, x in enumerate(a_list):
            for y in a_list[i:]:
                s.add(x + y)
        return s

    def greedy_expand(initial_list):
        a = list(initial_list)
        a_set = set(a)
        sums = get_sums(a)
        
        changed = True
        while changed:
            changed = False
            # Iterate through all possible candidates
            # We can optimize this by only checking candidates that don't conflict
            # But 1..284 is small enough.
            for x in range(1, 285):
                if x in a_set:
                    continue
                if can_add(x, a_set, sums):
                    a.append(x)
                    a_set.add(x)
                    # Update sums
                    for y in a[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    changed = True
                    # Restart loop to pick up new opportunities in the same order
                    break
        return a, a_set, sums

    # Get a good initial guess using greedy
    current_list, current_set, current_sums = greedy_expand(fixed_elements)
    best_list = current_list[:]
    best_size = len(best_list)
    
    # Main search loop: Randomized Local Search with Kicks
    # We will try to add elements. If stuck, we kick (remove) some elements.
    
    # To speed up "find an element to add", we can maintain a list of candidates.
    # But since the range is small (284), iterating through 1..284 is fast (284 checks).
    
    # We will use a randomized restart strategy if we get stuck for a while
    # or we can just do random kicks.
    
    # Strategy:
    # 1. Try to add any valid element.
    # 2. If none can be added, remove a random non-fixed element (kick).
    # 3. Repeat.
    
    # To avoid getting stuck in a specific local optimum, we can also
    # occasionally restart from the greedy solution or a random subset.
    
    # Let's implement a simple kick-based search.
    # We'll keep track of the best solution found.
    
    # Optimization: Instead of scanning 1..284 every time, we can maintain a list of candidates.
    # But since the set size is small, scanning is fast.
    
    # Let's try to be smarter about kicks.
    # If we remove an element, we might be able to add multiple new ones.
    
    # We'll use a time-based loop.
    
    # Random seed for reproducibility? No, random is fine.
    
    # Let's add a "restart" mechanism: if we haven't improved for a while,
    # we might restart from the greedy solution or a different greedy order.
    
    # Actually, simple random kicks are often effective for small N.
    
    # Let's improve the candidate selection.
    # Instead of random.shuffle on 1..284 every time, we can just pick a random candidate.
    # But we need to find *any* valid candidate to add.
    # So we must scan until we find one.
    
    # Let's try a different approach for the inner loop:
    # 1. Generate a random permutation of candidates 1..284.
    # 2. Try to add the first one that fits.
    # 3. If none fits, kick.
    
    # This is what the previous code did. Let's stick to it but optimize the kick.
    
    # Kick strategy:
    # - Remove 1 random non-fixed element.
    # - If that doesn't help (i.e., we still can't add anything after a few tries),
    #   maybe remove 2.
    
    # Let's just remove 1 random non-fixed element each time we are stuck.
    
    # To avoid infinite loops of adding/removing the same elements,
    # we can track the last few states, but for a 140s budget, it's unlikely to be a problem
    # unless we are very unlucky.
    
    # Let's also try to bias the kick towards elements that are "blocking" others.
    # This is hard to compute. Random is simpler.
    
    # Let's run the search.
    
    while time.time() < deadline:
        # 1. Try to add elements
        added = True
        while added and time.time() < deadline:
            added = False
            # Randomize the order of candidates to avoid bias
            candidates = [x for x in range(1, 285) if x not in current_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, current_set, current_sums):
                    # Add x
                    current_list.append(x)
                    current_set.add(x)
                    # Update sums
                    for y in current_list[:-1]:
                        current_sums.add(x + y)
                    current_sums.add(2 * x)
                    added = True
                    # Update best if improved
                    if len(current_list) > best_size:
                        best_size = len(current_list)
                        best_list = current_list[:]
                    # Break to restart candidate shuffling
                    break
        
        # 2. If we couldn't add anything, we are in a local optimum.
        if not added:
            non_fixed = [x for x in current_list if x not in fixed_set]
            
            if not non_fixed:
                # We are at the base size. We can't kick.
                # We might need to try different random seeds or restarts,
                # but with a fixed deadline, we just exit or wait.
                # Since we can't improve, we break.
                break
            
            # Perform a kick: remove 1 or 2 random non-fixed elements
            # Removing too many might be too costly, removing too few might not help.
            # Let's try removing 1 element first. If that fails repeatedly, try 2.
            # Simple strategy: remove 1 random non-fixed element.
            
            # To avoid oscillating between the same states, we can track recent states
            # but for now, simple random kick is effective.
            
            k = 1
            if random.random() < 0.2:
                k = 2
            if random.random() < 0.05:
                k = 3
            
            # Select elements to remove
            to_remove = random.sample(non_fixed, min(k, len(non_fixed)))
            
            # Remove them
            for x in to_remove:
                current_list.remove(x)
                current_set.remove(x)
            
            # Rebuild sums
            current_sums = get_sums(current_list)
                    
    return best_list

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found size: {len(res)}")
    # Verify
    s = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            sm = res[i] + res[j]
            if sm in s:
                valid = False
            s.add(sm)
    print(f"Valid: {valid}")
    print(f"Contains fixed: {set([1,2,5,11]).issubset(set(res))}")
