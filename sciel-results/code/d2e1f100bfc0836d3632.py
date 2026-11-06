import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    base = [1, 2, 5, 11]
    base_set = set(base)
    max_val = 284
    
    # Pre-compute all candidates
    all_candidates = list(range(1, max_val + 1))
    
    def get_sums(s):
        """Return a set of all pairwise sums a+b where a<=b from sorted list s."""
        sums = set()
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                b = s[j]
                sums.add(a + b)
        return sums
    
    def can_add_fast(s, sums, x):
        """Check if x can be added to s without creating duplicate sums. O(n) check."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def add_element(s, sums, x):
        """Add x to s and update sums incrementally. Returns new s and sums."""
        new_s = s + [x]
        new_s.sort()
        new_sums = set(sums)
        for y in s:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_s, new_sums
    
    def greedy_fill(s, sums, candidates, bias=None, time_limit=None):
        """
        Greedily add elements from candidates to s.
        Returns the updated sorted list s and the updated sums set.
        """
        s = sorted(s)
        s_set = set(s)
        avail = [c for c in candidates if c not in s_set]
        
        if bias == 'random':
            random.shuffle(avail)
        elif bias == 'low':
            avail.sort()
        elif bias == 'high':
            avail.sort(reverse=True)
        elif bias == 'sparse':
            # Prefer elements that are far from existing elements (less sum pressure)
            def pressure(c):
                return sum(1 for y in s if abs(c - y) < 30)
            avail.sort(key=pressure, reverse=True)
        elif bias == 'dense':
            # Prefer elements close to existing elements
            def pressure(c):
                return sum(1 for y in s if abs(c - y) < 30)
            avail.sort(key=pressure)
        else:
            random.shuffle(avail)
            
        for x in avail:
            if time_limit and time.time() > time_limit:
                break
            if time.time() > deadline - 2:
                break
            if can_add_fast(s, sums, x):
                s, sums = add_element(s, sums, x)
                
        return s, sums
    
    def kick_and_refill(current_set, current_sums, kick_size, bias=None):
        """
        Remove 'kick_size' non-base elements from current_set, then greedy refill.
        """
        if time.time() > deadline - 5:
            return current_set, current_sums
            
        # Identify non-base elements
        non_base_elements = [x for x in current_set if x not in base_set]
        
        if len(non_base_elements) < kick_size:
            new_set, new_sums = greedy_fill(current_set, current_sums, all_candidates, bias=bias or 'random')
            return new_set, new_sums
            
        # Randomly select elements to remove
        to_remove = random.sample(non_base_elements, kick_size)
        remove_set = set(to_remove)
        
        # Create new set without removed elements
        new_set = [x for x in current_set if x not in remove_set]
        new_sums = get_sums(new_set)
        
        # Refill with random bias
        new_set, new_sums = greedy_fill(new_set, new_sums, all_candidates, bias=bias or 'random')
        
        return new_set, new_sums
    
    def try_extend(s, sums):
        """Try to add one more element by systematic search."""
        s_set = set(s)
        for x in all_candidates:
            if x in s_set:
                continue
            if can_add_fast(s, sums, x):
                new_s, new_sums = add_element(s, sums, x)
                return new_s, new_sums
        return s, sums
    
    # Initial Solution
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)
    
    # Try several initial greedy fills with different biases to find a good starting point
    biases = ['random', 'low', 'high', 'sparse', 'dense', 'random', 'random']
    for bias in biases:
        if time.time() > deadline - 25:
            break
        s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias=bias)
        if len(s) > best_size:
            best_set = s
            best_sums = sums
            best_size = len(s)
            
    # Local Search Loop
    current_set = best_set[:]
    current_sums = best_sums
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 40
    
    # Track best kick strategies
    kick_history = []
    
    while time.time() < deadline - 15:
        
        # Determine kick size with some strategy
        r = random.random()
        if r < 0.4:
            kick_size = 1
        elif r < 0.8:
            kick_size = 2
        else:
            kick_size = 3
            
        # Choose bias randomly
        bias_choices = ['random', 'random', 'low', 'high', 'sparse', 'dense']
        bias = random.choice(bias_choices)
        
        new_set, new_sums = kick_and_refill(current_set, current_sums, kick_size, bias=bias)
        
        if len(new_set) > best_size:
            best_set = new_set
            best_sums = new_sums
            best_size = len(new_set)
            current_set = new_set
            current_sums = new_sums
            current_size = best_size
            no_improve_count = 0
        elif len(new_set) == best_size:
            # Accept equal size moves to explore the landscape
            if new_set != current_set:
                current_set = new_set
                current_sums = new_sums
                current_size = len(new_set)
            no_improve_count += 1
        else:
            # Worse size - sometimes accept to escape local optima
            if random.random() < 0.3:
                current_set = new_set
                current_sums = new_sums
                current_size = len(new_set)
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Reset current to best to restart exploration
            current_set = best_set[:]
            current_sums = best_sums
            current_size = best_size
            no_improve_count = 0
            
    # Final polish: try to extend the best set
    if time.time() < deadline - 5:
        polished_set, polished_sums = try_extend(best_set, best_sums)
        if len(polished_set) > best_size:
            best_set = polished_set
            best_sums = polished_sums
            best_size = len(best_set)
            
    # One more attempt with a fresh restart if time permits
    if time.time() < deadline - 10:
        for bias in ['random', 'sparse']:
            if time.time() > deadline - 5:
                break
            s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias=bias)
            if len(s) > best_size:
                best_set = s
                best_sums = sums
                best_size = len(s)
            # Try a few kicks from this new start
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                new_s, new_sums = kick_and_refill(s, sums, random.randint(1, 3), bias='random')
                if len(new_s) > best_size:
                    best_set = new_s
                    best_sums = new_sums
                    best_size = len(new_s)
                    s, sums = new_s, new_sums
    
    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
