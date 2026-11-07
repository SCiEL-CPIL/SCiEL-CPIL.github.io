import time
import random

def construct():
    deadline = time.time() + 52
    max_val = 256
    max_size = 60

    def is_valid_set(S):
        """Check if set S is valid."""
        if not S:
            return True
        max_sum = sum(S) + 1
        mask = (1 << max_sum) - 1
        sums = [0] * (len(S) + 1)
        sums[0] = 1
        for x in S:
            new_sums = [0] * (len(S) + 1)
            new_sums[0] = sums[0]
            for k in range(1, len(S) + 1):
                new_sums[k] = sums[k] | ((sums[k-1] << x) & mask)
            for k in range(len(S) + 1):
                for j in range(k+1, len(S) + 1):
                    if new_sums[k] & new_sums[j]:
                        return False
            sums = new_sums
        return True

    def get_sum_bitsets(S):
        """Return list of bitsets for each cardinality."""
        max_sum = sum(S) + 1
        mask = (1 << max_sum) - 1
        sums = [0] * (len(S) + 1)
        sums[0] = 1
        for x in S:
            new_sums = [0] * (len(S) + 1)
            new_sums[0] = sums[0]
            for k in range(1, len(S) + 1):
                new_sums[k] = sums[k] | ((sums[k-1] << x) & mask)
            sums = new_sums
        return sums

    def try_add(S, x):
        """Check if x can be added to S. Returns (valid, new_S)."""
        if x in S:
            return False, S
        test_S = S + [x]
        if is_valid_set(test_S):
            return True, test_S
        return False, S

    def validate_and_build(candidates, deadline):
        """Greedily build a valid set."""
        result = []
        for x in candidates:
            if x < 1 or x > 256:
                continue
            if x in result:
                continue
            valid, new_result = try_add(result, x)
            if valid:
                result = new_result
                if len(result) >= max_size:
                    break
            if time.time() > deadline:
                break
        return result

    # Generate initial candidates
    random.seed(42)
    best = []
    
    # Try different orderings
    orderings = [
        range(1, 257),
        range(256, 0, -1),
        sorted(range(1, 257), key=lambda x: random.random()),
        sorted(range(1, 257), key=lambda x: random.random()),
        sorted(range(1, 257), key=lambda x: random.random()),
    ]
    
    for ordering in orderings:
        res = validate_and_build(ordering, deadline)
        if len(res) > len(best):
            best = res
        if time.time() > deadline:
            break

    # Simulated Annealing / Hill Climbing
    if len(best) < max_size and time.time() < deadline:
        current_set = best
        current_sum = sum(current_set)
        best_set = current_set
        best_sum = current_sum
        
        used = set(current_set)
        remaining = [x for x in range(1, 257) if x not in used]
        
        temperature = 100.0
        cooling_rate = 0.99
        
        while time.time() < deadline - 2:
            improved = False
            
            # Try to add a random element
            if remaining:
                x = random.choice(remaining)
                valid, new_set = try_add(current_set, x)
                if valid:
                    current_set = new_set
                    used.add(x)
                    remaining.remove(x)
                    current_sum = sum(current_set)
                    if len(current_set) > len(best_set):
                        best_set = current_set
                        best_sum = current_sum
                    improved = True
                    temperature = max(1.0, temperature * cooling_rate)
                    continue
            
            # Try to swap
            if current_set:
                # Pick random element to remove
                i = random.randint(0, len(current_set) - 1)
                old_val = current_set[i]
                
                # Pick random element to add
                if remaining:
                    new_val = random.choice(remaining)
                    test_set = current_set[:]
                    test_set[i] = new_val
                    # Ensure distinctness
                    if len(set(test_set)) != len(test_set):
                        continue
                    
                    if is_valid_set(test_set):
                        new_sum = sum(test_set)
                        delta = new_sum - current_sum
                        # Accept if better (smaller sum) or with probability exp(-delta/T) if worse
                        if delta < 0 or random.random() < pow(2.71828, -delta / max(temperature, 1e-6)):
                            current_set = test_set
                            used.remove(old_val)
                            used.add(new_val)
                            remaining.remove(new_val)
                            remaining.append(old_val)
                            current_sum = new_sum
                            if len(current_set) > len(best_set):
                                best_set = current_set
                                best_sum = current_sum
                            improved = True
                            temperature = max(1.0, temperature * cooling_rate)
                            continue
            
            if not improved:
                temperature = max(1.0, temperature * 0.95)
                
                # If stuck, try a greedy restart from current best?
                # Or just keep trying swaps with lower temp.

    best = best_set
    best = [int(x) for x in best]
    return best
