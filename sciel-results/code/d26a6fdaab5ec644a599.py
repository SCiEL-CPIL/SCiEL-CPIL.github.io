import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_size = 0
    best_A = []
    best_B = []
    
    # We will run multiple restarts of Simulated Annealing
    # Each restart uses a different random seed for the initial greedy construction
    
    rng = random.Random(42)
    
    # Precompute all possible pairs for Sidon checks? No, dynamic is better for sparse sets.
    # But we can precompute difference sets? No, dynamic is fine.
    
    def get_initial_state():
        """Greedy construction to get a valid starting state."""
        A = []
        B = []
        sum_A = {}
        sum_B = {}
        diff_A = set()
        diff_B = set()
        
        # Try to add elements 1..N to A and B
        # We alternate or balance based on current sizes
        for x in range(1, N + 1):
            # Try adding to A
            valid_A = True
            for y in A:
                s = x + y
                if s in sum_A:
                    valid_A = False
                    break
                d = abs(x - y)
                if d in diff_A or d in diff_B:
                    valid_A = False
                    break
            if valid_A:
                # Check self-sum
                if x + x in sum_A:
                    valid_A = False
            if valid_A and len(A) < MAX_EACH:
                A.append(x)
                sum_A[x+x] = 1
                for y in A[:-1]:
                    sum_A[x+y] = 1
                    sum_A[y+x] = 1
                    diff_A.add(abs(x-y))
                continue
            
            # Try adding to B
            valid_B = True
            for y in B:
                s = x + y
                if s in sum_B:
                    valid_B = False
                    break
                d = abs(x - y)
                if d in diff_B or d in diff_A:
                    valid_B = False
                    break
            if valid_B:
                if x + x in sum_B:
                    valid_B = False
            if valid_B and len(B) < MAX_EACH:
                B.append(x)
                sum_B[x+x] = 1
                for y in B[:-1]:
                    sum_B[x+y] = 1
                    sum_B[y+x] = 1
                    diff_B.add(abs(x-y))
                
        return A, B, sum_A, sum_B, diff_A, diff_B

    def is_valid_addition(x, A, B, sum_A, sum_B, diff_A, diff_B, target_set):
        """Check if x can be added to target_set (0 for A, 1 for B)."""
        if target_set == 0:
            other_set = B
            other_sum = sum_B
            other_diff = diff_B
            curr_sum = sum_A
            curr_diff = diff_A
        else:
            other_set = A
            other_sum = sum_A
            other_diff = diff_A
            curr_sum = sum_B
            curr_diff = diff_B
        
        curr_set = A if target_set == 0 else B
        
        # Check self sum
        if x + x in curr_sum:
            return False
        
        for y in curr_set:
            s = x + y
            if s in curr_sum:
                return False
            d = abs(x - y)
            if d in curr_diff or d in other_diff:
                return False
        
        return True

    def add_element(x, A, B, sum_A, sum_B, diff_A, diff_B, target_set):
        """Add x to target set. Assumes validity was checked."""
        if target_set == 0:
            A.append(x)
            sum_A[x+x] = 1
            for y in A[:-1]:
                sum_A[x+y] = 1
                sum_A[y+x] = 1
                diff_A.add(abs(x-y))
        else:
            B.append(x)
            sum_B[x+x] = 1
            for y in B[:-1]:
                sum_B[x+y] = 1
                sum_B[y+x] = 1
                diff_B.add(abs(x-y))

    def remove_element(x, A, B, sum_A, sum_B, diff_A, diff_B, target_set):
        """Remove x from target set."""
        if target_set == 0:
            idx = A.index(x)
            A.pop(idx)
            # Rebuild sums and diffs for A? 
            # For simplicity and correctness in SA, rebuilding is O(k^2) which is small (64^2=4096).
            # Given the speed of Python, this might be acceptable if we limit steps.
            # However, incremental is better. Let's try incremental removal.
            # We need to decrement counts.
            # But we used sets for diffs and dicts for sums.
            # Let's stick to rebuilding for robustness if k is small.
            # Actually, let's just rebuild. It's simpler and less error-prone for the first pass.
            # If it's too slow, we'll optimize.
            
            # Rebuild A
            new_sum_A = {}
            new_diff_A = set()
            for i in range(len(A)):
                for j in range(i, len(A)):
                    new_sum_A[A[i] + A[j]] = 1
                    if i != j:
                        new_diff_A.add(abs(A[i] - A[j]))
            sum_A.clear()
            sum_A.update(new_sum_A)
            diff_A.clear()
            diff_A.update(new_diff_A)
        else:
            idx = B.index(x)
            B.pop(idx)
            new_sum_B = {}
            new_diff_B = set()
            for i in range(len(B)):
                for j in range(i, len(B)):
                    new_sum_B[B[i] + B[j]] = 1
                    if i != j:
                        new_diff_B.add(abs(B[i] - B[j]))
            sum_B.clear()
            sum_B.update(new_sum_B)
            diff_B.clear()
            diff_B.update(new_diff_B)

    # Simulated Annealing
    def simulated_annealing(A, B, sum_A, sum_B, diff_A, diff_B, max_time):
        current_size = len(A) + len(B)
        best_local_size = current_size
        best_local_A = list(A)
        best_local_B = list(B)
        
        T = 1.0
        cooling = 0.999
        steps = 0
        
        while time.time() < max_time:
            # Choose a move
            move_type = rng.random()
            if move_type < 0.3:
                # Try Add
                # Pick a random set
                target = rng.choice([0, 1])
                if len(A if target==0 else B) < MAX_EACH:
                    # Pick a random candidate not in A or B
                    candidates = [x for x in range(1, N+1) if x not in A and x not in B]
                    if candidates:
                        x = rng.choice(candidates)
                        if is_valid_addition(x, A, B, sum_A, sum_B, diff_A, diff_B, target):
                            add_element(x, A, B, sum_A, sum_B, diff_A, diff_B, target)
                            new_size = len(A) + len(B)
                            if new_size > best_local_size:
                                best_local_size = new_size
                                best_local_A = list(A)
                                best_local_B = list(B)
                            # Accept with probability 1 (since it improves or is neutral in terms of size, 
                            # but actually we only add if valid, so size increases by 1)
                            # Wait, if we add, size always increases. So always accept.
                            steps += 1
                            T *= cooling
                            continue
            elif move_type < 0.5:
                # Try Remove (to escape local minima)
                # Only if we have elements to remove
                if len(A) > 0 and len(B) > 0:
                    target = rng.choice([0, 1])
                    # Prefer removing from larger set?
                    if len(A) > len(B):
                        target = 0
                    elif len(B) > len(A):
                        target = 1
                    else:
                        target = rng.choice([0, 1])
                    
                    curr_set = A if target == 0 else B
                    if curr_set:
                        x = rng.choice(curr_set)
                        remove_element(x, A, B, sum_A, sum_B, diff_A, diff_B, target)
                        new_size = len(A) + len(B)
                        delta = new_size - current_size # delta = -1
                        # Accept if random < exp(delta/T)
                        if rng.random() < pow(2.71828, delta/T):
                            current_size = new_size
                            steps += 1
                            T *= cooling
                            continue
                        else:
                            # Revert
                            add_element(x, A, B, sum_A, sum_B, diff_A, diff_B, target)
            else:
                # Try Swap: Remove x from Set 1, Add y to Set 2 (or vice versa)
                # Or Swap within same set? No, that doesn't change size.
                # Let's try: Remove x from A, Add y to B.
                if len(A) > 0 and len(B) < MAX_EACH:
                    x = rng.choice(A)
                    # Tentatively remove x
                    remove_element(x, A, B, sum_A, sum_B, diff_A, diff_B, 0)
                    # Try to add y to B
                    candidates = [y for y in range(1, N+1) if y not in A and y not in B]
                    if candidates:
                        # Try a few random candidates
                        found = False
                        for _ in range(10): # Limit search depth for swap
                            y = rng.choice(candidates)
                            if is_valid_addition(y, A, B, sum_A, sum_B, diff_A, diff_B, 1):
                                add_element(y, A, B, sum_A, sum_B, diff_A, diff_B, 1)
                                new_size = len(A) + len(B)
                                delta = new_size - current_size # 0
                                # Accept with prob 1 if delta >= 0, else exp
                                if delta >= 0 or rng.random() < pow(2.71828, delta/T):
                                    current_size = new_size
                                    if new_size > best_local_size:
                                        best_local_size = new_size
                                        best_local_A = list(A)
                                        best_local_B = list(B)
                                    found = True
                                    break
                        if not found:
                            # Revert
                            add_element(x, A, B, sum_A, sum_B, diff_A, diff_B, 0)
                elif len(B) > 0 and len(A) < MAX_EACH:
                    x = rng.choice(B)
                    remove_element(x, A, B, sum_A, sum_B, diff_A, diff_B, 1)
                    candidates = [y for y in range(1, N+1) if y not in A and y not in B]
                    if candidates:
                        found = False
                        for _ in range(10):
                            y = rng.choice(candidates)
                            if is_valid_addition(y, A, B, sum_A, sum_B, diff_A, diff_B, 0):
                                add_element(y, A, B, sum_A, sum_B, diff_A, diff_B, 0)
                                new_size = len(A) + len(B)
                                delta = new_size - current_size
                                if delta >= 0 or rng.random() < pow(2.71828, delta/T):
                                    current_size = new_size
                                    if new_size > best_local_size:
                                        best_local_size = new_size
                                        best_local_A = list(A)
                                        best_local_B = list(B)
                                    found = True
                                    break
                        if not found:
                            add_element(x, A, B, sum_A, sum_B, diff_A, diff_B, 1)
            
            if T < 0.01:
                break

        return best_local_A, best_local_B, best_local_size

    # Run multiple restarts
    while time.time() < deadline - 5.0:
        A, B, sum_A, sum_B, diff_A, diff_B = get_initial_state()
        
        # Run SA for a short duration
        sa_deadline = time.time() + 5.0
        if sa_deadline > deadline - 5.0:
            sa_deadline = deadline - 5.0
            
        A, B, size = simulated_annealing(A, B, sum_A, sum_B, diff_A, diff_B, sa_deadline)
        
        if size > best_size:
            best_size = size
            best_A = list(A)
            best_B = list(B)
            
        # If we hit the max possible, stop
        if best_size >= 2 * MAX_EACH:
            break
            
    return [best_A, best_B]

if __name__ == "__main__":
    print(construct())
