import random
import time
import math

def count_3aps_with_x(A, x):
    """Count 3-APs in A completed by adding x."""
    count = 0
    n = max(A) if A else 0
    # x is middle
    for d in range(1, n + 1):
        if (x - d in A) and (x + d in A):
            count += 1
    # x is end
    for d in range(1, n + 1):
        if (x - d in A) and (x - 2 * d in A):
            count += 1
    # x is start
    for d in range(1, n + 1):
        if (x + d in A) and (x + 2 * d in A):
            count += 1
    return count

def has_4ap_with_x(A, x):
    """Check if adding x to A creates a 4-AP."""
    n = max(A) if A else 256
    # x, x+d, x+2d, x+3d
    for d in range(1, n // 3 + 1):
        if (x + d in A) and (x + 2 * d in A) and (x + 3 * d in A):
            return True
    # x-d, x, x+d, x+2d
    for d in range(1, n // 2 + 1):
        if (x - d in A) and (x + d in A) and (x + 2 * d in A):
            return True
    # x-2d, x-d, x, x+d
    for d in range(1, n // 2 + 1):
        if (x - 2 * d in A) and (x - d in A) and (x + d in A):
            return True
    # x-3d, x-2d, x-d, x
    for d in range(1, n // 3 + 1):
        if (x - 3 * d in A) and (x - 2 * d in A) and (x - d in A):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    best_set = set()
    best_score = -1
    
    # Try different initial densities and seeds
    for density in [0.15, 0.2, 0.25]:
        for seed in range(10):
            if time.time() > deadline:
                break
                
            random.seed(seed * 1000 + int(density * 100))
            A = set(random.sample(range(1, 257), int(256 * density)))
            
            # Prune 4-APs
            # We repeatedly find a 4-AP and remove an element
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Find a 4-AP
                found_4ap = None
                for a in A:
                    for d in range(1, (256 - a) // 3 + 1):
                        if (a + d in A) and (a + 2 * d in A) and (a + 3 * d in A):
                            found_4ap = (a, a + d, a + 2 * d, a + 3 * d)
                            break
                    if found_4ap:
                        break
                if found_4ap:
                    # Remove element with lowest 3-AP participation
                    min_part = float('inf')
                    min_elem = found_4ap[0]
                    for elem in found_4ap:
                        # Approximate participation: count 3-APs in A containing elem
                        # This is O(n^2) per removal, might be slow.
                        # Simpler: just remove random element from 4-AP
                        min_elem = random.choice(found_4ap)
                        min_part = 0 # dummy
                    A.remove(min_elem)
                    changed = True
            
            # If A is 4-AP free, calculate score
            # Optimize with Simulated Annealing
            A_sa = set(A)
            score = count_3aps(A_sa)
            
            # SA Loop
            T = 10.0
            cooling = 0.99
            stuck_counter = 0
            
            while T > 0.01 and time.time() < deadline:
                # Pick a move
                if random.random() < 0.5 and len(A_sa) < 256:
                    # Add
                    candidates = [x for x in range(1, 257) if x not in A_sa]
                    if candidates:
                        x = random.choice(candidates)
                        if not has_4ap_with_x(A_sa, x):
                            new_score = score + count_3aps_with_x(A_sa, x)
                            delta = new_score - score
                            if delta > 0 or random.random() < math.exp(-delta / T):
                                A_sa.add(x)
                                score = new_score
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                else:
                    # Swap or Remove
                    if A_sa:
                        x = random.choice(list(A_sa))
                        # Try to remove x and add y
                        candidates = [y for y in range(1, 257) if y not in A_sa]
                        if candidates:
                            y = random.choice(candidates)
                            # Check if (A_sa - {x} + {y}) is 4-AP free
                            # This is expensive. 
                            # Heuristic: Check if y creates 4-AP with A_sa - {x}
                            # And check if x was part of any 4-AP (it wasn't, so removing it is safe wrt 4-APs created by others? No, removing x might break 4-APs, but we only add y if it doesn't create new ones).
                            # Wait, if A_sa is 4-AP free, then A_sa - {x} is also 4-AP free.
                            # We just need to check if adding y to (A_sa - {x}) creates a 4-AP.
                            temp_A = A_sa - {x}
                            if not has_4ap_with_x(temp_A, y):
                                # Calculate new score
                                # Remove x's contribution, add y's contribution
                                # Score change = -contrib(x) + contrib(y)
                                # But interactions between x and y are gone.
                                # Simpler: recalculate score? O(n^2).
                                # Or maintain count.
                                # For simplicity in this sketch, let's just recalculate if T is low or move is accepted.
                                # Actually, let's just do a full check for validity and score update carefully.
                                
                                # Let's just do:
                                A_new = temp_A | {y}
                                if not has_4ap_in_set(A_new): # Need this function
                                    new_score = count_3aps(A_new)
                                    delta = new_score - score
                                    if delta > 0 or random.random() < math.exp(-delta / T):
                                        A_sa = A_new
                                        score = new_score
                                        stuck_counter = 0
                                    else:
                                        stuck_counter += 1
                        else:
                            # Just remove
                            A_new = A_sa - {x}
                            new_score = count_3aps(A_new)
                            delta = new_score - score
                            if delta > 0 or random.random() < math.exp(-delta / T):
                                A_sa = A_new
                                score = new_score
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                T *= cooling
                if stuck_counter > 100:
                    T *= 0.5
                    stuck_counter = 0
                
            if score > best_score:
                best_score = score
                best_set = A_sa
                
    return sorted(best_set)

# Helper functions needed
def count_3aps(A):
    count = 0
    A_list = sorted(A)
    for i in range(len(A_list)):
        for j in range(i + 1, len(A_list)):
            # x, y, z. 2y = x + z
            z = 2 * A_list[j] - A_list[i]
            if z in A and z > A_list[j]:
                count += 1
    return count

def has_4ap_in_set(A):
    for a in A:
        for d in range(1, (256 - a) // 3 + 1):
            if (a + d in A) and (a + 2 * d in A) and (a + 3 * d in A):
                return True
    return False
