import math
import random

def construct():
    # Calculate LCM of 1 to 22
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    D = 1
    for i in range(1, 23):
        D = lcm(D, i)
        
    def greedy(order):
        current_sums = {0}
        current_set = []
        
        for x in order:
            inc = D // x
            # Check for collision
            # Optimization: early exit
            collision = False
            for s in current_sums:
                if (s + inc) in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Update sums
                # We can do this in a way that avoids creating a huge intermediate set if possible, 
                # but set union is standard.
                new_sums = {s + inc for s in current_sums}
                current_sums.update(new_sums)
                current_set.append(x)
                
        return current_set

    # Try multiple orders and keep the best
    best_set = []
    
    # Order 1: Descending x (22 to 1) - Small values first
    order1 = list(range(22, 0, -1))
    s1 = greedy(order1)
    if len(s1) > len(best_set):
        best_set = s1

    # Order 2: Ascending x (1 to 22) - Large values first
    order2 = list(range(1, 23))
    s2 = greedy(order2)
    if len(s2) > len(best_set):
        best_set = s2

    # Order 3: Random shuffles
    # Set seed for reproducibility if needed, but random is fine for exploration
    # We will try a few random permutations
    random.seed(42)
    for _ in range(50):
        order = list(range(1, 23))
        random.shuffle(order)
        s_rand = greedy(order)
        if len(s_rand) > len(best_set):
            best_set = s_rand

    return best_set
