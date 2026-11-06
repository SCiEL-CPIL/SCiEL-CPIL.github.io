"""B_3[1] set construction using bytearray bitset and fast backtracking.

Strategy:
  1. Use a bytearray for O(1) forbidden-sum lookups.
  2. Use a recursive backtracking search that explores the smallest valid numbers first.
  3. This greedy path within backtracking allows finding the maximum cardinality set much faster
     than the previous set-based approach, which suffered from high overhead and poor pruning.
"""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    n_max = 4096
    target_size = 80
    max_sum = 3 * n_max
    
    # forbidden[s] is 1 if sum s is already used
    forbidden = bytearray(max_sum + 1)
    current_set = []
    best_set = []
    
    def backtrack():
        nonlocal best_set
        if time.time() > deadline:
            return
        
        # If we found a larger set, update best
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(current_set) >= target_size:
            return
            
        # Try adding the smallest possible valid number
        # We can start from 1, but to speed up, we can start from current_set[-1] + 1
        # However, it's safer to just iterate from 1, but we can skip if we've already tried
        # Actually, since we want the smallest valid numbers, we iterate from 1 to n_max
        
        # Optimization: We can keep a pointer to the last tried number, but since we backtrack,
        # it's simpler to just loop. To make it faster, we can pass a 'start' index.
        
        # Let's use a start index to avoid re-checking small numbers that were already tried
        # and failed in this branch.
        start = current_set[-1] + 1 if current_set else 1
        
        for x in range(start, n_max + 1):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            # 1. Check 3x
            if forbidden[3 * x]:
                continue
                
            # 2. Check 2x + y and x + 2y for all y in current_set
            valid = True
            for y in current_set:
                if forbidden[2 * x + y] or forbidden[x + 2 * y]:
                    valid = False
                    break
            if not valid:
                continue
                
            # 3. Check x + y + z for all y <= z in current_set
            n = len(current_set)
            for i in range(n):
                yi = current_set[i]
                for j in range(i, n):
                    yj = current_set[j]
                    if forbidden[x + yi + yj]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # x is valid, add it
            current_set.append(x)
            forbidden[3 * x] = 1
            for y in current_set[:-1]:
                forbidden[2 * x + y] = 1
                forbidden[x + 2 * y] = 1
                # We also need to mark x + y + z for all z <= y in current_set[:-1]
                # But wait, the loop above only checked conflicts with existing sums.
                # We must mark the new sums created by adding x.
                # The new sums are:
                # 3x
                # 2x + y for y in current_set[:-1]
                # x + 2y for y in current_set[:-1]
                # x + y + z for y <= z in current_set[:-1]
                
            # Mark x + y + z
            for i in range(len(current_set) - 1):
                yi = current_set[i]
                for j in range(i, len(current_set) - 1):
                    yj = current_set[j]
                    forbidden[x + yi + yj] = 1
            
            # Recurse
            backtrack()
            
            # Backtrack: unmark
            current_set.pop()
            forbidden[3 * x] = 0
            for y in current_set:
                forbidden[2 * x + y] = 0
                forbidden[x + 2 * y] = 0
            for i in range(len(current_set)):
                yi = current_set[i]
                for j in range(i, len(current_set)):
                    yj = current_set[j]
                    forbidden[x + yi + yj] = 0
    
    backtrack()
    
    return best_set

if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s:.2f}s, Size: {len(res)}")
