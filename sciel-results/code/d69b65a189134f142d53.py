import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Prescribed elements
    base_elements = [1, 2, 5, 11]
    
    def get_sums(elements):
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sums.add(elements[i] + elements[j])
        return sums

    # Initial sums
    initial_sums = get_sums(base_elements)
    
    def is_valid(x, current_elements, current_sums):
        # Check if adding x creates any duplicate sums
        # New sums will be x + y for y in current_elements, and 2*x
        # We need to ensure these are not in current_sums
        # And that they are distinct among themselves (they are, since y are distinct)
        # So we just check intersection with current_sums
        for y in current_elements:
            if x + y in current_sums:
                return False
        if 2 * x in current_sums:
            return False
        return True

    def construct_once(strategy="heuristic"):
        current_elements = list(base_elements)
        current_sums = initial_sums.copy()
        
        while True:
            # Find all valid candidates
            valid_candidates = []
            for x in range(1, 285):
                if x not in current_elements:
                    if is_valid(x, current_elements, current_sums):
                        valid_candidates.append(x)
                        
            if not valid_candidates:
                break
                
            if strategy == "random":
                x = random.choice(valid_candidates)
            elif strategy == "smallest":
                x = min(valid_candidates)
            elif strategy == "largest":
                x = max(valid_candidates)
            else: # heuristic
                # Score each candidate by how many other valid candidates it invalidates
                best_x = None
                min_conflicts = float('inf')
                # To make it faster, we might not check all pairs if list is large, but 284 is small
                for x in valid_candidates:
                    # Simulate adding x
                    new_sums = current_sums.copy()
                    for y in current_elements:
                        new_sums.add(x + y)
                    new_sums.add(2 * x)
                    
                    # Count how many other valid candidates become invalid
                    conflicts = 0
                    for z in valid_candidates:
                        if z == x:
                            continue
                        # Check if z is still valid with new_sums
                        valid_z = True
                        for y in current_elements:
                            if z + y in new_sums:
                                valid_z = False
                                break
                        if not valid_z:
                            continue
                        if 2 * z in new_sums:
                            valid_z = False
                            continue
                        if not valid_z:
                            conflicts += 1
                            
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_x = x
                        
                x = best_x
            
            # Add x
            current_elements.append(x)
            for y in current_elements[:-1]: # y in old set
                current_sums.add(x + y)
            current_sums.add(2 * x)
            
            # Time check inside loop?
            if time.time() > deadline - 1.0:
                break
                
        return current_elements

    best_set = base_elements[:]
    best_size = len(best_set)
    
    # Run multiple iterations
    while time.time() < deadline - 5.0:
        # Alternate strategies or use random choice
        r = random.random()
        if r < 0.4:
            strategy = "heuristic"
        elif r < 0.7:
            strategy = "random"
        elif r < 0.85:
            strategy = "largest"
        else:
            strategy = "smallest"
            
        s = construct_once(strategy)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        # If we found a very good set (e.g., 19 or 20), we can tighten the search
        # or just keep running to see if we can find 20 or 21 (though 20 is max for N=284 likely)
        
    return best_set
