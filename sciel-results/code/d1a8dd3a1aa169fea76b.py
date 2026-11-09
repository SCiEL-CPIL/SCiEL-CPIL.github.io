import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    mask = (1 << 201) - 1
    
    def can_add(x, reachable_bits):
        """Check if x can be added without creating sum 200."""
        return not ((reachable_bits >> (200 - x)) & 1)
    
    def add_element(x, reachable_bits):
        """Add element x and update reachable bits."""
        new_bits = (reachable_bits << x) & mask
        return reachable_bits | new_bits
    
    def greedy_with_order(order, reachable_bits=1):
        """Greedy subset selection using bitset, processing elements in given order."""
        selected = []
        for x in order:
            if can_add(x, reachable_bits):
                selected.append(x)
                reachable_bits = add_element(x, reachable_bits)
        return selected, reachable_bits
    
    def try_add_elements(selected, reachable_bits, candidates):
        """Try to add elements from candidates to selected, updating reachable."""
        improved = False
        selected_set = set(selected)
        for x in candidates:
            if x in selected_set:
                continue
            if not can_add(x, reachable_bits):
                continue
            selected.append(x)
            selected_set.add(x)
            reachable_bits = add_element(x, reachable_bits)
            improved = True
        return selected, reachable_bits, improved
    
    def greedy_min_impact(remaining, reachable_bits, selected):
        """At each step, pick the element whose addition causes the least growth in reachable sums."""
        while remaining:
            if time.time() > deadline:
                break
            
            best_x = None
            best_impact = float('inf')
            
            for x in remaining:
                if not can_add(x, reachable_bits):
                    continue
                
                new_reach = add_element(x, reachable_bits)
                # Impact = number of new bits set (new reachable sums)
                new_bits = new_reach & ~reachable_bits
                impact = bin(new_bits).count('1')
                
                # Tie-break: prefer larger elements
                if best_x is None or (impact, -x) < (best_impact, -best_x):
                    best_impact = impact
                    best_x = x
            
            if best_x is None:
                break
            
            selected.append(best_x)
            reachable_bits = add_element(best_x, reachable_bits)
            remaining.remove(best_x)
        
        return selected, reachable_bits
    
    def greedy_min_degree(remaining, reachable_bits, selected):
        """At each step, pick the element that blocks the fewest other remaining elements."""
        while remaining:
            if time.time() > deadline:
                break
            
            best_x = None
            best_score = float('inf')
            
            for x in remaining:
                if not can_add(x, reachable_bits):
                    continue
                
                # If we add x, compute how many other remaining elements get blocked
                new_reach = add_element(x, reachable_bits)
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    if not can_add(y, new_reach):
                        blocked += 1
                
                # Score: prefer elements that block fewer others
                # Tie-break: prefer larger elements
                score = (blocked, -x)
                
                if best_x is None or score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            selected.append(best_x)
            reachable_bits = add_element(best_x, reachable_bits)
            remaining.remove(best_x)
        
        return selected, reachable_bits
    
    # Generate diverse orderings
    orders = []
    orders.append(list(range(100, 0, -1)))
    orders.append(list(range(1, 101)))
    
    # Medium to extremes
    med_order = []
    for d in range(50):
        if 50 - d >= 1:
            med_order.append(50 - d)
        if 51 + d <= 100:
            med_order.append(51 + d)
    orders.append(med_order)
    
    # Random orderings
    for _ in range(15):
        order = list(range(1, 101))
        random.shuffle(order)
        orders.append(order)
    
    # Alternating large and small
    alt_order = []
    for i in range(50):
        alt_order.append(100 - i)
        if i + 1 <= 50:
            alt_order.append(i + 1)
    orders.append(alt_order)
    
    # Sorted by value mod some number to create diversity
    for mod in [3, 4, 5]:
        order = sorted(range(1, 101), key=lambda x: (x % mod, x))
        orders.append(order)
    
    best_set = []
    best_size = 0
    best_reachable = 1
    
    for order in orders:
        if time.time() > deadline:
            break
        
        selected, reachable = greedy_with_order(order)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            best_reachable = reachable
        
        # Local search: try to add more elements
        selected_set = set(selected)
        candidates = [x for x in range(1, 101) if x not in selected_set]
        
        for add_order in [candidates, list(reversed(candidates))]:
            if time.time() > deadline:
                break
            new_selected = selected[:]
            new_reachable = reachable
            new_selected, new_reachable, improved = try_add_elements(
                new_selected, new_reachable, add_order
            )
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = new_selected[:]
                best_reachable = new_reachable
    
    # Try minimum-impact greedy from scratch
    if time.time() < deadline - 15:
        remaining = set(range(1, 101))
        sel, reach = greedy_min_impact(remaining, 1, [])
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel[:]
            best_reachable = reach
    
    # Try minimum-degree greedy from scratch
    if time.time() < deadline - 15:
        remaining = set(range(1, 101))
        sel, reach = greedy_min_degree(remaining, 1, [])
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel[:]
            best_reachable = reach
    
    # Additional local search on the best set found
    if time.time() < deadline - 5:
        for round_num in range(20):
            if time.time() > deadline:
                break
            
            selected_set = set(best_set)
            candidates = [x for x in range(1, 101) if x not in selected_set]
            
            if not candidates:
                break
            
            random.shuffle(candidates)
            new_selected = best_set[:]
            new_reachable = best_reachable
            new_selected, new_reachable, improved = try_add_elements(
                new_selected, new_reachable, candidates
            )
            
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = new_selected[:]
                best_reachable = new_reachable
            else:
                break
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
