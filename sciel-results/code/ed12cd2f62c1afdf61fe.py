import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n = 16
    points = list(range(n))
    
    # Precompute all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    
    # Precompute masks for triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = mask
        
    # Precompute conflicts
    # conflict[i] is a set of indices j such that triples i and j share >= 2 points
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)
            
    conflicts = [set() for _ in range(num_triples)]
    for p, idxs in pair_to_triples.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                conflicts[idxs[i]].add(idxs[j])
                conflicts[idxs[j]].add(idx[i])

    # Helper: Calculate independence number (alpha)
    # We want to minimize alpha.
    # Alpha is the size of the largest subset of points containing no triple.
    
    def calc_alpha(system_indices):
        if not system_indices:
            return n
            
        sys_masks = [triple_masks[i] for i in system_indices]
        best = 0
        
        # Sort points by degree in the system to prune earlier
        degrees = [0] * n
        for idx in system_indices:
            t = all_triples[idx]
            degrees[t[0]] += 1
            degrees[t[1]] += 1
            degrees[t[2]] += 1
            
        ordered_points = sorted(range(n), key=lambda p: degrees[p])
        
        def is_independent(mask):
            for m in sys_masks:
                if (mask & m) == m:
                    return False
            return True
            
        def backtrack(pos, current_mask, current_count):
            nonlocal best
            if pos == n:
                if current_count > best:
                    best = current_count
                return
                
            if current_count + (n - pos) <= best:
                return
                
            p = ordered_points[pos]
            
            # Option 1: Include p
            new_mask = current_mask | (1 << p)
            if is_independent(new_mask):
                backtrack(pos + 1, new_mask, current_count + 1)
                
            # Option 2: Exclude p
            backtrack(pos + 1, current_mask, current_count)
            
        backtrack(0, 0, 0)
        return best

    def has_independent_set_of_size(system_indices, size):
        # Returns True if there is an independent set of size 'size'
        if not system_indices:
            return True
            
        sys_masks = [triple_masks[i] for i in system_indices]
        
        # Sort points by degree to prune earlier
        degrees = [0] * n
        for idx in system_indices:
            t = all_triples[idx]
            degrees[t[0]] += 1
            degrees[t[1]] += 1
            degrees[t[2]] += 1
            
        ordered_points = sorted(range(n), key=lambda p: degrees[p])
        
        def backtrack(pos, current_mask, current_count):
            if current_count == size:
                return True
            if pos == n:
                return False
            if (n - pos) < (size - current_count):
                return False
                
            p = ordered_points[pos]
            
            # Option 1: Include p
            new_mask = current_mask | (1 << p)
            ok = True
            for m in sys_masks:
                if (new_mask & m) == m:
                    ok = False
                    break
            if ok:
                if backtrack(pos + 1, new_mask, current_count + 1):
                    return True
                    
            # Option 2: Exclude p
            if backtrack(pos + 1, current_mask, current_count):
                return True
                
            return False
            
        return backtrack(0, 0, 0)

    def fast_alpha_check(system_indices):
        # Returns 8 if alpha >= 8, 7 if alpha <= 7
        if has_independent_set_of_size(system_indices, 8):
            return 8
        else:
            return 7

    def random_maximal_system():
        available = set(range(num_triples))
        system = []
        while available:
            t_idx = random.choice(list(available))
            system.append(t_idx)
            to_remove = {t_idx} | conflicts[t_idx]
            available.difference_update(to_remove)
        return system

    def local_search_step(system_indices):
        if not system_indices:
            return system_indices
            
        remove_idx = random.randint(0, len(system_indices) - 1)
        t_remove = system_indices[remove_idx]
        
        for _ in range(20):
            t_add = random.randint(0, num_triples - 1)
            if t_add == t_remove:
                continue
            if t_add in system_indices:
                continue
                
            conflict = False
            for t_other in system_indices:
                if t_other == t_remove:
                    continue
                if t_add in conflicts[t_other]:
                    conflict = True
                    break
            if not conflict:
                new_system = system_indices[:]
                new_system[remove_idx] = t_add
                return new_system
                
        return system_indices

    best_fast_alpha = n + 1
    best_system = []
    no_improve_steps = 0
    
    while time.time() < deadline - 5:
        if no_improve_steps > 50 or not best_system:
            current_system = random_maximal_system()
            no_improve_steps = 0
        else:
            current_system = local_search_step(best_system)
            
        fast_alpha = fast_alpha_check(current_system)
        
        if fast_alpha < best_fast_alpha:
            best_fast_alpha = fast_alpha
            best_system = current_system[:]
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            
    # Final polish with exact alpha
    time_left = deadline - time.time()
    if time_left > 5 and best_system:
        current_alpha = calc_alpha(best_system)
        
        while time.time() < deadline - 1:
            new_system = local_search_step(best_system)
            if new_system == best_system:
                # Try a few more random restarts if stuck?
                # Or just break to save time for other candidates?
                # Let's try a few more steps
                if random.random() > 0.5:
                    continue
                break
                
            new_alpha = calc_alpha(new_system)
            
            if new_alpha < current_alpha:
                best_system = new_system
                current_alpha = new_alpha
            else:
                # Accept with small probability to escape local minima?
                # Simulated annealing style?
                # For now, simple hill climbing.
                pass

    result = [list(all_triples[i]) for i in best_system]
    return result

if __name__ == "__main__":
    t = construct()
    print(f"Constructed {len(t)} triples")
