"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    
    N = 4096
    
    def build_greedy(start_from=1):
        """Greedy: add numbers in order, maintaining set of triple sums."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, N + 1):
            if time.time() > DEADLINE - 20:
                break
            # Check if x can be added: all new triple sums involving x must be unique
            # and not conflict with existing triple sums
            new_sums = []
            n = len(selected)
            ok = True
            # Triples with one x: x + a_i + a_j for i <= j
            for i in range(n):
                for j in range(i, n):
                    s = x + selected[i] + selected[j]
                    new_sums.append(s)
            # Triples with two x's: 2*x + a_i
            for i in range(n):
                s = 2 * x + selected[i]
                new_sums.append(s)
            # Triple with three x's
            s = 3 * x
            new_sums.append(s)
            
            # Check for duplicates among new sums and against existing
            for s in new_sums:
                if s in triple_sums or new_sums.count(s) > 1:
                    ok = False
                    break
            # More efficient duplicate check
            if ok:
                seen_new = set()
                for s in new_sums:
                    if s in seen_new or s in triple_sums:
                        ok = False
                        break
                    seen_new.add(s)
            
            if ok:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected
    
    # Phase 1: Standard greedy
    result = build_greedy(1)
    
    # Phase 2: Try to improve by removing one element and adding a better one
    if time.time() < DEADLINE - 15:
        current_size = len(result)
        for _ in range(5):  # Limited improvement attempts
            if time.time() > DEADLINE - 15:
                break
            best_swap = None
            best_size = current_size
            for i in range(len(result)):
                if time.time() > DEADLINE - 15:
                    break
                # Try removing result[i] and rebuilding
                temp = [x for x in result if x != result[i]]
                # Rebuild greedy from temp
                triple_sums = set()
                n = len(temp)
                for ii in range(n):
                    for jj in range(ii, n):
                        for kk in range(jj, n):
                            triple_sums.add(temp[ii] + temp[jj] + temp[kk])
                
                # Try to add new numbers
                improved = list(temp)
                for x in range(1, N + 1):
                    if x in improved:
                        continue
                    new_sums = []
                    m = len(improved)
                    for ii in range(m):
                        for jj in range(ii, m):
                            new_sums.append(x + improved[ii] + improved[jj])
                    for ii in range(m):
                        new_sums.append(2 * x + improved[ii])
                    new_sums.append(3 * x)
                    
                    seen_new = set()
                    ok = True
                    for s in new_sums:
                        if s in seen_new or s in triple_sums:
                            ok = False
                            break
                        seen_new.add(s)
                    
                    if ok:
                        improved.append(x)
                        triple_sums.update(new_sums)
                    elif time.time() > DEADLINE - 15:
                        break
                
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_swap = improved
            
            if best_swap is not None:
                result = best_swap
                current_size = best_size
    
    return result
# EVOLVE-BLOCK-END
