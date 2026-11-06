import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def build_paley(p):
        """
        Build Paley tournament on p vertices.
        Returns orientation list.
        """
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return orientation

    def build_in_nb(n, orientation):
        """
        Build in-neighborhood bitsets.
        in_nb[i] is a bitmask of vertices j such that j -> i.
        """
        in_nb = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    # j's in-neighbor is i
                    in_nb[j] |= (1 << i)
                else:
                    # j -> i
                    # i's in-neighbor is j
                    in_nb[i] |= (1 << j)
        return in_nb

    def is_valid(n, in_nb):
        """
        Check if every 4-set has a common dominator.
        Dominator of {a,b,c,d} is v such that v -> a, v -> b, v -> c, v -> d.
        This is equivalent to v being in In(a) ∩ In(b) ∩ In(c) ∩ In(d).
        """
        for a in range(n - 3):
            if elapsed() > DEADLINE:
                return False
            for b in range(a + 1, n - 2):
                ab = in_nb[a] & in_nb[b]
                if ab == 0:
                    continue
                for c in range(b + 1, n - 1):
                    abc = ab & in_nb[c]
                    if abc == 0:
                        continue
                    for d in range(c + 1, n):
                        if (abc & in_nb[d]) == 0:
                            return False
        return True

    def try_reduce(n, orientation):
        """
        Try to reduce the tournament by removing non-critical vertices.
        A vertex v is critical if it is the unique dominator of some 4-set.
        Optimized version with better criticality tracking.
        """
        current_n = n
        in_nb = build_in_nb(current_n, orientation)
        active = set(range(n))
        
        # Track which vertices are critical (unique dominators of some 4-set)
        critical_count = [0] * n  # Number of 4-sets where v is the unique dominator
        
        while True:
            if elapsed() > DEADLINE:
                break
            
            # Reset critical counts
            critical_count = [0] * current_n
            
            # Iterate over all 4-sets of active vertices
            active_list = sorted(active)
            m = len(active_list)
            
            for a_idx in range(m - 3):
                if elapsed() > DEADLINE:
                    break
                a = active_list[a_idx]
                for b_idx in range(a_idx + 1, m - 2):
                    b = active_list[b_idx]
                    ab = in_nb[a] & in_nb[b]
                    if ab == 0:
                        continue
                    for c_idx in range(b_idx + 1, m - 1):
                        c = active_list[c_idx]
                        abc = ab & in_nb[c]
                        if abc == 0:
                            continue
                        for d_idx in range(c_idx + 1, m):
                            d = active_list[d_idx]
                            doms = abc & in_nb[d]
                            if doms == 0:
                                # Should not happen if we started valid
                                return None
                            
                            # Count bits in doms
                            cnt = doms.bit_count()
                            
                            if cnt == 1:
                                # Unique dominator
                                v = doms.bit_length() - 1
                                if v in active:
                                    critical_count[v] += 1
            
            # Find a non-critical vertex to remove
            removed = -1
            for v in active:
                if critical_count[v] == 0:
                    removed = v
                    break
            
            if removed == -1:
                # All vertices are critical, cannot reduce
                break
            
            # Remove 'removed' from active
            active.remove(removed)
            current_n -= 1
            
            # If we've reduced enough, stop
            if current_n <= 35:
                break
            
        # Rebuild orientation for the active set
        if not active:
            return None
            
        active_list = sorted(active)
        new_orientation = []
        
        for i_new in range(len(active_list)):
            old_i = active_list[i_new]
            for j_new in range(i_new + 1, len(active_list)):
                old_j = active_list[j_new]
                # Check direction old_i -> old_j
                # If in_nb[old_j] has bit old_i, then old_i -> old_j (bit 0)
                # Else old_j -> old_i (bit 1)
                if in_nb[old_j] & (1 << old_i):
                    new_orientation.append(0)
                else:
                    new_orientation.append(1)
                    
        return len(active_list), new_orientation

    # Candidate primes p ≡ 3 mod 4
    # We want to minimize n.
    # Try smaller primes first.
    candidates = [31, 43, 59, 67, 79, 83]
    
    best_n = 91
    best_orientation = None
    
    for p in candidates:
        if elapsed() > DEADLINE - 10:
            break
            
        if p > best_n:
            break
            
        orientation = build_paley(p)
        in_nb = build_in_nb(p, orientation)
        
        # Check validity
        if not is_valid(p, in_nb):
            continue
            
        # It is valid. Try to reduce.
        reduced = try_reduce(p, orientation)
        
        if reduced:
            n_red, ori_red = reduced
            if n_red < best_n:
                best_n = n_red
                best_orientation = ori_red
        else:
            # Could not reduce or reduction failed, use original
            if p < best_n:
                best_n = p
                best_orientation = orientation
                
        # If we found a very small n, we might stop?
        # Lower bound is 31. If we get 31, stop.
        if best_n <= 31:
            break

    if best_orientation is None:
        # Fallback to 67 if something went wrong
        p = 67
        orientation = build_paley(p)
        return {'n': p, 'orientation': orientation}
        
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
