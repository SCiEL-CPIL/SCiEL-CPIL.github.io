import time
import random
import sys
from itertools import combinations

def construct():
    # Set a hard deadline with safety margin
    deadline = time.time() + 285.0
    
    # 1. Generate Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = q
    
    # Helper to get K4s and Alpha4s (K4 in complement)
    def get_k4s(adj, n):
        k4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n):
                    if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if ((adj[a] >> d) & 1 and (adj[b] >> d) & 1 and (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            k4s.append(mask)
        return k4s

    def get_alpha4s(adj, n):
        alpha4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if (adj[a] >> b) & 1: # If edge exists, not in independent set
                    continue
                for c in range(b + 1, n):
                    if ((adj[a] >> c) & 1 or (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if not ((adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            alpha4s.append(mask)
        return alpha4s

    # Initial lists
    k4_list = get_k4s(adj, n)
    alpha4_list = get_alpha4s(adj, n)
    
    # Function to check if a neighbor set N is valid for a new vertex
    # N is a bitmask of neighbors in the current graph (0..n-1)
    def is_valid_neighbor_set(N, n, k4_list, alpha4_list):
        # Check K4 in N
        for k4 in k4_list:
            if (N & k4) == k4:
                return False
        # Check Alpha4 in ~N
        not_N = ((1 << n) - 1) & ~N
        for a4 in alpha4_list:
            if (not_N & a4) == a4:
                return False
        return True

    # Local search to find a valid N using simulated annealing / hill climbing
    # This is the "refinement machinery" from the current program, optimized.
    def find_valid_N(n, k4_list, alpha4_list, time_limit=5.0):
        start_time = time.time()
        best_score = float('inf')
        best_N = None
        full_mask = (1 << n) - 1
        
        # Try a few random restarts
        for restart in range(30):
            if time.time() > start_time + time_limit:
                break
                
            # Initialize N randomly
            N = 0
            for i in range(n):
                if random.random() < 0.5:
                    N |= (1 << i)
            
            current_score = 0
            # Calculate initial score
            for k4 in k4_list:
                if (N & k4) == k4:
                    current_score += 1
            not_N = full_mask & ~N
            for a4 in alpha4_list:
                if (not_N & a4) == a4:
                    current_score += 1
            
            if current_score == 0:
                return N
            
            if current_score < best_score:
                best_score = current_score
                best_N = N
            
            # Hill climbing with occasional random jumps
            T = 1.0
            while time.time() - start_time < time_limit:
                if current_score == 0:
                    return N
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                delta_bit = 1 << bit
                new_N = N ^ delta_bit
                
                # Calculate change in score efficiently?
                # For now, recalculate. n is small (<=46), lists are small.
                new_score = 0
                for k4 in k4_list:
                    if (new_N & k4) == k4:
                        new_score += 1
                new_not_N = full_mask & ~new_N
                for a4 in alpha4_list:
                    if (new_not_N & a4) == a4:
                        new_score += 1
                
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    N = new_N
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_N = N
                T *= 0.995 # Cool down slowly
            
            if best_score == 0:
                return best_N

        return best_N if best_score == 0 else None

    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 15.0:
            break
        
        # Time budget per vertex
        time_per_vertex = 4.0
        if target_n > 40:
            time_per_vertex = 8.0
        
        found_N = find_valid_N(n, k4_list, alpha4_list, time_per_vertex)
        
        if found_N is None:
            # Try a few more times with different random seeds or longer time?
            # If we fail, we might be stuck.
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha4_list
        new_k4s = []
        neigh_mask = adj[new_vertex]
        # Iterate over triples in neigh_mask
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1: # Edge exists, so not independent
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Prune lists if they get too large? 
        # For n=46, max K4s is C(46,4) ~ 160k. List size might grow but stay manageable.
        # However, checking 160k items in is_valid_neighbor_set might be slow.
        # We can prune dominated sets? No, simple pruning is hard.
        # Let's just keep them. 160k * 100 restarts * 50 flips = 800M ops? Too slow.
        # Optimization: In find_valid_N, we can precompute which K4s/Alpha4s contain each vertex.
        # Then when flipping bit i, we only need to check K4s containing i.
        
        # Let's implement this optimization in find_valid_N by passing precomputed maps.
        # But for now, let's see if the simple version works within time.
        # If not, we will need to optimize.
        
        # Actually, let's optimize find_valid_N now to use vertex->constraint maps.
        # This is a structural change to the search.

    # Optimization: Precompute maps from vertex to K4s/Alpha4s containing it
    # This allows incremental score updates.
    
    # Re-initialize maps for current n
    # We need to do this after each extension or just once? 
    # The lists change. Let's rebuild them or update them.
    # Rebuilding is O(n * len(lists)).
    
    # Let's create a helper to build these maps
    def build_maps(k4_list, alpha4_list, n):
        k4_by_vertex = [[] for _ in range(n)]
        alpha4_by_vertex = [[] for _ in range(n)]
        for k4 in k4_list:
            # Find set bits
            m = k4
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                k4_by_vertex[idx].append(k4)
                m ^= lsb
        for a4 in alpha4_list:
            m = a4
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                alpha4_by_vertex[idx].append(a4)
                m ^= lsb
        return k4_by_vertex, alpha4_by_vertex

    # We should probably use the optimized search from the start or rebuild maps when n changes.
    # Let's restart the extension loop with the optimized search.
    
    # Reset to initial state to use optimized search
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    n = q
    k4_list = get_k4s(adj, n)
    alpha4_list = get_alpha4s(adj, n)
    
    def find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_limit=5.0):
        start_time = time.time()
        best_score = float('inf')
        best_N = None
        full_mask = (1 << n) - 1
        
        for restart in range(30):
            if time.time() > start_time + time_limit:
                break
                
            N = 0
            for i in range(n):
                if random.random() < 0.5:
                    N |= (1 << i)
            
            # Calculate initial score
            current_score = 0
            for k4 in k4_list:
                if (N & k4) == k4:
                    current_score += 1
            not_N = full_mask & ~N
            for a4 in alpha4_list:
                if (not_N & a4) == a4:
                    current_score += 1
            
            if current_score == 0:
                return N
            
            if current_score < best_score:
                best_score = current_score
                best_N = N
            
            T = 1.0
            while time.time() - start_time < time_limit:
                if current_score == 0:
                    return N
                
                bit = random.randint(0, n - 1)
                delta_bit = 1 << bit
                new_N = N ^ delta_bit
                
                # Incremental update
                # If we add bit (N -> new_N), we might complete K4s in N and break Alpha4s in ~N (which become N)
                # If we remove bit (N -> new_N), we might break K4s in N and complete Alpha4s in ~N (which become N)
                
                added = (new_N & delta_bit) != 0
                
                delta_score = 0
                
                if added:
                    # Check K4s containing bit that are now fully in N
                    for k4 in k4_by_vertex[bit]:
                        if (new_N & k4) == k4 and (N & k4) != k4:
                            delta_score += 1
                    # Check Alpha4s containing bit that are now fully in N (so broken in ~N)
                    # Before: bit was in ~N. After: bit is in N.
                    # Alpha4 is "bad" if it is fully in ~N.
                    # If bit was in ~N and now in N, an Alpha4 containing bit can no longer be fully in ~N.
                    # So we remove violations.
                    for a4 in alpha4_by_vertex[bit]:
                        if (N & a4) == (a4 ^ delta_bit) and (new_N & a4) != a4: 
                            # Wait, (N & a4) == (a4 ^ delta_bit) means all bits of a4 except bit were in N? No.
                            # a4 is a mask. ~N contains a4 if (not_N & a4) == a4.
                            # not_N = full_mask & ~N.
                            # Condition for a4 to be in ~N: (full_mask & ~N & a4) == a4 => (~N & a4) == a4 => (N & a4) == 0.
                            # So a4 is a violation if N has NO bits from a4.
                            # If we add bit to N, and bit is in a4, then N now has a bit from a4, so it's no longer a violation (if it was one).
                            # So we check: was it a violation before? (N & a4) == 0.
                            # Is it a violation now? (new_N & a4) == 0. Since bit in a4 and bit in new_N, (new_N & a4) != 0.
                            # So if (N & a4) == 0, we remove one violation.
                            if (N & a4) == 0:
                                delta_score -= 1
                else:
                    # Removing bit
                    # K4s: If bit was in N and in a4, and all other bits of a4 were in N, then it was a violation. Now it's not.
                    for k4 in k4_by_vertex[bit]:
                        if (N & k4) == k4 and (new_N & k4) != k4:
                            delta_score -= 1
                    # Alpha4s: If bit was in ~N and in a4, and all other bits of a4 were in ~N, then it was a violation.
                    # Now bit is in N, so a4 has a bit in N, so it's not a violation?
                    # Wait. Alpha4 violation: a4 subset of ~N.
                    # If we remove bit from N (add to ~N), we might complete an Alpha4.
                    # Condition: a4 subset of ~N_new.
                    # ~N_new = ~N_old | delta_bit.
                    # So a4 subset of ~N_old | delta_bit.
                    # If a4 contains bit, and a4 \ {bit} is subset of ~N_old, then a4 is now a violation.
                    # (new_N & a4) == 0?
                    # new_N = N ^ delta_bit (bit removed).
                    # So (new_N & a4) == 0 means a4 has no bits in new_N.
                    # Since bit is not in new_N, this is equivalent to (N & (a4 ^ delta_bit)) == 0?
                    # Yes, if bit is in a4.
                    for a4 in alpha4_by_vertex[bit]:
                        if (new_N & a4) == 0 and (N & a4) != 0:
                            # It was not a violation before (N had a bit from a4, which must be bit since new_N doesn't have it? No, N had bit).
                            # If N had bit, and new_N doesn't, and new_N has no other bits from a4, then N had exactly bit from a4.
                            # So (N & a4) == delta_bit.
                            if (N & a4) == delta_bit:
                                delta_score += 1
                
                new_score = current_score + delta_score
                
                if new_score < 0: new_score = 0 # Safety
                
                if new_score < current_score or random.random() < pow(2.718, -(new_score - current_score) / T):
                    N = new_N
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_N = N
                T *= 0.995
            
            if best_score == 0:
                return best_N

        return best_N if best_score == 0 else None

    # Extension loop with optimized search
    k4_by_vertex, alpha4_by_vertex = build_maps(k4_list, alpha4_list, n)
    
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 15.0:
            break
        
        time_per_vertex = 4.0
        if target_n > 40:
            time_per_vertex = 8.0
        
        found_N = find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_per_vertex)
        
        if found_N is None:
            break
        
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update lists and maps
        new_k4s = []
        neigh_mask = adj[new_vertex]
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Update maps
        # We can just rebuild for simplicity, as n is small and lists are not huge yet.
        # Or update incrementally.
        # Rebuild is O(n * total_constraints).
        k4_by_vertex, alpha4_by_vertex = build_maps(k4_list, alpha4_list, n)

    # Convert adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    
    return n, edges
