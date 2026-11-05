<<<<<<< SEARCH
    def local_search(S_list):
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try to remove single points (shrink)
            new_S = shrink(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try to remove pairs of points
            new_S = shrink_pairs(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try swaps
            S_bits = 0
            for idx in S_list:
                S_bits |= (1 << idx)
            
            # Pick a random point in S and a random point not in S
            if len(S_list) == 0:
                break
                
            p = random.choice(S_list)
            # Find a q not in S
            q = random.randint(0, N2 - 1)
            while (S_bits >> q) & 1:
                q = random.randint(0, N2 - 1)
                
            # Check if S \ {p} U {q} is valid
            S_minus_p_bits = S_bits ^ (1 << p)
            S_minus_p_list = [x for x in S_list if x != p]
            
            valid_swap = True
            for a in S_minus_p_list:
                if (iso_with[q][a] & S_minus_p_bits):
                    valid_swap = False
                    break
                    
            if valid_swap:
                new_S = S_minus_p_list + [q]
                # Try to shrink the new set
                new_S = shrink(new_S)
                new_S = shrink_pairs(new_S)
                if len(new_S) < len(S_list):
                    S_list = new_S
                    improved = True
                    break
        return S_list
=======
    def local_search(S_list):
        S_list = list(S_list)
        improved = True
        stall_count = 0
        while improved and time.time() < deadline - 10:
            improved = False
            stall_count += 1
            
            # Try to remove single points (shrink)
            new_S = shrink(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                stall_count = 0
                continue
            
            # Try to remove pairs of points
            new_S = shrink_pairs(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                stall_count = 0
                continue
            
            # Try swaps - multiple attempts
            S_bits = 0
            for idx in S_list:
                S_bits |= (1 << idx)
            
            # Pick a random point in S and a random point not in S
            if len(S_list) == 0:
                break
                
            # Try multiple swap candidates
            for _ in range(min(5, stall_count * 2)):
                p = random.choice(S_list)
                # Find a q not in S
                q = random.randint(0, N2 - 1)
                while (S_bits >> q) & 1:
                    q = random.randint(0, N2 - 1)
                    
                # Check if S \ {p} U {q} is valid
                S_minus_p_bits = S_bits ^ (1 << p)
                S_minus_p_list = [x for x in S_list if x != p]
                
                valid_swap = True
                for a in S_minus_p_list:
                    if (iso_with[q][a] & S_minus_p_bits):
                        valid_swap = False
                        break
                        
                if valid_swap:
                    new_S = S_minus_p_list + [q]
                    # Try to shrink the new set
                    new_S = shrink(new_S)
                    new_S = shrink_pairs(new_S)
                    if len(new_S) < len(S_list):
                        S_list = new_S
                        improved = True
                        stall_count = 0
                        break
        return S_list
>>>>>>> REPLACE

