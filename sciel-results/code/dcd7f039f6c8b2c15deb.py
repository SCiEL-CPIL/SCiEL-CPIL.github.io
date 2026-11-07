        # Option 1: Include current candidate
        v = sorted_vectors[start_idx]
        c = sorted_candidates[start_idx]
        
        # Check for collision
        collision = False
        for s in current_sums:
            new_s = tuple(s[j] + v[j] for j in range(num_primes))
            if new_s in current_sums:
                collision = True
                break
                
        if not collision:
            new_sums = set(current_sums)
            for s in current_sums:
                new_s = tuple(s[j] + v[j] for j in range(num_primes))
                new_sums.add(new_s)
                
            dfs(start_idx + 1, new_sums, current_count + 1, current_list + [c])
            
        # Option 2: Skip current candidate
        dfs(start_idx + 1, current_sums, current_count, current_list)
