    def try_swap_and_reduce(code_list, cov, balls_arr):
        if not code_list:
            return False
        i = random.randint(0, len(code_list) - 1)
        w_old = code_list[i]
        w_new = random.randint(0, total_words - 1)
        if w_new == w_old or w_new in code_list:
            return False
        
        # Affected words
        affected = set(balls_arr[w_old])
        affected.update(balls_arr[w_new])
        
        # Simulate
        for idx in balls_arr[w_old]:
            cov[idx] -= 1
        for idx in balls_arr[w_new]:
            cov[idx] += 1
            
        if np.any(cov < 0):
            for idx in balls_arr[w_new]:
                cov[idx] -= 1
            for idx in balls_arr[w_old]:
                cov[idx] += 1
            return False
            
        # Check victims
        # A victim w_v is a candidate if it shares a ball with affected words?
        # Actually, w_v is redundant if min(cov[balls_arr[w_v]]) > 1.
        # If balls_arr[w_v] is disjoint from affected, cov[balls_arr[w_v]] is unchanged.
        # So we only check w_v where balls_arr[w_v] intersects affected.
        
        # Precomputed mapping: for each word u, which codewords have u in their ball?
        # This is inverse balls.
        # inverse_balls[u] = list of indices w such that u in balls[w]
        
        removed_victim = -1
        # We need to check codewords in code_list.
        # We can iterate through code_list and check intersection.
        # To speed up, we can precompute inverse_balls for the *current* code? No, code changes.
        # But we can precompute inverse_balls for the *space*.
        # inv_balls[u] contains all w in space such that u in balls[w].
        # Then, for a specific code_list, the relevant victims are those w in code_list 
        # such that inv_balls[u] contains w for some u in affected.
        
        # Let's just iterate through code_list for now, it's ~200 items.
        for j in range(len(code_list)):
            if j == i:
                continue
            w_v = code_list[j]
            # Check if balls_arr[w_v] intersects affected
            # We can just check min cov. If it was > 1 before, it might be redundant.
            # But we don't know "before" easily.
            # Just check min cov now.
            if np.min(cov[balls_arr[w_v]]) > 1:
                removed_victim = j
                break
                
        if removed_victim != -1:
            # Perform swap and removal
            for idx in balls_arr[code_list[removed_victim]]:
                cov[idx] -= 1
            code_list.pop(removed_victim)
            
            # Update w_old to w_new
            # Find index of w_old
            # Since we popped, index might change.
            # But w_old is still in list.
            idx_old = code_list.index(w_old)
            code_list[idx_old] = w_new
            return True
        else:
            for idx in balls_arr[w_new]:
                cov[idx] -= 1
            for idx in balls_arr[w_old]:
                cov[idx] += 1
            return False
