    def try_swap_and_optimize(code_set, coverage):
        c_old_list = list(code_set)
        if not c_old_list:
            return None, None, False
            
        c_old = c_old_list[np.random.randint(len(c_old_list))]
        c_old_idx = word_to_idx[c_old]
        
        c_new = all_words[np.random.randint(N)]
        if c_new == c_old or c_new in code_set:
            return None, None, False
            
        c_new_idx = word_to_idx[c_new]
        
        # Check validity of removal
        for n_idx in neigh_indices[c_old_idx]:
            if coverage[n_idx] == 1:
                return None, None, False
                
        # Create new set and coverage
        new_set = set(code_set)
        new_set.remove(c_old)
        new_set.add(c_new)
        
        new_cov = coverage.copy()
        for n_idx in neigh_indices[c_old_idx]:
            new_cov[n_idx] -= 1
        for n_idx in neigh_indices[c_new_idx]:
            new_cov[n_idx] += 1
            
        return new_set, new_cov, True
