import math

def construct():
    N = math.factorial(12)
    
    best_max = float('inf')
    best_solution = None
    
    def update_best(factors):
        nonlocal best_max, best_solution
        if len(set(factors)) != len(factors):
            return
        if min(factors) <= 12:
            return
        if max(factors) < best_max:
            best_max = max(factors)
            best_solution = list(factors)
            
    for k in range(1, 13):
        initial_products = [1] * k
        
        # 11 is in bin 0
        initial_products[0] *= 11
        
        def solve_7_and_5(has_7_in_0, has_7_in_i, i_val):
            nonlocal initial_products
            
            # Reset initial products for 7 and 5
            prods = [1] * k
            prods[0] *= 11 # 11 is always in 0
            
            if has_7_in_0:
                prods[0] *= 7
            if has_7_in_i:
                if i_val < k:
                    prods[i_val] *= 7
            
            # Assign 5s
            # Case 1: 5s in same bin j
            for j in range(k):
                p5 = [1] * k
                p5[j] = 25
                base_prods = [prods[i] * p5[i] for i in range(k)]
                
                # Distribute 10 twos and 5 threes
                def dfs_dist(idx, rem_2, rem_3, curr_prods):
                    nonlocal best_max, best_solution
                    
                    if idx == k:
                        if rem_2 == 0 and rem_3 == 0:
                            if len(set(curr_prods)) == k and min(curr_prods) > 12:
                                if max(curr_prods) < best_max:
                                    best_max = max(curr_prods)
                                    best_solution = list(curr_prods)
                        return

                    if any(p > best_max for p in curr_prods):
                        return

                    # Check distinctness with previous bins
                    prev_prods = curr_prods[:idx]
                    
                    for c2 in range(rem_2 + 1):
                        for c3 in range(rem_3 + 1):
                            new_prod = curr_prods[idx] * (2 ** c2) * (3 ** c3)
                            if new_prod > best_max:
                                continue
                            if new_prod in prev_prods:
                                continue
                            
                            temp_prods = list(curr_prods)
                            temp_prods[idx] = new_prod
                            dfs_dist(idx + 1, rem_2 - c2, rem_3 - c3, temp_prods)

                dfs_dist(0, 10, 5, base_prods)

            # Case 2: 5s in different bins j, m
            for j in range(k):
                for m in range(j + 1, k):
                    p5 = [1] * k
                    p5[j] = 5
                    p5[m] = 5
                    base_prods = [prods[i] * p5[i] for i in range(k)]
                    
                    def dfs_dist(idx, rem_2, rem_3, curr_prods):
                        nonlocal best_max, best_solution
                        
                        if idx == k:
                            if rem_2 == 0 and rem_3 == 0:
                                if len(set(curr_prods)) == k and min(curr_prods) > 12:
                                    if max(curr_prods) < best_max:
                                        best_max = max(curr_prods)
                                        best_solution = list(curr_prods)
                            return

                        if any(p > best_max for p in curr_prods):
                            return

                        prev_prods = curr_prods[:idx]
                        
                        for c2 in range(rem_2 + 1):
                            for c3 in range(rem_3 + 1):
                                new_prod = curr_prods[idx] * (2 ** c2) * (3 ** c3)
                                if new_prod > best_max:
                                    continue
                                if new_prod in prev_prods:
                                    continue
                                
                                temp_prods = list(curr_prods)
                                temp_prods[idx] = new_prod
                                dfs_dist(idx + 1, rem_2 - c2, rem_3 - c3, temp_prods)

                    dfs_dist(0, 10, 5, base_prods)

        # 7 in bin 0
        solve_7_and_5(True, False, -1)
        
        # 7 in bin i, i > 0
        for i in range(1, k):
            solve_7_and_5(False, True, i)

    if best_solution is None:
        return [N]
        
    return best_solution
