    def get_heuristic_solution():
        # Greedy initial
        initial_set = []
        current_mask = 0
        for num in range(N, 0, -1):
            if (current_mask & conflict[num]) == 0:
                initial_set.append(num)
                current_mask |= (1 << num)
        
        best_sum = sum(initial_set)
        
        # Local Search
        # 1. Insert
        # 2. Swap 1-1
        # 3. Swap 2-1
        # 4. Swap 1-2
        
        def try_improve(current_mask, current_sum):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                
                # 1-for-0
                for num in range(N, 0, -1):
                    if current_mask & (1 << num):
                        continue
                    if (current_mask & conflict[num]) == 0:
                        current_mask |= (1 << num)
                        current_sum += num
                        improved = True

                # 1-for-1
                if not improved:
                    for num_out in range(1, N + 1):
                        if not (current_mask & (1 << num_out)):
                            continue
                        mask_without_out = current_mask & ~(1 << num_out)
                        for num_in in range(N, num_out, -1):
                            if current_mask & (1 << num_in):
                                continue
                            if (mask_without_out & conflict[num_in]) == 0:
                                current_mask = mask_without_out | (1 << num_in)
                                current_sum = current_sum - num_out + num_in
                                improved = True
                                break
                        if improved:
                            break

                # 2-for-1
                if not improved:
                    for num_out1 in range(1, N + 1):
                        if not (current_mask & (1 << num_out1)):
                            continue
                        for num_out2 in range(num_out1 + 1, N + 1):
                            if not (current_mask & (1 << num_out2)):
                                continue
                            mask_without = current_mask & ~(1 << num_out1) & ~(1 << num_out2)
                            for num_in in range(N, max(num_out1, num_out2), -1):
                                if current_mask & (1 << num_in):
                                    continue
                                if (mask_without & conflict[num_in]) == 0:
                                    new_sum = current_sum - num_out1 - num_out2 + num_in
                                    if new_sum > current_sum:
                                        current_mask = mask_without | (1 << num_in)
                                        current_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break

                # 1-for-2
                if not improved:
                    for num_out in range(1, N + 1):
                        if not (current_mask & (1 << num_out)):
                            continue
                        mask_without_out = current_mask & ~(1 << num_out)
                        added = False
                        for num_in1 in range(N, 0, -1):
                            if current_mask & (1 << num_in1):
                                continue
                            if (mask_without_out & conflict[num_in1]) != 0:
                                continue
                            mask_with_in1 = mask_without_out | (1 << num_in1)
                            for num_in2 in range(num_in1 - 1, 0, -1):
                                if current_mask & (1 << num_in2):
                                    continue
                                if (mask_with_in1 & conflict[num_in2]) != 0:
                                    continue
                                new_sum = current_sum - num_out + num_in1 + num_in2
                                if new_sum > current_sum:
                                    current_mask = mask_with_in1 | (1 << num_in2)
                                    current_sum = new_sum
                                    improved = True
                                    added = True
                                    break
                            if added:
                                break
                        if added:
                            break
            return current_mask, current_sum

        current_mask, current_sum = try_improve(current_mask, best_sum)
        return [i for i in range(N + 1) if current_mask & (1 << i)]
