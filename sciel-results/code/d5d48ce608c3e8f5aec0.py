        # Local improvement
        unaccepted = [i for i in range(num_points) if i not in accepted_set]
        improved = True
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            i = 0
            while i < len(unaccepted):
                if time.monotonic() + safety_margin >= deadline:
                    break
                idx = unaccepted[i]
                if not forbidden[idx]:
                    # Add point
                    accepted_indices.append(idx)
                    accepted_set.add(idx)
                    unaccepted.pop(i)
                    improved = True
                    
                    # Update forbidden
                    # ... (same update logic)
                    
                    # Do not increment i, check next element which shifted to i
                else:
                    i += 1
