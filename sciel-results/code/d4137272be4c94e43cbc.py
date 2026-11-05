    def find_set(k, start_idx, current_set, current_sums, min_s, max_s):
        # ...
        if min_s + v > max_s:
             # No collision
             new_min = min_s
             new_max = max_s + v
             # ...
        else:
             # Check collision
             # ...
             new_min = min_s
             new_max = max_s + v
