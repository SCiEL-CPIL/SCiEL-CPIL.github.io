if set_type == 0: # 4-set
    if old_color == 0:
        # Red -> Blue
        if old_r == 4:
            delta_penalty -= 1
    else:
        # Blue -> Red
        if new_r == 4: # new_r = old_r + 1
            delta_penalty += 1
else: # 5-set
    if old_color == 1:
        # Blue -> Red
        if old_b == 10:
            delta_penalty -= 1
    else:
        # Red -> Blue
        if new_b == 10: # new_b = old_b + 1
            delta_penalty += 1
