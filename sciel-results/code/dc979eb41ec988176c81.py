def remove_redundants_optimized(cw_list):
    total_mask = 0
    for cw in cw_list:
        total_mask |= ball_masks[cw]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(cw_list)):
            # If total_mask without cw_list[i] still covers full_mask
            if (total_mask & ~ball_masks[cw_list[i]]) == full_mask:
                total_mask &= ~ball_masks[cw_list[i]]
                cw_list.pop(i)
                changed = True
                break
    return cw_list
