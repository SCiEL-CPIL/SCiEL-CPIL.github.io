                    if _is_valid_add(selected_bool, x_in):
                         selected_bool[x_in] = True
                         new_score = _count_3aps(selected_bool)
                         if new_score > current_score:
                             current_score = new_score
                             improved = True
                             break
                         else:
                             selected_bool[x_in] = False
                             selected_bool[x_out] = True
                    else:
                         selected_bool[x_out] = True
