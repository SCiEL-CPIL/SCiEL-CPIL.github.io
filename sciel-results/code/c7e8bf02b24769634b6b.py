                    best_new_val = old_val
                    best_err = current_err
                    
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val == old_val:
                            continue
                        
                        if mat_idx == 0:
                            U[row, col] = new_val
                        elif mat_idx == 1:
                            V[row, col] = new_val
                        else:
                            W[row, col] = new_val
                            
                        new_err = compute_error(U, V, W)
                        
                        # Revert immediately
                        if mat_idx == 0:
                            U[row, col] = old_val
                        elif mat_idx == 1:
                            V[row, col] = old_val
                        else:
                            W[row, col] = old_val
                            
                        if new_err < best_err:
                            best_err = new_err
                            best_new_val = new_val
                            
                    if best_new_val != old_val:
                        # Apply the best one
                        if mat_idx == 0:
                            U[row, col] = best_new_val
                        elif mat_idx == 1:
                            V[row, col] = best_new_val
                        else:
                            W[row, col] = best_new_val
                        current_err = best_err
