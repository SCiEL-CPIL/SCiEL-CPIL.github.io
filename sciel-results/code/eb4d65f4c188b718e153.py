                if compatible:
                    # Perform swap
                    selected.remove(s)
                    selected.add(n)
                    improved = True
                    best_value = score(selected)
                    if best_value > best_value: # Redundant but safe
                        pass
                    # Update best if better
                    if score(selected) > best_value:
                        best_value = score(selected)
                        best_subset = set(selected)
