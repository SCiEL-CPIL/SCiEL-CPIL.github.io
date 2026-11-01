            else:
                # Revert
                selected[:] = old_selected
                cover_count[:] = old_cover_count
                uncovered[:] = (cover_count == 0)
