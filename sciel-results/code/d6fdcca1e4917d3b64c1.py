            if len(best) < 2:
                # Can't do swap if fewer than 2 elements
                if move_type > 0.35:
                    move_type = 'change'
                else:
                    move_type = 'insert'
