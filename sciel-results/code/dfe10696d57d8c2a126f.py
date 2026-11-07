<<<<<<< SEARCH
                if delta > 0:
                    # Accept the swap
                    family[i] = new_mask
                    family_set.remove(old_mask)
                    family_set.add(new_mask)
                    outside.remove(old_mask)
                    outside.append(new_mask)
                    current_score += delta
                    improved = True
=======
                if delta > 0:
                    # Accept the swap
                    family[i] = new_mask
                    family_set.remove(old_mask)
                    family_set.add(new_mask)
                    outside.remove(new_mask)
                    outside.append(old_mask)
                    current_score += delta
                    improved = True
>>>>>>> REPLACE
