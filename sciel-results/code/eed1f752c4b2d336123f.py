<<<<<<< SEARCH
        current_block.sort()
        blocks.append(current_block)
        
        if time.time() > deadline:
            break
=======
        current_block.sort()
        if len(current_block) == k:
            blocks.append(current_block)
        else:
            # If we couldn't complete the block (shouldn't happen with valid input),
            # we discard it to ensure validity.
            pass
        
        if time.time() > deadline:
            break
>>>>>>> REPLACE
