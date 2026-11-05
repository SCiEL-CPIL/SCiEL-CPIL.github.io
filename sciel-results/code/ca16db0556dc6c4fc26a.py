            if valid:
                # Add new_block
                new_sids = get_subsets_of_block(new_block)
                for sid in new_sids:
                    cover_count[sid] += 1
                blocks[i] = new_block
            else:
                # Rollback
                for sid in old_sids:
                    cover_count[sid] += 1
                # Try a few more random blocks
                for _ in range(5):
                    new_block = random.sample(range(v), k)
                    new_sids = get_subsets_of_block(new_block)
                    
                    # Check validity against current state (old_block removed)
                    # Wait, we already rolled back. So we need to remove old_block again?
                    # No, we can just check if new_block covers all subsets that old_block uniquely covered.
                    # A swap is valid if: for all s in old_sids, cover_count[s] - 1 + (1 if s in new_sids else 0) > 0
                    # i.e., cover_count[s] + (1 if s in new_sids else 0) > 1
                    # i.e., if cover_count[s] == 1, then s MUST be in new_sids.
                    
                    valid2 = True
                    for sid in old_sids:
                        if cover_count[sid] == 1:
                            # This subset is only covered by old_block
                            # So new_block MUST cover it
                            if sid not in new_sids:
                                valid2 = False
                                break
                    
                    if valid2:
                        # Perform swap
                        for sid in old_sids:
                            cover_count[sid] -= 1
                        for sid in new_sids:
                            cover_count[sid] += 1
                        blocks[i] = new_block
                        break
