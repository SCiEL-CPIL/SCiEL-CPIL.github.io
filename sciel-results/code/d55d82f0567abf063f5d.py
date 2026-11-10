            res = dfs_row([0], set())
            if res is None:
                return False
            row, all_diffs = res
