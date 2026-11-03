     if not found:
        # No overlap > 0, pick the lexicographically smallest unvisited permutation
        for i in range(N):
            if not visited[i]:
                visited[i] = True
                next_perm = perms[i]
                current_word.extend(next_perm)
                current_idx = i
                found = True
                break
     