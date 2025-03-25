import heapq


def a_star(start_state, goal_pole=2):
    def is_goal_state(state):
        return state[goal_pole] == [4, 3, 2, 1]

    def heuristic(state):
        """
        state is a list of lists, where each sublist represents a peg's disks.
        The target peg is assumed to be the last peg (index 3).
        """
        target_peg = state[goal_pole]
        final_state = [4, 3, 2, 1]  # Correct order from bottom to top
        current_length = 0

        # Determine how many disks are correctly placed starting from the bottom
        for i in range(len(target_peg)):
            if i < len(final_state) and target_peg[i] == final_state[i]:
                current_length += 1
            else:
                break

        # Calculate the sum of 2^(d-1) for each disk not yet correctly placed
        sum_heuristic = 0
        for d in final_state[current_length:]:
            sum_heuristic += 2 ** (d - 1)

        return sum_heuristic

    def get_neighbors(state):
        neighbors = []
        for i in range(3):  # From peg i
            if not state[i]:
                continue  # Skip empty pegs
            for j in range(3):  # To peg j
                if i == j:
                    continue  # Can't move to the same peg

                new_state = [list(peg) for peg in state]  # Deep copy
                ring = new_state[i].pop()

                if not new_state[j] or new_state[j][-1] > ring:  # Valid move (smallest on top)
                    new_state[j].append(ring)  # Place ring on top of destination peg
                    neighbors.append([new_state, (i, j)])
        return neighbors

    open_states = []
    heapq.heappush(open_states, (0, start_state, []))  # (f-score, state, path)
    closed_state = list()

    while open_states:
        f, current_state, path = heapq.heappop(open_states)

        if is_goal_state(current_state):
            return path  # Return the sequence of moves

        for state in closed_state:
            if state == current_state:
                continue

        closed_state.append(current_state)

        for neighbor, action in get_neighbors(current_state):
            if neighbor not in closed_state:
                new_path = path + [action]
                g = len(new_path)
                h = heuristic(neighbor)
                heapq.heappush(open_states, (g + h, neighbor, new_path))
    return None
