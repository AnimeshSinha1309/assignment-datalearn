x = 0.88
call_reward = 11

locale = [(i, j) for i in range(3) for j in range(3)]
states = [(agent, target, call) for agent in range(9) for target in range(9)for call in [0, 1]]
actions = ['S', 'L', 'R', 'U', 'D']

def namify(locale):
    return locale[0] + locale[1] * 3

def statify(locale):
    return (locale % 3, locale // 3)

def move(locale, direction):
    locale = statify(locale)
    if direction == 'S':
        return namify(locale)
    if direction == 'L':
        return namify((max(locale[0] - 1, 0), locale[1]))
    if direction == 'R':
        return namify((min(locale[0] + 1, 2), locale[1]))
    if direction == 'U':
        return namify((locale[0], max(locale[1] - 1, 0)))
    if direction == 'D':
        return namify((locale[0], min(locale[1] + 1, 2)))

def get_transitions():
    transitions = {}
    y = 0.85
    for state_idx, state_val in enumerate(states):
        agent, target, call = state_val

        end_agent_actlist = [
            {move(agent, 'S'): 1},
            {move(agent, 'L'): x, move(agent, 'R'): 1 - x},
            {move(agent, 'R'): x, move(agent, 'L'): 1 - x},
            {move(agent, 'U'): x, move(agent, 'D'): 1 - x},
            {move(agent, 'D'): x, move(agent, 'U'): 1 - x},
        ]
        end_target = {target: 0.4}
        for target_move in actions[1:]:
            pos = move(target, target_move)
            end_target[pos] = end_target[pos] + 0.15 if target == pos else 0.15
        end_call = {1: 0.4, 0: 0.6} if call == 0 or target == agent else {0: 0.2, 1: 0.8}

        for action in range(5):
            for agent_state, agent_prob in end_agent_actlist[action].items():
                for target_state, target_prob in end_target.items():
                    for call_state, call_prob in end_call.items():
                        new_state = agent_state * 18 + target_state * 2 + call_state
                        prob = agent_prob * target_prob * call_prob
                        tx_idx = (state_idx, action, new_state)
                        transitions[tx_idx] = transitions[tx_idx] + prob if tx_idx in transitions.keys() else prob
    return transitions


def get_observations():
    observations = {}
    for state_idx, state in enumerate(states):
        agent, target, call = state
        if agent == target:
            observations[state_idx] = "o1"
        elif move(agent, 'R') == target:
            observations[state_idx] = "o2"
        elif move(agent, 'D') == target:
            observations[state_idx] = "o3"
        elif move(agent, 'L') == target:
            observations[state_idx] = "o4"
        elif move(agent, 'U') == target:
            observations[state_idx] = "o5"
        else:
            observations[state_idx] = "o6"
    return observations

def get_rewards():
    rewards = {}
    for state_idx, state_val in enumerate(states):
        agent, target, call = state_val
        for action_idx, action_val in enumerate(actions):
            rewards[(action_idx, state_idx)] = (call_reward if agent == target and call == 1 else 0) \
                - (1 if action_val != 'S' else 0)
    return rewards


if __name__ == "__main__":
    print("discount: 0.5")
    print("values: reward")
    print("states: 162")
    print("actions: 5")
    print("observations: o1 o2 o3 o4 o5 o6")
    print()
    for state_move_state, probability in get_transitions().items():
        print("T:", state_move_state[1], ":", state_move_state[0], ":", state_move_state[2], probability)
    get_transitions()
    for state_idx, observation in get_observations().items():
        print("O: * :", state_idx, ":", observation, "1")
    for action_state, reward in get_rewards().items():
        print("R:", action_state[0], ": * :", action_state[1], ": *", reward)
