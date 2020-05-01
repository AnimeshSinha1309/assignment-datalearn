x = 0.88
call_reward = 101

class Locale:

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, value):
        return self.x == value.x and self.y == value.y

    @property
    def x(self):
        return self.x

    @property
    def y(self):
        return self.y

    @property
    def name(self):
        return self.x + 3 * self.y

    @property
    def pair(self):
        return (self.x, self.y)

    def __str__(self):
        return str(self.name)

    def move(self, direction):
        if direction == 'L':
            return Locale((max(self.x - 1, 0), self.y))
        if direction == 'R':
            return Locale((min(self.x + 1, 2), self.y))
        if direction == 'U':
            return Locale((self.x, max(self.y - 1, 0)))
        if direction == 'D':
            return Locale((self.x, min(self.y + 1, 2)))
        if direction == 'S':
            return Locale(locale)


class State:

    def __init__(self, agent, target, call):
        self.agent = agent
        self.target = target
        self.call = call

    @property
    def agent(self):
        return self.agent

    @property
    def target(self):
        return self.target

    @property
    def call(self):
        return self.call

    @property
    def name(self):
        return self.agent.name + self.target.name * 9 + self.call * 81

    def __str__(self):
        return str(self.name)

locale = [Locale(i, j) for i in range(3) for j in range(3)]
states = [State(agent, target, call) for call in [0, 1] for target in range(9) for agent in range(9)]
actions = ['S', 'L', 'R', 'U', 'D']


def get_transitions():
    transitions = {}
    y = 0.85
    for state_idx, state_val in enumerate(states):
        call, target, agent = state_val

        end_agent_actlist = [
            {move(agent, 'S'): 1},
            {move(agent, 'U'): x, move(agent, 'D'): 1 - x},
            {move(agent, 'D'): x, move(agent, 'U'): 1 - x},
            {move(agent, 'L'): x, move(agent, 'R'): 1 - x},
            {move(agent, 'R'): x, move(agent, 'L'): 1 - x}
        ]
        end_target = {target: 0.4}
        for target_action in ['R', 'L', 'U', 'D']:
            pos = move(target, target_action)
            end_target[pos] = end_target[pos] + 0.15 if target == pos else 0.15
        end_call = {1: 0.4, 0: 0.6} if call == 0 or target == agent else {0: 0.2, 1: 0.8}

        for action in range(5):
            for agent_state, agent_prob in end_agent_actlist[action].items():
                for target_state, target_prob in end_target.items():
                    for call_state, call_prob in end_call.items():
                        new_state = agent_state + target_state * 9 + call_state * 81
                        prob = agent_prob * target_prob * call_prob
                        tran = (state_idx, action, new_state)
                        transitions[tran] = transitions[tran] + prob if tran in transitions.keys() else prob
    return transitions


def get_observations():
    observations = {}
    for state_idx, state in enumerate(states):
        call, target_pos, agent_pos = state
        if agent_pos == target_pos:
            observations[state_idx] = "o1"
        elif move(agent_pos, 'L') == target_pos:
            observations[state_idx] = "o4"
        elif move(agent_pos, 'R') == target_pos:
            observations[state_idx] = "o2"
        elif move(agent_pos, 'U') == target_pos:
            observations[state_idx] = "o5"
        elif move(agent_pos, 'D') == target_pos:
            observations[state_idx] = "o3"
        else:
            observations[state_idx] = "o6"
    return observations

def get_rewards():
    rewards = {}
    for state_idx, state_val in enumerate(states):
        call, target_pos, agent_pos = state_val
        for action_idx, action_val in enumerate(actions):
            rewards[(action_idx, state_idx)] = (call_reward if agent_pos == target_pos and call == 1 else 0) \
                - (1 if action_val != 'S' else 0)
    return rewards


if __name__ == "__main__":
    print("discount: 0.5")
    print("values: reward")
    print("states: 162")
    print("actions: 5")
    print("observations: o1 o2 o3 o4 o5 o6")
    print()
    for st_act_st, probability in get_transitions().items():
        print("T:", st_act_st[1], ":", st_act_st[0], ":", st_act_st[2], probability)
    get_transitions()
    for state_idx, observation in get_observations().items():
        print("O: * :", state_idx, ":", observation, "1")
    for action_state, reward in get_rewards().items():
        print("R:", action_state[0], ": * :", action_state[1], ": *", reward)
