import random

# Define the initial Q-values for each state-action pair
q_values = {
    ("hello", "greeting"): 0,
    ("how are you", "greeting"): 0,
    ("goodbye", "farewell"): 0,
}

# Define a function to choose the best action given a state
def choose_action(state):
    actions = ["greeting", "farewell"]
    best_action = actions[0]
    best_value = -float("inf")
    for action in actions:
        value = q_values.get((state, action), 0)
        if value > best_value:
            best_action = action
            best_value = value
    return best_action

# Define a function to update the Q-values based on a new experience
def update_q_value(state, action, reward, next_state):
    alpha = 0.5 # learning rate
    gamma = 0.9 # discount factor
    next_action = choose_action(next_state)
    current_q_value = q_values.get((state, action), 0)
    next_q_value = q_values.get((next_state, next_action), 0)
    new_q_value = current_q_value + alpha * (reward + gamma * next_q_value - current_q_value)
    q_values[(state, action)] = new_q_value

# Define the main function
def main():
    state = "start"
    print("Chatbot: Hello, how can I help you?")
    while state != "farewell":
        action = choose_action(state)
        if action == "greeting":
            print("User: Hi there!")
            reward = 0
            next_state = "greeting"
        elif action == "farewell":
            print("User: Bye for now!")
            reward = 0
            next_state = "farewell"
        else:
            print("Unknown action!")
            break
        update_q_value(state, action, reward, next_state)
        state = next_state
    print("Chatbot: Goodbye!")

# Call the main function
if __name__ == "__main__":
    main()
