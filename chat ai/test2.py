import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Define the initial Q-values for each state-action pair
q_values = {
    ("start", "greeting"): 0,
    ("start", "query"): 0,
    ("query", "response"): 0,
    ("query", "goodbye"): 0,
}

# Define a function to choose the best action given a state
def choose_action(state):
    actions = ["greeting", "query", "goodbye"]
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

# Define a function to calculate the reward for a given action
def calculate_reward(state, action, next_state, user_input):
    if action == "greeting":
        return 0 # no reward for a greeting
    elif action == "query":
        # Calculate the sentiment of the user's query
        sentiment = sia.polarity_scores(user_input)["compound"]
        if sentiment > 0.5:
            return 1 # positive sentiment gets a high reward
        elif sentiment < -0.5:
            return -1 # negative sentiment gets a low reward
        else:
            return 0 # neutral sentiment gets no reward
    elif action == "goodbye":
        return 0 # no reward for a goodbye

# Define the main function
def main():
    state = "start"
    while True:
        action = choose_action(state)
        if action == "greeting":
            print("Chatbot: Hello, how can I help you?")
        elif action == "query":
            user_input = input("User: ")
            reward = calculate_reward(state, action, "response", user_input)
            update_q_value(state, action, reward, "response")
            print("Chatbot: Here is my response.")
        elif action == "goodbye":
            print("Chatbot: Goodbye!")
            break
        state = "query"
