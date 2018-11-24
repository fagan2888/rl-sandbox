import gym
import numpy as np
from flask import Flask
import json


def main():

    env = gym.make('CartPole-v1')
    max_scores = (0, [], [])
    for _ in range(100):
        # random policy
        policy = np.random.rand(1, 4) - 0.5
        score, observations = play(env, policy)
        if score > max_scores[0]:
            max_scores = (score, observations, policy)
    print('Max Policy Score', max_scores[0])

    scores = []
    for _ in range(100):
        score, _ = play(env, max_scores[2])
        scores += [score]
    print('Average Score (100 trials)', np.mean(scores))

    # Flask server to see the cart pole
    app = Flask(__name__, static_folder='.')

    @app.route("/data")
    def data():
        return json.dumps(max_scores[1])

    @app.route('/')
    def root():
        return app.send_static_file('./index.html')
    app.run(host='0.0.0.0', port='3000')


def play(env, policy):
    observation = env.reset()
    done = False
    score = 0
    observations = []
    for _ in range(5000):
        # record observations for normalization and replay
        observations += [observation.tolist()]
        if done:
            break
        # pick an action according to the policy matrix
        outcome = np.dot(policy, observation)
        action = 1 if outcome > 0 else 0
        # make the action, record reward
        observation, reward, done, info = env.step(action)
        score += reward
    return score, observations


if __name__ == '__main__':
    main()
