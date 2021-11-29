from flask import Flask, Response, jsonify, make_response
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

app = Flask(__name__)


@app.route('/')
def hello_world() -> Response:  # put application's code here
    response = make_response('Hello World!', 200)
    response.mimetype = "text/plain"
    return response


@app.route('/analysis')
def perform_analysis() -> Response:
    env = gym.make("Taxi-v3")
    model = PPO.load("models/taxi/best_model", env)

    state = env.reset()

    reward_score = 0
    steps = 0
    for _ in range(1000):
        action, _states = model.predict(state)
        new_state, reward, done, info = env.step(action)
        state = new_state

        reward_score += reward
        steps += 1

        env.render()
        print(f"Reward: {reward}, Step#: {steps}")

        if done:
            break

    env.close()

    json_result = jsonify({'reward': reward_score, 'steps': steps})
    response = make_response(json_result, 200)
    response.mimetype = "application/json"
    return response


@app.route('/test_agent')
def get_agents_averages() -> Response:
    monitored_env = Monitor(gym.make("Taxi-v3"))
    model = PPO.load("models/taxi/best_model", monitored_env)

    mean_rewards, std_rewards = evaluate_policy(model, monitored_env, n_eval_episodes=10)

    monitored_env.close()

    json_result = jsonify({'mean_rewards': mean_rewards, 'std_rewards': std_rewards})
    response = make_response(json_result, 200)
    response.mimetype = "application/json"
    return response


if __name__ == '__main__':
    app.run()
