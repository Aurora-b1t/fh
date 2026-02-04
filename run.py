from fh.newray import FHSSQPSKEnv
from fh.speedDQN import DQNAgent, DQNConfig
import numpy as np

def train_speed(num_episodes: int = 10_000):
    cfg = DQNConfig()
    agent = DQNAgent(cfg)
    env = FHSSQPSKEnv(enable_reactive=True, enable_sweep=True, enable_rayleigh=True)

    for ep in range(num_episodes):
        # 环境 reset（观测不用，状态恒定为 fixed_state）
        env.reset()
        state = agent.fixed_state.copy()

        # 选择动作（离散索引 -> hoprate）
        action_idx = agent.select_action(state)
        hoprate = agent.action_values[action_idx+1]
        obs, reward, done, _, info = env.step(np.array([hoprate], dtype=np.float32))

        next_state = agent.fixed_state.copy()  # 状态恒定

        agent.replay.push(state, action_idx, reward, next_state, float(done))

        loss = agent.train_step()

        print(f"Episode {ep + 1}, selected hoprate: {hoprate}, reward: {reward}, loss: {loss}, info: {info}")

        if agent.steps_done % agent.cfg.target_update_interval == 0:
            agent.update_target()

    env.close()

if __name__ == "__main__":
    train_speed(num_episodes=1500)