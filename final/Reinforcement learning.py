import gymnasium as gym

"""Recibe datos y le decimos que si se esta yendo a la izquierda (angulo menor de 
0 grados hasta -24) entonces se mueva a la izquierda para balancear, y a la derecha 
en caso contra."""

def pi_1(state):
    pos, vel, ang, ang_vel = state
    return 1 if ang > 0 else 0 #1 es derecha, 0 es izquierda

def main():
    env = gym.make(id="CartPole-v0", render_mode="human")
    s0, _ = env.reset()
    done = False
    print(s0)
    action_space = env.action_space.n

    while not done:
        action = env.action_space.sample()  # Action to perform
        state_1, reward, done, _, _ = env.step(action)  # Result
        s0 = state_1

if __name__ == "__main__":
    main()