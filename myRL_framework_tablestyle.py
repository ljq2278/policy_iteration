import gym

game_name = 'MountainCar-v0'
env = gym.make(game_name)
maxEpo = 100
a_n = env.action_space.n
obv_low_list = env.observation_space.low
obv_high_list = env.observation_space.high
gamma = 0.9
lr = 0.03

def make_decision(obv):
    q = q_table.get_q(obv,0)
    a = 0
    for i in range(1,a_n):
        q_cur = q_table.get_q(obv,i)
        if q_cur>q:
            q = q_cur
            a = i
    return a

def learn(obv,act,reward,obv_nxt,act_nxt):
    q_obj = reward + gamma * q_table.get_q(obv_nxt,act_nxt)
    delta = q_obj - q_table.get_q(obv,act)
    q_table.set_q(obv, act, q_table.get_q(obv,act) + lr * delta)

def train():
    epo = 0
    while epo < maxEpo:
        obv = env.reset()
        act = make_decision(obv)
        done = False
        while not done:
            obv_nxt, reward, done, _ = env.step(act)
            act_nxt = make_decision(obv_nxt)
            learn(obv,act,reward,obv_nxt,act_nxt)


