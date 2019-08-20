from multiprocessing import Process, Queue
import tensorflow as tf
import gym
import numpy as np

GAME = 'CartPole-v0'
env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
testor = None

class ActorNet:
    def build_inference(self,scope):
        self.scope = scope
        with tf.variable_scope(self.scope+'_actor'):
            w_init = tf.random_normal_initializer(0., .1)
            self.s = tf.placeholder(shape=[None, N_S],dtype=tf.float32, name='S')
            hl = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='hl')
            self.a_prob = tf.layers.dense(hl, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_actor')
    def build_loss(self,td_np):
        with tf.variable_scope(self.scope + '_actor'):
            self.a = tf.placeholder(tf.int32, [None, ], 'A')
            log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a, N_A, dtype=tf.float32), axis=1,
                                     keep_dims=True)
            exp_v = log_prob * td_np
            entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                     axis=1, keep_dims=True)  # encourage exploration
            self.exp_v = 0.001 * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
    def global_build_grad(self):
        with tf.variable_scope(self.scope + '_actor'):
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.train_op = tf.train.RMSPropOptimizer(0.001, name='RMSPropA').apply_gradients(zip(self.a_grads, self.a_params))

    def local_build_param_pull(self,actor):
        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, actor.a_params)]

    def inference(self,sess,s_np):
        a_prob = sess.run(self.a_prob,feed_dict={self.s:s_np})
        return a_prob

    def local_setWeights(self,sess):
        sess.run(self.pull_a_params_op)

class CriticNet:
    def build_inference(self, scope):
        self.scope = scope
        with tf.variable_scope(self.scope + '_critic'):
            w_init = tf.random_normal_initializer(0., .1)
            self.s = tf.placeholder(shape=[None, N_S], dtype=tf.float32, name='S')
            hl = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='hl')
            self.v = tf.layers.dense(hl, 1, tf.nn.softmax, kernel_initializer=w_init, name='v')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_critic')

    def build_loss(self):
        with tf.variable_scope(self.scope + '_critic'):
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            self.c_loss = tf.reduce_mean(tf.square(td))

    def global_build_grad_and_apply(self):
        with tf.variable_scope(self.scope + '_critic'):
            self.c_grads = tf.gradients(self.c_loss, self.c_params)
            self.train_op = tf.train.RMSPropOptimizer(0.001, name='RMSPropC').apply_gradients(zip(self.c_grads, self.c_params))

    def local_build_param_pull(self,critic):
        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, critic.c_params)]

    def inference(self,sess,s_np):
        v = sess.run(self.v, feed_dict={self.s: s_np})
        return v

    def local_setWeights(self,sess):
        sess.run(self.pull_c_params_op)


class workor:
    def __init__(self,a_net,c_net):
        self.env = gym.make(GAME).unwrapped
        self.a_net = a_net
        self.c_net = c_net
    def work(self):
        s = self.env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        total_step = 0
        while True:
            a_prob = self.a_net.inference(s)
            a = np.random.choice(range(a_prob.shape[1]),p=a_prob.ravel())
            s_, r, done, info = self.env.step(a)
            ep_r += r
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            total_step+=1
            if total_step % 100 == 0 or done:
                v_s_ = self.c_net.inference(s_)
                buffer_v_target = []
                for r in buffer_r[::-1]:  # reverse buffer r
                    v_s_ = r + 0.9 * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()
                updateTestor()
                self.a_net.setWeights(testor.actor)
                self.c_net.setWeights(testor.critic)
                if done:
                    s = self.env.reset()
                else:
                    s = s_
# def actor_net(s,scope):
#     h = tf.keras.layers.Dense(200,activation='relu',name='actor_hidden_'+scope)(s)
#     return tf.keras.layers.Dense(N_A, activation='softmax', name='actor_out_'+scope)(h)
#
# def critic_net(s,scope):
#     h = tf.keras.layers.Dense(100,activation='relu',name='critic_hidden_'+scope)(s)
#     return tf.keras.layers.Dense(1,  name='critic_out_'+scope)(h)

a_net = actor_net(s)
# a_prop = a_net(s)
c_net = critic_net(s)
# v = c_net(s)

def getErrTd():

    v = net.infer(s)
    return v_target-v



work_num = 3
workors = []
if __name__ == "__main__":

    for i in range(0,work_num):
        workors.append(Workor())

    testor = Testor()

    dataQueue = Queue(30)
    dataPreparation = [None] * 3
    for proc in range(0, 3):
        # dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
        dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue,))
        dataPreparation[proc].daemon = True
        dataPreparation[proc].start()

    updateTestor(testor,workors)
    updateWorks(testor, workors)