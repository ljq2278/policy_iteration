from multiprocessing import Process, Queue
import tensorflow as tf
import gym
import numpy as np

GAME = 'CartPole-v0'
env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
testor = None
work_num = 3
workors = []

class ActorNet:
    def __init__(self,role,scope,acpny_obj):
        self.build_inference(scope)
        if role=='worker':
            self.local_build_loss()
            self.local_build_grad()
            self.local_build_param_pull(acpny_obj.a_net)
            self.local_build_apply_grad_to_global(acpny_obj.a_net)

    def build_inference(self,scope):
        self.scope = scope
        with tf.variable_scope(self.scope+'_actor'):
            w_init = tf.random_normal_initializer(0., .1)
            self.s = tf.placeholder(shape=[None, N_S],dtype=tf.float32, name='S')
            hl = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='hl')
            self.a_prob = tf.layers.dense(hl, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_actor')

    def local_build_loss(self):
        with tf.variable_scope(self.scope + '_actor'):
            self.a = tf.placeholder(tf.int32, [None, ], 'A')
            self.td = tf.placeholder(tf.float32, [None, ], 'td')
        log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a, N_A, dtype=tf.float32), axis=1,keep_dims=True)
        exp_v = log_prob * self.td
        entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                 axis=1, keep_dims=True)  # encourage exploration
        self.exp_v = 0.001 * entropy + exp_v
        self.a_loss = tf.reduce_mean(-self.exp_v)

    def local_build_grad(self):
        self.a_grads = tf.gradients(self.a_loss, self.a_params)

    def local_build_apply_grad_to_global(self,actor):
        self.update = tf.train.RMSPropOptimizer(0.001, name='RMSPropA').apply_gradients(zip(self.a_grads, actor.a_params))

    def local_build_param_pull(self,actor):
        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, actor.a_params)]

    def global_apply_grad(self,sess,feed):
        sess.run(self.update,feed_dict=feed)

    def inference(self,sess,s_np):
        a_prob = sess.run(self.a_prob,feed_dict={self.s:s_np})
        return a_prob

    def local_setWeights(self,sess):
        sess.run(self.pull_a_params_op)

class CriticNet:
    def __init__(self,role,scope,acpny_obj):
        self.build_inference(scope)
        if role=='worker':
            self.local_build_loss()
            self.local_build_grad()
            self.local_build_param_pull(acpny_obj.c_net)
            self.local_build_apply_grad_to_global(acpny_obj.c_net)

    def build_inference(self, scope):
        self.scope = scope
        with tf.variable_scope(self.scope + '_critic'):
            w_init = tf.random_normal_initializer(0., .1)
            self.s = tf.placeholder(shape=[None, N_S], dtype=tf.float32, name='S')
            hl = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='hl')
            self.v = tf.layers.dense(hl, 1, tf.nn.softmax, kernel_initializer=w_init, name='v')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_critic')

    def local_build_loss(self):
        with tf.variable_scope(self.scope + '_critic'):
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
        td = tf.subtract(self.v_target, self.v, name='TD_error')
        self.c_loss = tf.reduce_mean(tf.square(td))

    def local_build_grad(self):
        self.c_grads = tf.gradients(self.c_loss, self.c_params)

    def local_build_apply_grad_to_global(self,critic):
        self.update = tf.train.RMSPropOptimizer(0.001, name='RMSPropC').apply_gradients(zip(self.c_grads, critic.c_params))

    def local_build_param_pull(self,critic):
        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, critic.c_params)]

    def global_apply_grad(self,sess,feed):
        sess.run(self.update,feed_dict=feed)

    def inference(self,sess,s_np):
        v = sess.run(self.v, feed_dict={self.s: s_np})
        return v

    def local_setWeights(self,sess):
        sess.run(self.pull_c_params_op)


class Testor:
    def __init__(self,a_net,c_net):
        self.env = gym.make(GAME).unwrapped
        self.a_net = a_net
        self.c_net = c_net

class Workor:
    def __init__(self,a_net,c_net):
        self.env = gym.make(GAME).unwrapped
        self.a_net = a_net
        self.c_net = c_net
    def work(self,sess):
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
                buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)

                feed_dict_a = {
                    self.a_net.s: buffer_s,
                    self.a_net.a: buffer_a,
                    self.a_net.td: self.c_net.inference(sess,buffer_s)
                }
                self.a_net.global_apply_grad(sess,feed=feed_dict_a)
                feed_dict_c = {
                    self.c_net.s: buffer_s,
                    self.c_net.v_target: buffer_v_target,
                }
                self.c_net.global_apply_grad(sess, feed=feed_dict_c)
                self.a_net.setWeights(testor.actor)
                self.c_net.setWeights(testor.critic)

                if done:
                    s = self.env.reset()
                else:
                    s = s_


if __name__ == "__main__":
    testor = Testor(a_net=ActorNet('testor','testor',None),c_net=CriticNet('testor','testor',None))
    for i in range(0,work_num):
        workors.append(Workor(a_net=ActorNet('worker',str(i),testor),c_net=CriticNet('worker',str(i),testor)))

    dataQueue = Queue(30)
    dataPreparation = [None] * 3
    for i in range(0, 3):
        dataPreparation[i] = Process(target=workors[i].work)
        dataPreparation[i].daemon = True
        dataPreparation[i].start()

