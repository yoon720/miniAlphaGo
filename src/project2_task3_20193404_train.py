# EE807 Special Topics in EE <Deep Reinforcement Learning and AlphaGo>, Fall 2019
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# This is an example code to show how your code should be formatted for task 3 for training.
# Written by Jisoo Lee, Su Young Lee, Minguk Jang
# For virtual student with student id 20180001

import tensorflow as tf
import numpy as np
from boardgame import game1, game2, game3, game4


#####################################################################
"""                    DEFINE HYPERPARAMETERS                     """
#####################################################################
student_id = '20193404'
last_digit = int(student_id)%10

alpha = 0.001
max_epoch = 500
nx=6
ny=6


def data_augmentation1(d, v, p):
    dnew = np.zeros((nx, ny, 3, 8))
    dnew[:, :, :, 0] = d[:, :, :]
    dnew[:, :, :, 1] = d[::-1, :, :]
    dnew[:, :, :, 2] = d[:, ::-1, :]
    dnew[:, :, :, 3] = d[::-1, ::-1, :]
    dnew[:, :, :, 4] = np.rollaxis(d, 1, 0)
    dnew[:, :, :, 5] = dnew[::-1, :, :, 4]
    dnew[:, :, :, 6] = dnew[:, ::-1, :, 4]
    dnew[:, :, :, 7] = dnew[::-1, ::-1, :, 4]
    
    p_sq = p.reshape((6, 6))
    pnew = np.zeros((6, 6, 8))
    pnew[:, :, 0] = p_sq[:, :]
    pnew[:, :, 1] = p_sq[::-1, :]
    pnew[:, :, 2] = p_sq[:, ::-1]
    pnew[:, :, 3] = p_sq[::-1, ::-1]
    pnew[:, :, 4] = np.rollaxis(p_sq, 1, 0)
    pnew[:, :, 5] = pnew[::-1, :, 4]
    pnew[:, :, 6] = pnew[:, ::-1, 4]
    pnew[:, :, 7] = pnew[::-1, ::-1, 4]
    pnew = pnew.reshape(-1, 8)
    
    vnew = np.asarray([v for i in range(8)])
    
    daug = np.rollaxis(dnew, 3)
    paug = np.rollaxis(pnew, 1)
    
    return daug, vnew, paug

#####################################################################
"""  YOU CAN MODIFY OR ADD FUNCTIONS 
     WITHIN THE FOLLOWING CLASS game1_new(game1)                 """
#####################################################################
class game1_new(game1):
    def __init__(self, nx = 5, ny = 5, name = 'simple go'):
        self.nx = nx
        self.ny = ny
        self.name = name
        self.b = np.zeros((self.nx, self.ny, 1))
        self.state = -np.ones([2,1])
        self.game_in_progress = np.ones(1)
        self.player = 1
        self.n_valid_moves = nx*ny
        self.moves = 0
    
    def move(self, k):
        if hasattr(k, '__len__'):
            x = k[0]
            y = k[1]
            ixy = np.array([[x, y]])

        else:
            ixy = self.xy(k)
            
        b = self.b; state = self.state; p = self.player
        nxy = self.nx * self.ny
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]
        n_valid_moves = np.zeros((1))
        
        isvalid, bn, sn = self.valid(b[:, :, [0]], state[:, [0]], ixy, p)
        new_board[:, :, [0]] = bn
        new_state[:, [0]] = sn
        
        for p1 in range(nxy):
            vm1, b1, state1 = self.valid(new_board, new_state, self.xy(p1), p)
            n_valid_moves += vm1
            
        self.b = new_board
        self.state = new_state
        self.player = 3 - p
        self.n_valid_moves = n_valid_moves
        self.moves += 1
        return ixy
    
#####################################################################
"""                  Define Node class for MCTS                   """
#####################################################################
class Node:
    def __init__(self, game, mother=None, prob = np.ones(nx*ny)/(nx*ny)):
        self.game = copy.deepcopy(game)
        self.child = {}
        self.U = 0
        self.N = 0
        self.V = 0
        self.prob = prob
        
        if self.game.n_valid_moves == 0:
            self.outcome,_,_,_ = self.game.winner(self.game.b, self.game.state)
        else:
            self.outcome = None
        
        if self.outcome is not None:
            self.V = 1 if (self.game.player == self.outcome) else -1
            self.U = 0 if self.outcome is 0 else self.V*float('inf')
        
        self.mother = mother
     
    def valid_actions(self):
        game = self.game
        b = game.b
        state = game.state
        p = game.player
        nxy = game.nx * game.ny
        actions = []
        for p1 in range(nxy):
            vm1, b1, state1 = game.valid(b, state, game.xy(p1), p)
            if vm1 == 1:
                actions.append(p1)
        return actions
        
    def create_child(self, actions, probs):
        games = [ copy.deepcopy(self.game) for a in actions ]
    
        for action, game in zip(actions, games):
            game.move(action)
        
        child = { a: Node(g, self, probs[a]) for a,g in zip(actions, games) }
        self.child = child
        
    def create_d(self):
        b = self.game.b
        p = self.game.player
        d = np.zeros((nx, ny, 3), dtype=np.float32)
        d[:, :, 0] = (b[:, :, 0] == p)     # 1 if current player's stone is present, 0 otherwise
        d[:, :, 1] = (b[:, :, 0] == 3 - p) # 1 if opponent's stone is present, 0 otherwise
        d[:, :, 2] = 2 - p
        return d
        
    def search(self, sess, S):
        current = self
        while current.child and current.outcome is None:

            child = current.child
            max_U = max(c.U for c in child.values())

            actions = [ a for a,c in child.items() if c.U == max_U ]
            action = np.random.choice(actions)            

            if max_U == -float("inf"):
                current.U = float("inf")
                current.V = 1.0
                break
            
            elif max_U == float("inf"):
                current.U = -float("inf")
                current.V = -1.0
                break
                
            current = child[action]
        
        # expanded
        if not current.child and current.outcome is None:
            next_actions = current.valid_actions()
            feed_dict = {S: [current.create_d()]}
            sess_prob = sess.run(P, feed_dict = feed_dict).squeeze()
            sess_v = sess.run(v, feed_dict = feed_dict).squeeze()

            current.create_child(next_actions, sess_prob)
            current.V = -float(sess_v)
        current.N += 1

        # backup
        while current.mother:
            mother = current.mother
            mother.N += 1
            # beteen mother and child, the player is switched, extra - sign
            mother.V += (-current.V - mother.V)/mother.N

            #update U for all sibling nodes
            for sibling in mother.child.values():
                if sibling.U is not float("inf") and sibling.U is not -float("inf"):
                    sibling.U = sibling.V + 1.0*float(sibling.prob)* np.sqrt(mother.N)/(1+sibling.N)

            current = current.mother
            
    def next(self, temperature=1.0):

        child=self.child

        max_U = max([child[i].U if i in child.keys() else 0 for i in range(nx*ny)])

        prob = []
        if max_U == float("inf"):
            for i in range(nx*ny):
                if i in child.keys() and child[i].U == float("inf"):
                    prob.append(1.0)
                else: 
                    prob.append(0)
        else:
            maxN = max([child[i].N if i in child.keys() else 0 for i in range(nx*ny)])+1
            if maxN == 0:
                prob = [0]
            else:
                for i in range(nx*ny):                    
                    if i in child.keys():
                        prob.append((child[i].N/maxN)**(1/temperature))
                    else:
                        prob.append(0)
                    
        prob = np.asarray(prob)

        if np.sum(prob) > 0:
            prob /= np.sum(prob)  # normalize
        else:
            prob = np.ones(nx*ny) / (nx*ny)  # random
        
        nextstate = np.random.choice([child[i] if i in child.keys() else 0 for i in range(nx*ny)], p=prob)

        # -V for current
        return nextstate, (self.create_d(), -self.V, prob)

    def detach_mother(self):
        del self.mother
        self.mother = None


        
#####################################################################
"""                COMPUTATIONAL GRAPH CONSTRUCTION               """
#####################################################################
### DEFINE OPTIMIZER ###
def network_optimizer(v, z, Y, W, alpha, scope):
    loss_v = tf.reduce_mean(tf.square(z-v))
    loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Y, labels = W))
    loss = loss_v + loss_p
    
    # Parameters in this scope
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
    
    # L2 regularization
    for i in range(len(variables)):
        loss += 0.0001 * tf.nn.l2_loss(variables[i])
        
    # Optimizer
    optimizer = tf.train.AdamOptimizer(alpha).minimize(loss,\
            var_list = variables)
    return loss, optimizer

### NETWORK ARCHITECTURE ###

def network(state):
    init_weight = tf.random_normal_initializer(stddev = 0.1)
    init_bias = tf.constant_initializer(0.1)

    # Create variables "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer = init_weight)
    biases1 = tf.get_variable("biases1", [30], initializer = init_bias)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides = [1, 1, 1, 1], padding = 'SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variables "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer = init_weight)
    biases2 = tf.get_variable("biases2", [50], initializer = init_bias)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides = [1, 1, 1, 1], padding ='SAME')
    out2 = tf.nn.relu(conv2 + biases2)

    # Create variables "weights3" and "biases3".
    weights3 = tf.get_variable("weights3", [3, 3, 50, 2], initializer = init_weight)
    biases3 = tf.get_variable("biases3", [2], initializer = init_bias)

    # Create 3-1 layer for policy
    conv3 = tf.nn.conv2d(out2, weights3, strides = [1, 1, 1, 1], padding ='SAME')
    out3 = tf.nn.relu(conv3 + biases3)

    # Create 1st fully connected layer
    weights4 = tf.get_variable("weights4", [nx*ny*2, 60+last_digit], initializer = init_weight)
    biases4 = tf.get_variable("biases4", [60+last_digit], initializer = init_bias)
    
    out3_flat = tf.reshape(out3, [-1, nx * ny * 2])
    fc1 = tf.nn.relu(tf.matmul(out3_flat, weights4) + biases4)
    
    # Create 2nd fully connected layer
    weights5 = tf.get_variable("weights5", [60+last_digit, nx * ny], initializer = init_weight)
    biases5 = tf.get_variable("biases5", [nx*ny], initializer = init_bias)
    
    logit = tf.matmul(fc1, weights5) + biases5
    #logit = tf.reshape(out3, [-1, 6 * 6])
    
    # Create 3-2 layer for value
    weightsv1 = tf.get_variable("weightsv1", [3, 3, 50, 1], initializer = init_weight)
    biasesv1 = tf.get_variable("biasesv1", [1], initializer = init_bias)
    
    convv1 = tf.nn.conv2d(out2, weightsv1, strides = [1, 1, 1, 1], padding ='SAME')
    outv1 = tf.nn.relu(convv1 + biasesv1)

    # Create variables "weights1fc" and "biases1fc".
    weightsv2 = tf.get_variable("weightsv2", [nx * ny, 1], initializer = init_weight)
    biasesv2 = tf.get_variable("biasesv2", [1], initializer = init_bias)

    # Create 1st fully connected layer
    fcv1 = tf.reshape(outv1, [-1, nx * ny])
    value = tf.math.tanh(tf.matmul(fcv1, weightsv2) + biasesv2)

    ''' [IMPORTANT] tf.nn.softmax(logit) WILL BE USED FOR GRADING '''
    return (value, logit), tf.nn.softmax(logit, name='softmax')


def construct_graph(scope=student_id):

    #####################################################################
    ''' [IMPORTANT] ALL TENSORS MUST BE DEFINED IN scope              '''
    #####################################################################

    with tf.variable_scope(scope):
        # Input
        S = tf.placeholder(tf.float32, shape = [None, nx, ny, 3], name = "S")
        # Estimation for unnormalized log probability / probability
        (v, Y), P = network(S)
        # Target
        z = tf.placeholder(tf.float32, shape = [None], name = 'z')
        W = tf.placeholder(tf.float32, shape = [None, nx*ny], name = "W")
        # Define loss and optimizer for value network
        loss, optimizer = network_optimizer(v, z, Y, W, alpha, scope)

    return S, v, Y, P, z, W, loss, optimizer

def train(sess, saver, optimizer, S):

    #####################################################################
    '''             IMPLEMENT YOUR OWN TRAINING ALGORITHM             '''
  
    sess.run(tf.global_variables_initializer())
    

    for epoch in range(max_epoch):
        mytree = Node(game1_new(nx=nx, ny=ny))
        iteration = 0
        losses = 0
        while mytree.outcome is None:
            for _ in range(100):
                mytree.search(sess, S)

            mytree, (d, v, p) = mytree.next()   
            if str(type(mytree)) in ["<class 'numpy.int64'>" , "<class 'int'>"]:
                break
            
            mytree.detach_mother()
            
            daug, vaug, paug = data_augmentation1(d, v, p)
            
            feed_dict = {S: daug, W: paug, z: vaug}
            sess.run(optimizer, feed_dict=feed_dict)
            iteration += 1
            losses += sess.run(loss, feed_dict = feed_dict)
            
            
        print("Epoch: %3d\t Mean Loss: %10.5f" %(epoch, losses/iteration))

        if epoch % 10 == 0:
            saver.save(sess, './ckpt/project2_task3_%s.ckpt' % student_id)

    #####################################################################
    
class test_game(game1):
    def next_move_with_action(self, b, state, xy, p):     
    # next move assuming a given action xy. 'ng' is assumed to be 1.
        vm, bn, sn = self.valid(b[:,:, [0]], state[:, [0]], xy, p)
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]
        if vm[0] > 0:
            new_board[:, :, [0]] = bn
            new_state[:, [0]] = sn
        else:   # if invalid move
            _, _, sn = self.valid(b[:, :, [0]], state[:, [0]], -np.ones((1, 2)), p)
            new_state[:, [0]] = sn
        return new_board, new_state, vm
    
def find_action_and_execute(game, output, p, board, state):
    sorted_output = np.argsort(-output)    # descending order
    for xy in sorted_output:
        bn, sn, vm = game.next_move_with_action(b=board, state=state, xy=game.xy(xy), p=p)
        if vm > 0:
            return bn, sn, vm[0]
    return board, state, 0     # return the current board if no valid moves are possible for the current player



ckpt1 = './ckpt/project2_task3_%s.ckpt' % student_id 
def test_random(nx = 6, ny = 6, network = network, ckpt = ckpt1, n_test = 50000):
    # Load *.ckpt and play 50,000 games as black and 50,000 games as white 
    # against a pure random policy
    game = test_game(nx=nx, ny=ny)
    scope1 = student_id
    sess = tf.Session()

    S = tf.placeholder(tf.float32, shape=[None, game.nx, game.ny, 3], name="S")
    with tf.variable_scope(scope1):
        _, Y1 = network(S)
    
    # load
    saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope1))
    saver1.restore(sess, ckpt1)
    
    np0 = game.nx * game.ny * 2
    vm0 = 1
    
    winner = [0, 0, 0]
    for n in range(n_test):
        [board, state] = game.game_init(1)
        p = 1
        d = np.zeros((game.nx, game.ny, 3), dtype=np.float32)

        for i in range(np0):
            d[:, :, 0] = (board[:, :, 0] == p)     # 1 if current player's stone is present, 0 otherwise
            d[:, :, 1] = (board[:, :, 0] == 3 - p) # 1 if opponent's stone is present, 0 otherwise
            d[:, :, 2] = 2 - p            # 1: current player is black, 0: white        

            if p == 1:
                policy = sess.run(Y1, feed_dict={S: [d]})
                board, state, valid_move = find_action_and_execute(game, policy[0], p=p, board=board, state=state)
            else:
                rand_policy = np.random.uniform(0.0, 1.0, 36)
                policy = [rand_policy / sum(rand_policy)]
                board, state, valid_move = find_action_and_execute(game, policy[0], p=p, board=board, state=state)

            if vm0 + valid_move == 0:
                break    # game over if both players pass

            p = 3 - p   # change the current player
            vm0 = valid_move

        r, _, s1, s2 = game.winner(board, state)

        winner[r[0]] += 1
    
    print("[play as black]\twin: %.5f\t loss: %.5f\ttie: %.5f" %(winner[1]/n_test, winner[2]/n_test, winner[0]/n_test))

    winner = [0, 0, 0]
    for n in range(n_test):
        [board, state] = game.game_init(1)
        p = 1
        d = np.zeros((game.nx, game.ny, 3), dtype=np.float32)

        for i in range(np0):
            # update d
            d[:, :, 0] = (board[:, :, 0] == p)     # 1 if current player's stone is present, 0 otherwise
            d[:, :, 1] = (board[:, :, 0] == 3 - p) # 1 if opponent's stone is present, 0 otherwise
            d[:, :, 2] = 2 - p            # 1: current player is black, 0: white        

            if p == 1:
                rand_policy = np.random.uniform(0.0, 1.0, 36)
                policy = [rand_policy / sum(rand_policy)]            
                board, state, valid_move = find_action_and_execute(game, policy[0], p=p, board=board, state=state)
            else:
                policy = sess.run(Y1, feed_dict={S: [d]})
                board, state, valid_move = find_action_and_execute(game, policy[0], p=p, board=board, state=state)

            if vm0 + valid_move == 0:
                break    # game over if both players pass

            p = 3 - p   # change the current player
            vm0 = valid_move

        r, _, s1, s2 = game.winner(board, state)

        winner[r[0]] += 1

    print("[play as white]\twin: %.5f\t loss: %.5f\ttie: %.5f" %(winner[2]/n_test, winner[1]/n_test, winner[0]/n_test))



def test_two(nx = 6, ny = 6, network1=network, network2=network, ckpt1=ckpt1, ckpt2=ckpt1):
    # Load two *.ckpt files and play a game between them, 
    # where the first *.ckpt is black and the second is white
    game = test_game(nx=nx, ny=ny)
    sess = tf.Session()
    S = tf.placeholder(tf.float32, shape=[None, game.nx, game.ny, 3], name="S")
    
    scope1 = ckpt1.split('_')[2][:8]
    with tf.variable_scope(scope1):
        _, Y1 = network1(S)
        
    saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope1))
    saver1.restore(sess, ckpt1)
    
    if ckpt1 != ckpt2:
        scope2 = ckpt2.split('_')[2][:8]
        with tf.variable_scope(scope2):
            _, Y2 = network2(S)
        saver2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope2))
        saver2.restore(sess, ckpt2)
    
    np0 = game.nx * game.ny * 2
    vm0 = 1
    
    [board, state] = game.game_init(1)

    # first player is black (1)
    p = 1
    d = np.zeros((game.nx, game.ny, 3), dtype=np.float32)
    
    for i in range(np0):
        # update d
        d[:, :, 0] = (board[:, :, 0] == p)     # 1 if current player's stone is present, 0 otherwise
        d[:, :, 1] = (board[:, :, 0] == 3 - p) # 1 if opponent's stone is present, 0 otherwise
        d[:, :, 2] = 2 - p            # 1: current player is black, 0: white        

        if p == 1:
            policy = sess.run(Y1, feed_dict={S: [d]})
            board, state, valid_move = find_action_and_execute(game, policy[0], p=p, board=board, state=state)
            if valid_move == 0:
                print("No more moves possible for black. Must pass.")
        else:
            if ckpt1 != ckpt2:
                policy = sess.run(Y2, feed_dict={S: [d]})
            else:
                policy = sess.run(Y1, feed_dict={S: [d]})
                
            board, state, valid_move = find_action_and_execute(game, policy[0], p=p, board=board, state=state)
            if valid_move == 0:
                print("No more moves possible for white. Must pass.")

        if vm0 + valid_move == 0:
            break    # game over if both players pass

        print(valid_move, vm0)
        print(board[:,:,0])
 
        p = 3 - p   # change the current player
        vm0 = valid_move

    r, _, s1, s2 = game.winner(board, state)

    print("player 1: " + str(int(s1[0])) + " points")
    print("player 2: " + str(int(s2[0])) + " points")
    

if __name__ == "__main__":
    #####################################################################
    '''     THIS PART IS FOR YOU TO CHECK YOUR IMPLEMENTATION         '''
    '''     THIS PART IS NOT EXECUTED WHEN IMPORTING THIS SCRIPT      '''
    '''     BE CAREFUL WHEN USING GLOBAL VARIABLES                    '''

    '''             IMPLEMENT YOUR OWN ALGORITHM                      '''

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        S, v, Y, P, z, W, loss, optimizer = construct_graph(scope=student_id)
        saver = tf.train.Saver()
        train(sess, saver, optimizer, S)
    #####################################################################
