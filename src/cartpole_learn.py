'''
Copyright (c) 2017, Juan Camilo Gamboa Higuera, Anqi Xu, Victor Barbaros, Alex Chatron-Michaud, David Meger

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from plant import gTrig_np
from cartpole import default_params
from cartpole import CartpoleDraw
from control.matlab import *

#np.random.seed(31337)
np.set_printoptions(linewidth=500)

# FOR YOU TODO: Fill in this function with a control 
# policy that computes a useful u from the input x

def policyfn_PID(previous_error,error,integral,delta_t):
    kp = -200
    ki = -0.2
    kd = -300

    integral += error
    derivative = (error - previous_error)/delta_t
    u = kp * error  + ki*integral + kd * derivative

    return np.array([u]), integral



def policyfn_lqr(theta, m, M, l, x):
    g = 9.8
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, g * m / M, 0],
        [0, 0, 0, 1],
        [0, 0, (m + M) * g / (l * M), 0]
    ])

    B = np.matrix([
        [0],
        [1 / M],
        [0],
        [1 / (l * M)]
    ])

    # The Q and R matrices are emperically tuned. It is described further in the report
    Q = np.matrix([
        [100, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 10000, 0],
        [0, 0, 0, 100]
    ])

    R = np.matrix([500])

    K,S,E = lqr(A,B,Q,R)
    np.matrix(K)
    x_lqr = np.matrix([
        [np.squeeze(np.asarray(x[0]))],
        [np.squeeze(np.asarray(x[1]))],
        [np.squeeze(np.asarray(theta))],
        [np.squeeze(np.asarray(x[2]))]
    ])
    desired = np.matrix([
        [x[0]],
        [0],
        [np.pi],
        [0]
    ])
    F = -(K * (x_lqr - desired))
    u = F.item(0)
    return np.array([u])

#  x = transpose(x_pos, x_dot, theta_dot, sin(theta), cos(theta))
def policy_swing_up(theta,m,M,l, x):

    # theta representation:
    g = 9.8
    if(x[4] > 0):
        #swing up
        Ee = 0.5*m*l*l*x[2]**2 - m*g*l*(1 + x[4])
        k = 0.6
        A = k* Ee*x[4] * x[2]
        delta = m*x[3]**2 + M
        u = A*delta - m*l*(x[2]**2)*x[3] - m*g*x[3]*x[4] #  dynamic model
        if( -np.pi/2 <theta <0 and x[2] < 0): #  deal with the situation on the (-1,0) direction
            u = 0
        # print "theta: ", theta
        # print "x3: ", x[3]
        # print "x4: ", x[4]
        # print "u: ", u
        return np.array([u])
    else:
        # balance with lqr
        g = 9.8
        A = np.matrix([
            [0, 1, 0, 0],
            [0, 0, g * m / M, 0],
            [0, 0, 0, 1],
            [0, 0, (m + M) * g / (l * M), 0]
        ])

        B = np.matrix([
            [0],
            [1 / M],
            [0],
            [1 / (l * M)]
        ])

        # The Q and R matrices are emperically tuned. It is described further in the report
        Q = np.matrix([
            [100, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 10000, 0],
            [0, 0, 0, 100]
        ])

        R = np.matrix([500])

        K, S, E = lqr(A, B, Q, R)
        np.matrix(K)
        x_lqr = np.matrix([
            [np.squeeze(np.asarray(x[0]))],
            [np.squeeze(np.asarray(x[1]))],
            [np.squeeze(np.asarray(theta))],
            [np.squeeze(np.asarray(x[2]))]
        ])
        desired = np.matrix([
            [x[0]],
            [0],
            [np.pi],
            [0]
        ])
        F = -(K * (x_lqr - desired))
        u = F.item(0)

    return np.array([u])

def find_error(x_t):
    previous_error = x_t[3] - np.pi
    return previous_error

def apply_controller(plant,params,H,flag, policy=None):
    '''
    Starts the plant and applies the current policy to the plant for a duration specified by H (in seconds).

    @param plant is a class that controls our robot (simulation)
    @param params is a dictionary with some useful values 
    @param H Horizon for applying controller (in seconds)
    @param policy is a function pointer to the code that implements your 
            control solution. It will be called as u = policy( state )
    '''

    # start robot
    x_t, t = plant.get_plant_state()
    integral = 0
    delta_t = plant.dt
    previous_error = find_error(x_t)
    if plant.noise is not None:
        # randomize state
        Sx_t = np.zeros((x_t.shape[0],x_t.shape[0]))
        L_noise = np.linalg.cholesky(plant.noise)
        x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
    
    sum_of_error = 0
    H_steps = int(np.ceil(H/plant.dt))
    for i in xrange(H_steps):
        error = find_error(x_t)
        # convert input angle dimensions to complex representation
        x_t_ = gTrig_np(x_t[None,:], params['angle_dims']).flatten()
        l = plant.params['l']
        m = plant.params['m']
        b = plant.params['b']
        M = plant.params['M']
        #  get command from policy (this should be fast, or at least account for delays in processing):
        if(flag == 1):
            u_t, integral = policy(previous_error, error, integral, delta_t)  # using pid to find out the control
        if(flag == 2):
            #  if theta < 3.14 and goes to -3.14, the cart performs a clockwise turn, if theta > 3.14 and goes to 9.42, cart goes counter clockwise
            #  but in any case, the right position for stabilizing is to set theta to 3.14 or 9.42 or -3.14 ... in any case, cos(theta) = -1
            # print 'theta: ', x_t[3]
            u_t = policy(x_t[3],m,M,l,x_t_)
        if(flag == 3):
            u_t = policy(x_t[3],m,M,l,x_t_)
        previous_error = error
	    #send command to robot:
        plant.apply_control(u_t)
        plant.step()
        x_t, t = plant.get_plant_state()
	err = np.array([0,l]) - np.array( [ x_t[0] + l*x_t_[3], -l*x_t_[4] ] )
        dist = np.dot(err,err )
        sum_of_error = sum_of_error + dist

        if plant.noise is not None:
            # randomize state
            x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        
        if plant.done:
            break

    print "Error this episode %f"%(sum_of_error)
        
    # stop robot
    plant.stop()

def main():
    
    # learning iterations
    N = 5   
    H = 10

    learner_params = default_params()
    plant_params = learner_params['params']['plant'] 
    plant = learner_params['plant_class'](**plant_params)
   
    draw_cp = CartpoleDraw(plant)
    draw_cp.start()
    
    # loop to run controller repeatedly
    for i in xrange(N):
        
        # execute it on the robot
        plant.reset_state()
        flag = 3;  # 1 for PID, 2 for LQR to do balance, 3 for swing up
        apply_controller(plant,learner_params['params'], H, flag, policy_swing_up)  # using PID to find out the control

    
if __name__ == '__main__':
    main()
    
