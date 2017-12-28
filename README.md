# COMP765-Assignment2
Assignment 2 for COMP765, Intelligent Robotics.

### Usage

Modify the parameter in line 249, file cartpole_learn.py.
change function name: `policy_swing_up` or `policyfn_PID` or `policyfn_lqr` to invoke different control policy.

```sh
 flag = 3;  # 1 for PID, 2 for LQR to do balance, 3 for swing up
 apply_controller(plant,learner_params['params'], H, flag, policy_swing_up)  # using PID to find out the control
```
