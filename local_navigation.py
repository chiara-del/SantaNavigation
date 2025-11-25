FORWARD_SPEED = 100       # nominal speed --> FORWARD_SPEED in your code 
OBST_THRL = 10      # low obstacle threshold to switch state 1->0
OBST_THRH = 20      # high obstacle threshold to switch state 0->1
OBST_SPEEDGAIN = 5  # /100 (actual gain: 5/100=0.05)

FOLLOW_PATH=0
OBSTACLE_AVOIDANCE=1

#after locking the node (await node.lock()) call:
await node.wait_for_variables({"prox.horizontal"})
state = FOLLOW_PATH          # 0=follow path, 1=obstacle avoidance

#in loop:
vals = list(node["prox.horizontal"])
obst_left, obst_right = vals[0], vals[4]
obst = [obst_left, obst_right]  # measurements from left and right prox sensors
     

#state switch
if state == FOLLOW_PATH: 
    # switch from goal tracking to obst avoidance if obstacle detected
    if (obst[0] > OBST_THRH):  # values higher if object near
        state = OBSTACLE_AVOIDANCE
    elif (obst[1] > OBST_THRH):
        state = OBSTACLE_AVOIDANCE
elif state == OBSTACLE_AVOIDANCE:
    if obst[0] < OBST_THRL: #values lower if object far 
        if obst[1] < OBST_THRL : 
            # switch from obst avoidance to goal tracking if obstacle got unseen
            state = FOLLOW_PATH

if  state == FOLLOW_PATH :
    #add already existing control command here 
    # goal tracking: turn toward the goal
    #leds_top = [0,0,0]
else:
    #leds_top = [30,30,30]
    # obstacle avoidance: accelerate wheel near obstacle
    left = FORWARD_SPEED + OBST_SPEEDGAIN * (obst[0] // 100)  #(left motor)
    right = FORWARD_SPEED + OBST_SPEEDGAIN * (obst[1] // 100) #(right motor)