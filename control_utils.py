import math

class ThymioController:
    """Wrapper to handle Thymio motor commands via the node."""
    def __init__(self, node):
        self.node = node

    async def stop(self):
        await self.set_motors(0, 0)

    async def set_motors(self, left, right):
        if self.node:
            v = {
                "motor.left.target": [int(left)],
                "motor.right.target": [int(right)],
            }
            await self.node.set_variables(v)

class PathFollower:
    """Manages the state of path following."""
    def __init__(self, speed=100, gain=3.0, reach_threshold=30):
        self.threshold = reach_threshold
        self.speed = speed
        self.gain = gain
        self.path = None
        self.current_idx = 0  # <--- This matches main.py

    def set_path(self, path):
        self.path = path
        self.current_idx = 1 # Start at index 1 (0 is current robot pos)

    def get_command(self, robot_pose):
        """
        Returns (left_speed, right_speed, goal_reached_bool).
        """
        if not self.path or self.current_idx >= len(self.path):
            return 0, 0, True # Reached / No Path

        target = self.path[self.current_idx]
        robot_xy = robot_pose[0]
        robot_angle = robot_pose[1]

        # Check if reached waypoint
        dist = math.hypot(robot_xy[0] - target[0], robot_xy[1] - target[1])
        if dist < self.threshold:
            self.current_idx += 1
            if self.current_idx >= len(self.path):
                return 0, 0, True # Just reached goal
            target = self.path[self.current_idx]

        # Calculate P-Control
        dx = target[0] - robot_xy[0]
        dy = target[1] - robot_xy[1]
        
        target_angle = math.degrees(math.atan2(dx, -dy))
        error = (target_angle - robot_angle + 180) % 360 - 180
        
        turn = error * self.gain
        left = int(max(min(self.speed + turn, 500), -500))
        right = int(max(min(self.speed - turn, 500), -500))
        
        return left, right, False

def calculate_avoidance_commands(prox, speed, gain):
    """Calculates Local Avoidance command (Braitenberg)."""
    # Left: 0,1,2. Right: 2,3,4
    left_stim = prox[0]*1.0 + prox[1]*0.8 + prox[2]*0.2
    right_stim = prox[4]*1.0 + prox[3]*0.8 + prox[2]*0.2
    
    turn = (left_stim - right_stim) * gain
    
    # Slow down if very close (Safety)
    if max(prox) > 3500: speed = 0 
    
    left = int(max(min(speed + turn, 500), -500))
    right = int(max(min(speed - turn, 500), -500))
    return left, right