import math

class ThymioController:
    def __init__(self, node):
        self.node = node

    async def stop(self):
        if self.node:
            v = {"motor.left.target": [0], "motor.right.target": [0]}
            await self.node.set_variables(v)

    async def set_motors(self, left, right):
        if self.node:
            v = {
                "motor.left.target": [int(left)],
                "motor.right.target": [int(right)],
            }
            await self.node.set_variables(v)

# ... (Keep calculate_control_command and check_waypoint_reached unchanged) ...
# ... (Copy them from previous responses if needed) ...
def calculate_control_command(robot_pos, robot_angle, target_pos, base_speed=100, k_p=2.0):
    rx, ry = robot_pos
    tx, ty = target_pos
    dx = tx - rx
    dy = ty - ry
    target_angle_rad = math.atan2(dx, -dy)
    target_angle_deg = math.degrees(target_angle_rad)
    angle_error = target_angle_deg - robot_angle
    angle_error = (angle_error + 180) % 360 - 180
    turn_speed = angle_error * k_p
    left_speed = base_speed + turn_speed
    right_speed = base_speed - turn_speed
    left_speed = max(min(left_speed, 500), -500)
    right_speed = max(min(right_speed, 500), -500)
    return left_speed, right_speed

def check_waypoint_reached(robot_pos, target_pos, threshold=20):
    dist = math.hypot(robot_pos[0] - target_pos[0], robot_pos[1] - target_pos[1])
    return dist < threshold