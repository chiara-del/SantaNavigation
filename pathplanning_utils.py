from math import hypot  
import numpy as np
import heapq

# ---------- 2D geometry ----------
#math tools 
def sub(a,b): return (a[0]-b[0], a[1]-b[1])               #Return vector difference a - b
def cross(a,b): return a[0]*b[1] - a[1]*b[0]              #scalar product 
def dist(a,b): return hypot(a[0]-b[0], a[1]-b[1])         #euclidean distance between the two nodes a and b 

def orient(a,b,c, eps=1e-9): 
    """Orientation test for triplet (a,b,c).
    Returns:
         +1 → counterclockwise turn
         -1 → clockwise turn
         0 → collinear"""
    v = cross(sub(b,a), sub(c,a))   
    return 1 if v > eps else (-1 if v < -eps else 0) 
 
def on_segment(p, q, r, eps=1e-9):  
    """Check whether point q lies on segment p-r (including boundaries).
    Uses bounding-box test + collinearity check."""
    if min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps:    
        return abs(cross(sub(q,p), sub(r,p))) <= eps      
    return False  

def segments_intersect(p1, q1, p2, q2, allow_shared_endpoint=True):  
    """Robust segment–segment intersection test.
   Handles general case + collinear overlapping edges."""
    o1 = orient(p1, q1, p2)   
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    # proper intersection
    if o1 != o2 and o3 != o4: 
        return True

    # collinear cases
    if o1 == 0 and on_segment(p1, p2, q1):
        return not allow_shared_endpoint or (p2 != p1 and p2 != q1)
    if o2 == 0 and on_segment(p1, q2, q1):
        return not allow_shared_endpoint or (q2 != p1 and q2 != q1)
    if o3 == 0 and on_segment(p2, p1, q2):
        return not allow_shared_endpoint or (p1 != p2 and p1 != q2)
    if o4 == 0 and on_segment(p2, q1, q2):
        return not allow_shared_endpoint or (q1 != p2 and q1 != q2) 

    return False 
 
def point_in_polygon(pt, poly):   
    """Standard ray-casting point-in-polygon test.
    Returns True if point lies inside the polygon."""
    x, y = pt
    inside = False
    n = len(poly)  
    for i in range(n):
        x1, y1 = poly[i]  
        x2, y2 = poly[(i+1) % n]
        # Check if ray intersects polygon edge
        if ((y1 > y) != (y2 > y)):
            # Compute intersection x-coordinate of the segment with horizontal ray
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1
            if x < x_int:
                inside = not inside
    return inside

# ---------- Visibility ----------
def segment_hits_polygon(a, b, poly, safety=0.0): 
    """Check whether segment a-b intersects polygon 'poly'.
     Includes:
       - Intersection with edges
       - Midpoint being inside polygon
       - Optional safety distance check"""
    n = len(poly)

    # Check for edge intersection (except at shared endpoints)
    for i in range(n):
        p = poly[i]     
        q = poly[(i+1) % n]
        allow_endpoint = (a == p or a == q or b == p or b == q)                   

        if segments_intersect(a, b, p, q, allow_shared_endpoint=allow_endpoint):  
            if not allow_endpoint:
                return True
            
    # Check if the middle of the segment lies inside the polygon        
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    if point_in_polygon(mid, poly):
        return True

    if safety > 0.0:
        # crude guard: ensure segment midpoint not too close to polygon vertices
        mid = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)
        for v in poly:
            if dist(mid, v) < safety:
                return True
    return False

def visible(a, b, polygons, safety=0.0): 
    """ Check whether points a and b have direct line of sight.
    Rejects if either point lies inside an obstacle or the segment intersects any obstacle."""  
    for P in polygons:
        if point_in_polygon(a, P) or point_in_polygon(b, P): 
            return False
    for P in polygons:
        if segment_hits_polygon(a, b, P, safety=safety): 
            return False
    return True  

# ---------- Graph + A* ----------
def build_visibility_graph(start, goal, polygons, safety=0.0):
    """Build a visibility graph containing:
       - start and goal
       - all polygon vertices
     Edges are inserted only if unobstructed."""
    nodes = [start, goal] + [v for poly in polygons for v in poly]   
    # deduplicate points (avoid exact duplicates)
    seen = set()  
    uniq = []
    for v in nodes:
        key = (round(v[0], 9), round(v[1], 9))   
        if key not in seen:
            seen.add(key)  
            uniq.append(v)
    nodes = uniq    

    # Adjacency list: node → list of (neighbor, weight)
    adj = {v: [] for v in nodes} 
    # Attempt visibility between all pairs
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a, b = nodes[i], nodes[j]  
            if visible(a, b, polygons, safety=safety): 
                w = dist(a, b) 
                adj[a].append((b, w)) 
                adj[b].append((a, w))
    return adj 

def a_star(adj, start, goal):
    """Classic A* search on a visibility graph.
     Uses Euclidean distance as heuristic.
     Returns (path, total_cost)."""
    def h(u): return dist(u, goal)   

    pq = [] 
    heapq.heappush(pq, (h(start), 0.0, start)) 
    g = {start: 0.0} 
    parent = {start: None} 
    closed = set() 
    
    while pq:  
        _, gcost, u = heapq.heappop(pq) 
        if u in closed:  
            continue
        if u == goal: 
            # reconstruct final path
            path = []  
            cur = u  
            while cur is not None: 
                path.append(cur) 
                cur = parent[cur] 
            path.reverse() 
            return path, g[u] 
        closed.add(u) 
        for v, w in adj[u]: 
            ng = gcost + w 
            if v not in g or ng < g[v]: 
                g[v] = ng 
                parent[v] = u 
                heapq.heappush(pq, (ng + h(v), ng, v)) 
    return None, float('inf')  

# ---------- Path cleanup ----------
def simplify_collinear(path, eps=1e-3): 
    """
    Remove unnecessary intermediate points that lie on perfectly straight lines.
    Keeps endpoints and only retains points that introduce direction change.
    """ 
    if len(path) <= 2:
        return path[:]
    out = [path[0]] 
    for i in range(1, len(path)-1):
        a, b, c = path[i-1], path[i], path[i+1] 
        # Use cross product to detect deviation from collinearity
        if abs(cross((b[0]-a[0], b[1]-a[1]), (c[0]-b[0], c[1]-b[1]))) > eps: 
            out.append(b)
    out.append(path[-1]) 
    return out

# ---------- Public ----------
def plan_path(start, goal, grown_polygons, safety=0.02):
    """
    High-level path planning function.
    - Constructs visibility graph
    - Runs A*
    - Simplifies resulting path
    Returns:
        (list of waypoints, total_length)
    """
    graph = build_visibility_graph(start, goal, grown_polygons, safety=safety)
    path, L = a_star(graph, start, goal)
    if path is None:
        return None, float('inf')
    return simplify_collinear(path), L


def check_kidnapping(pose, target, path, prev_idx, threshold):
    """
    Check whether the robot has deviated too far from the expected path segment,
    which may indicate 'kidnapping' (unexpected displacement).
    """
    if not path:
        return False
    p, a, b = np.array(pose[0]), np.array(path[prev_idx]), np.array(target)
    n = b - a
    norm_n = np.linalg.norm(n)
    dist = np.abs(np.cross(n, a - p)) / norm_n if norm_n > 0 else np.linalg.norm(p - a)
    return dist > threshold