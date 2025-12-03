from math import hypot  #imports euclidean norm for distance 
import numpy as np
import heapq

# ---------- 2D geometry ----------
#math tools 
def sub(a,b): return (a[0]-b[0], a[1]-b[1])               #vector difference a,b 
def cross(a,b): return a[0]*b[1] - a[1]*b[0]              #scalar product 
def dist(a,b): return hypot(a[0]-b[0], a[1]-b[1])         #euclidean distance between the two nodes a and b 

#orientation de trois points 
def orient(a,b,c, eps=1e-9):    #eps pour etre robuste au bruit 
    v = cross(sub(b,a), sub(c,a))   #construit les vecteurs a-b et a-c et effectue produit scalaire
    return 1 if v > eps else (-1 if v < -eps else 0) #renvoit 1 si c à guauche de a-b, -1 si c à droite de a-b, 0 si collinéaire (aligné)
#v>0 alors c à guauche (rotation anti-horaire), v<0 alors c à droite (rotation horaire), v=0 alors a,b,c sont colinéaires (alignés)  


#appartenance à un segment 
def on_segment(p, q, r, eps=1e-9):  #point q est-il sur le segment q-r
    if min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps:    #test de boite englobante, on vérifie si x et y de q compris entre x et y de p et r. on élargit les bornes eps pour éviter de rejeter à cause d'un arrondi
        return abs(cross(sub(q,p), sub(r,p))) <= eps      #test de collinéarité: indique si q est sur le segment p-r (si produit scalire environ égal a 0 alors collinéaire)
    return False  #si q n'est pas dans le rectangle englobant p-r alors il ne peut pas être dans le segment p-r

def segments_intersect(p1, q1, p2, q2, allow_shared_endpoint=True):  #dire si le segment p1–q1 intersecte le segment p2–q2 (au sens “ils se coupent ou se chevauchent”), avec la possibilité d’ignorer le cas où ils ne font que se toucher à un sommet commun
    o1 = orient(p1, q1, p2)   #On évalue l’orientation (gauche/droite/aligné) des quatre triplets nécessaires.o1 et o2 comparent où se trouvent p2 et q2 par rapport à la droite orientée p1→q1, o3 et o4 comparent où se trouvent p1 et q1 par rapport à la droite orientée p2→q2.
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    # proper intersection
    if o1 != o2 and o3 != o4: #si p2 et q2 sont de part et d’autre de p1→q1 (signes différents) ET p1 et q1 sont de part et d’autre de p2→q2, alors les segments se coupent en un point unique à l’intérieur des deux segments.
        return True

    # collinear cases
    #Si allow_shared_endpoint est True (par défaut): revenir False lorsque l’unique contact est exactement un sommet commun (autorisé). Si allow_shared_endpoint est False: un contact au sommet compte comme intersection → on retourne True. Le test (p2 != p1 and p2 != q1) vérifie que le point de contact n’est pas un des deux bouts du premier segment; si c’en est un, et que les endpoints partagés sont autorisés, on ne compte PAS ça comme “intersection bloquante”.

    if o1 == 0 and on_segment(p1, p2, q1):
        return not allow_shared_endpoint or (p2 != p1 and p2 != q1)
    if o2 == 0 and on_segment(p1, q2, q1):
        return not allow_shared_endpoint or (q2 != p1 and q2 != q1)
    if o3 == 0 and on_segment(p2, p1, q2):
        return not allow_shared_endpoint or (p1 != p2 and p1 != q2)
    if o4 == 0 and on_segment(p2, q1, q2):
        return not allow_shared_endpoint or (q1 != p2 and q1 != q2)  #si l’un des points est collinéaire (orientation 0), on vérifie s’il est posé “sur” l’autre segment avec on_segment(...). chevauchement partiel (segments posés l’un sur l’autre), contact à une extrémité (touchent au bout), ou pas de contact (alignés mais disjoints).les segments d’un graphe de visibilité passent souvent exactement par des sommets d’obstacles; on doit accepter un “contact au sommet commun” sinon on rejetterait des arêtes valides.

    return False #si aucun des points précédents, les segments ne s'intersectent pas 

def point_in_polygon(pt, poly):   #dit si un point pt est à l'intérieur d'un polygone (poly=liste de sommets) 
    # ray-cast
    x, y = pt
    inside = False
    n = len(poly)  #nombre de sommets 
    for i in range(n):
        x1, y1 = poly[i]  #calcul des coordonnées des deux extrémités de l'arrête existante
        x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1
            if x < x_int:
                inside = not inside
    return inside

#Si le point (x, y) est à gauche du point d’intersection (le croisement est “devant” le point sur le rayon positif), alors notre rayon croise cette arête. 
#Règle de parité: à chaque croisement du rayon avec une arête, on inverse l’état dedans/dehors. À la fin, si on a croisé un nombre impair d’arêtes, on est dedans; pair → dehors.
#Lorsqu’on teste la visibilité d’un segment entre deux nœuds (sommet d’obstacle, start ou goal), on veut rejeter d’emblée les cas où un des deux points est déjà à l’intérieur d’un obstacle “gonflé”. Ça signifie que la donnée est invalide pour le graphe de visibilité (un nœud ne doit pas être dans un obstacle).

# ---------- Visibility ----------
def segment_hits_polygon(a, b, poly, safety=0.0): #dire si le segment a–b “touche” un polygone P (donc il n’est pas en espace libre). On autorise juste le cas où l’on partage exactement un sommet (pour pouvoir partir/arriver à un coin sans considérer cela comme une collision).
    n = len(poly)
    # Reject if intersects any edge (except at shared endpoints)
    for i in range(n):
        p = poly[i]     #calcul des coordonnées des deux extrémités de l'arrête existante
        q = poly[(i+1) % n]
        allow_endpoint = (a == p or a == q or b == p or b == q)                   #True si notre segment a–b partage un sommet EXACT avec l’arête p–q (a ou b égale p ou q). Si on part/arrive à un coin du polygone, c’est autorisé; on ne veut pas considérer cela comme une “collision”.

        if segments_intersect(a, b, p, q, allow_shared_endpoint=allow_endpoint):  #On teste l’intersection entre notre segment et l’arête du polygone. On passe le flag allow_shared_endpoint:True si on partage un sommet → toucher au sommet commun est permis.False sinon → tout contact compte comme intersection.
            if not allow_endpoint:
                return True
            
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

def visible(a, b, polygons, safety=0.0): # dire si deux points a et b sont “visibles” à travers TOUS les polygones, c.-à-d. si le segment a–b est entièrement en espace libre.
    for P in polygons:
        if point_in_polygon(a, P) or point_in_polygon(b, P): #si a ou b est DANS un obstacle, l’arête est invalide tout de suite
            return False
    for P in polygons:
        if segment_hits_polygon(a, b, P, safety=safety): #si le segment touche n’importe quel polygone, on dit “pas visible”
            return False
    return True #le segment est en espace libre 

# ---------- Graph + A* ----------
def build_visibility_graph(start, goal, polygons, safety=0.0):
    nodes = [start, goal] + [v for poly in polygons for v in poly]   #on rassemble tous les candidats des noeuds (start goal et sommets des aretes)
    # deduplicate
    seen = set()  #deduplication car les memes sommets peuvent apparaitre plusieurs fois si plusieurs polygones partagent le meme 
    uniq = []
    for v in nodes:
        key = (round(v[0], 9), round(v[1], 9))  #arrondi à 9 decimales pour eviter graphes denses et bugs 
        if key not in seen:
            seen.add(key)  #ajoute le point s'il n'a pas ete vu 
            uniq.append(v)
    nodes = uniq   #evite les redondances 

    adj = {v: [] for v in nodes} #Initialise une liste vide de voisins pour chaque nœud-> obtient une paire (voisin,poids)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a, b = nodes[i], nodes[j] #pair de noeud 
            if visible(a, b, polygons, safety=safety): #vérifie que ni a ni b ne sont à l’intérieur d’un obstacle (sanity check),que le segment a–b ne coupe aucune arête de polygone (avec allowance si on partage un sommet),et applique une petite marge safety optionnelle contre les frôlements.
                w = dist(a, b) #poids de l'arete=distance euclidienne
                adj[a].append((b, w)) #a adjacent à b avec cout w
                adj[b].append((a, w))
    return adj #renvoit le graphe de visibilité pour A*

def a_star(adj, start, goal):
    def h(u): return dist(u, goal)  #distance euclidienne de u au but 

    pq = [] #queue that picks best node 
    heapq.heappush(pq, (h(start), 0.0, start)) #start from node start
    g = {start: 0.0} #remembers best cost for each node 
    parent = {start: None} #to reconstruct path = predecessor of each node 
    closed = set() #nodes finalized 
    
    while pq: #if there are nodes to explore 
        f, gcost, u = heapq.heappop(pq) #take node with smallest f and retrieve gcost
        if u in closed: #if we finalized u skip it 
            continue
        if u == goal: #if u is the goal
            # reconstruct->we rebuild path from u down to start 
            path = []  #empty list to accumulate nodes of the path 
            cur = u  #start from goal node 
            while cur is not None: #walk backwards using parent until start (none)
                path.append(cur) #add current node 
                cur = parent[cur] #move to its parent (predecessor) 
            path.reverse() #reverse path as it is from goal to start 
            return path, g[u] #returns the path and its total cost
        closed.add(u) #mark u as done 
        for v, w in adj[u]: #for each neighbor v of u with edge weight w
            ng = gcost + w #new cost to reach v via u
            if v not in g or ng < g[v]: #if v has no cost yet or cost is cheaper
                g[v] = ng #update cost 
                parent[v] = u #best path to v goes through u 
                heapq.heappush(pq, (ng + h(v), ng, v)) #push v into the heap with priority f
    return None, float('inf') #no path, infinite cost 

# ---------- Path cleanup ----------
def simplify_collinear(path, eps=1e-3): #removes unnecessary intermediate points that lie on straight lines 
    if len(path) <= 2:
        return path[:]
    out = [path[0]] #start from start 
    for i in range(1, len(path)-1):
        a, b, c = path[i-1], path[i], path[i+1] #three neighbors 
        if abs(cross((b[0]-a[0], b[1]-a[1]), (c[0]-b[0], c[1]-b[1]))) > eps: #if cross product larger than eps then there is a turn at this node and we need to keep it (a,b,c not colinear) 
            out.append(b)
    out.append(path[-1]) #always keep last point=goal 
    return out

# ---------- Public ----------
def plan_path(start, goal, grown_polygons, safety=0.02):
    """
    start: (x,y) in meters
    goal:  (x,y) in meters
    grown_polygons: list of polygons, each polygon is list[(x,y)], already inflated by robot radius
    safety: extra clearance (m) to keep edges away from boundaries (for vision noise)
    returns: (waypoints list[(x,y)], total_length)
    """
    graph = build_visibility_graph(start, goal, grown_polygons, safety=safety)
    path, L = a_star(graph, start, goal)
    if path is None:
        return None, float('inf')
    return simplify_collinear(path), L


def check_kidnapping(pose, target, path, prev_idx, threshold):
    if not path:
        return False
    p, a, b = np.array(pose[0]), np.array(path[prev_idx]), np.array(target)
    n = b - a
    norm_n = np.linalg.norm(n)
    dist = np.abs(np.cross(n, a - p)) / norm_n if norm_n > 0 else np.linalg.norm(p - a)
    return dist > threshold