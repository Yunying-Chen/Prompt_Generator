import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import KDTree


def generate_candidate_points(polygon_hull):
    candidates = []
    x_min, y_min, x_max, y_max = polygon_hull.bounds
    for i in range(0,2000):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)
        if polygon_hull.contains(point):
            candidates.append(point)
    
    return candidates

def evaluate_coverage(candidates, polygon_hull, radius):
    scored_candidates = []
    for point in candidates:
        circle = point.buffer(radius)
        intersection_area = polygon_hull.intersection(circle).area
        scored_candidates.append((intersection_area, point))
    
    return sorted(scored_candidates, reverse=True, key=lambda x: x[0])

def select_circle_centers(scored_candidates, n, radius):
    selected_centers = []
    
    for _, point in scored_candidates:
        if not selected_centers:  # First selection, no need to check distance
            selected_centers.append(point)
        else:
            used_tree = KDTree([p.coords[0] for p in selected_centers])
            if used_tree.query(point.coords[0], k=1, distance_upper_bound=2 * radius)[0] == np.inf:
                selected_centers.append(point)

        if len(selected_centers) >= n:
            break
    
    return selected_centers

def find_optimal_circle_centers(polygon, n, radius):
    candidates = generate_candidate_points(polygon)
    scored_candidates = evaluate_coverage(candidates, polygon, radius)
    centers = select_circle_centers(scored_candidates, n, radius)
    
    return [center.coords[0] for center in centers]


def evaluate_coverage_points(candidates, points_to_cover, radius):
    if len(points_to_cover) < 1:
        return []
    
    # Convert points_to_cover into a numpy array of points (2D)
    points_array = np.array([(p[0], p[1]) for p in points_to_cover])
    
    # Create the KDTree from the points
    tree = KDTree(points_array)
    
    scored_candidates = []
    for point in candidates:
        # Query the KDTree to get indices of points within the radius
        indices = tree.query_ball_point([point.x, point.y], radius)
        
        # Append the count of points within the radius and the candidate point
        scored_candidates.append((len(indices), point))
    
    # Sort candidates by the number of covered points in descending order
    return sorted(scored_candidates, reverse=True, key=lambda x: x[0])

def select_circle_centers_points(scored_candidates, n, radius):
    selected_centers = []
    #used_tree = KDTree([])
    
    for count, point in scored_candidates:
        if len(selected_centers) == 0:
            selected_centers.append(point)
            used_tree = KDTree([point.coords[0]])
        else:
            distance, _ = used_tree.query(point.coords[0], k=1, distance_upper_bound=2*radius)
            if distance > 2 * radius:
                selected_centers.append(point)
                coords = [p.coords[0] for p in selected_centers]
                used_tree = KDTree(coords)
        if len(selected_centers) >= n:
            break
    return selected_centers

def find_optimal_circle_centers_points(polygon, points_to_cover, n, radius):
    candidates = generate_candidate_points(polygon)
    scored_candidates = evaluate_coverage_points(candidates, points_to_cover, radius)
    centers = select_circle_centers_points(scored_candidates, n, radius)
    return [center.coords[0] for center in centers]

def PointGenerator(polygon_hull,class_points=None,mode='points'):
        valid_points = []
        times = 0
        while len(valid_points)<1:
            x_min, y_min, x_max, y_max = polygon_hull.bounds
            x_min = x_min 
            y_min = y_min 
            x_max = x_max 
            y_max = y_max 
            
            #Circle cover
            #print(polygon_hull.area)
            if polygon_hull.area >1000000:
                radius = 150  # Circle radius
                n =8
            elif polygon_hull.area >500000:
                n =5
                radius = 150  # Circle radius
            elif polygon_hull.area >100000:
                n =3
                radius = 100  # Circle radius
            else :
                n=2
                radius = 90  # Circle radius

            if mode == 'points':
                valid_points = find_optimal_circle_centers_points(polygon_hull,class_points, n, radius)
            else:
                 valid_points = find_optimal_circle_centers(polygon_hull, n, radius)
            times +=1
            if times>10:
              return None
        return valid_points
