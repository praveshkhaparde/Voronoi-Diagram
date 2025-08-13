import heapq
import itertools
import math
import tkinter as tk
from tkinter import messagebox


# Enhanced Priority Queue

class MinHeap:
    def __init__(self):
        self.heap_data = []
        self.entry_map = {}
        self.id_generator = itertools.count()

    def enqueue(self, element, priority):
        if element in self.entry_map:
            return
        unique_id = next(self.id_generator)
        entry = [priority, unique_id, element]
        self.entry_map[element] = entry
        heapq.heappush(self.heap_data, entry)

    def deactivate(self, element):
        entry = self.entry_map.pop(element)
        entry[-1] = 'INVALID'

    def extract_min(self):
        while self.heap_data:
            priority, uid, element = heapq.heappop(self.heap_data)
            if element != 'INVALID':
                del self.entry_map[element]
                return element
        raise IndexError('Extracting from empty heap')

    def peek_min(self):
        while self.heap_data and self.heap_data[0][-1] == 'INVALID':
            heapq.heappop(self.heap_data)
        if self.heap_data:
            return self.heap_data[0][-1]
        raise IndexError('Peeking empty heap')

    def is_empty(self):
        self._clean_invalid()
        return not self.heap_data

    def _clean_invalid(self):
        while self.heap_data and self.heap_data[0][-1] == 'INVALID':
            heapq.heappop(self.heap_data)


# Computational Geometry Classes

class Vector2D:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

class SiteEvent:
    def __init__(self, point):
        self.point = point
        self.priority = point.x  # For site events, priority is x-coordinate

class CircleEvent:
    def __init__(self, x_pos, vertex, arc):
        self.priority = x_pos  # For circle events, priority is x-coordinate
        self.vertex = vertex
        self.arc = arc
        self.is_active = True

class BeachlineArc:
    def __init__(self, focus, left_arc=None, right_arc=None):
        self.focus = focus
        self.left = left_arc
        self.right = right_arc
        self.left_edge = None
        self.right_edge = None
        self.circle_event = None

class VoronoiEdge:
    def __init__(self, origin):
        self.origin = origin
        self.terminus = None
        self.is_complete = False

    def complete(self, terminus):
        if not self.is_complete:
            self.terminus = terminus
            self.is_complete = True


# Voronoi Diagram Generator

class VoronoiBuilder:
    def __init__(self, sites):
        self.edges = []
        self.beachline = None
        self.site_queue = MinHeap()
        self.event_queue = MinHeap()
        
        # Initialize boundaries
        self.min_x = float('inf')
        self.max_x = -float('inf')
        self.min_y = float('inf')
        self.max_y = -float('inf')

        # Process input sites
        for site in sites:
            point = Vector2D(site[0], site[1])
            self.site_queue.enqueue(SiteEvent(point), point.x)
            # Update boundaries
            self.min_x = min(self.min_x, point.x)
            self.min_y = min(self.min_y, point.y)
            self.max_x = max(self.max_x, point.x)
            self.max_y = max(self.max_y, point.y)

        # Add padding to boundaries
        padding_x = (self.max_x - self.min_x + 1) / 5.0
        padding_y = (self.max_y - self.min_y + 1) / 5.0
        self.min_x -= padding_x
        self.max_x += padding_x
        self.min_y -= padding_y
        self.max_y += padding_y

    def construct_diagram(self):
        while not self.site_queue.is_empty():
            if (not self.event_queue.is_empty() and 
                self.event_queue.peek_min().priority <= self.site_queue.peek_min().priority):
                self._handle_circle_event()
            else:
                self._handle_site_event()

        # Process remaining circle events
        while not self.event_queue.is_empty():
            self._handle_circle_event()

        self._finalize_edges()

    def _handle_site_event(self):
        event = self.site_queue.extract_min()
        self._insert_parabola(event.point)

    def _handle_circle_event(self):
        event = self.event_queue.extract_min()
        if event.is_active:
            new_edge = VoronoiEdge(event.vertex)
            self.edges.append(new_edge)
            arc = event.arc
            
            # Update arc neighbors
            if arc.left:
                arc.left.right = arc.right
                arc.left.right_edge = new_edge
            if arc.right:
                arc.right.left = arc.left
                arc.right.left_edge = new_edge

            # Complete adjacent edges
            if arc.left_edge: arc.left_edge.complete(event.vertex)
            if arc.right_edge: arc.right_edge.complete(event.vertex)

            # Check for new circle events
            if arc.left: self._detect_circle_event(arc.left, event.priority)
            if arc.right: self._detect_circle_event(arc.right, event.priority)

    def _insert_parabola(self, site):
        if not self.beachline:
            self.beachline = BeachlineArc(site)
            return
            
        current = self.beachline
        while current:
            intersects, intersection = self._find_parabola_intersection(site, current)
            if intersects:
                intersects_next, _ = self._find_parabola_intersection(site, current.right)
                if current.right and not intersects_next:
                    current.right.left = BeachlineArc(current.focus, current, current.right)
                    current.right = current.right.left
                else:
                    current.right = BeachlineArc(current.focus, current)
                current.right.right_edge = current.right_edge

                # Insert new parabola between current and current.right
                current.right.left = BeachlineArc(site, current, current.right)
                current.right = current.right.left
                current = current.right  # Move to new parabola

                # Create new edges
                edge1 = VoronoiEdge(intersection)
                self.edges.append(edge1)
                current.left.right_edge = current.left_edge = edge1

                edge2 = VoronoiEdge(intersection)
                self.edges.append(edge2)
                current.right.left_edge = current.right_edge = edge2

                # Check for new circle events
                self._detect_circle_event(current, site.x)
                self._detect_circle_event(current.left, site.x)
                self._detect_circle_event(current.right, site.x)
                return
            current = current.right

        # Add to end if no intersections found
        current = self.beachline
        while current.right:
            current = current.right
        current.right = BeachlineArc(site, current)
        
        # Create edge between last two parabolas
        x = self.min_x
        y = (current.right.focus.y + current.focus.y) / 2.0
        start = Vector2D(x, y)
        edge = VoronoiEdge(start)
        current.right_edge = current.right.left_edge = edge
        self.edges.append(edge)

    def _detect_circle_event(self, arc, x_limit):
        if arc.circle_event and arc.circle_event.priority != self.min_x:
            arc.circle_event.is_active = False
        arc.circle_event = None

        if not arc.left or not arc.right: return

        is_valid, x, center = self._compute_circle_center(
            arc.left.focus, arc.focus, arc.right.focus)
        if is_valid and x > x_limit:
            arc.circle_event = CircleEvent(x, center, arc)
            self.event_queue.enqueue(arc.circle_event, x)

    def _compute_circle_center(self, a, b, c):
        # Check for counter-clockwise turn
        cross_product = (b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y)
        if cross_product > 0: return False, None, None

        # Circle center calculation
        A = b.x - a.x
        B = b.y - a.y
        C = c.x - a.x
        D = c.y - a.y
        E = A*(a.x + b.x) + B*(a.y + b.y)
        F = C*(a.x + c.x) + D*(a.y + c.y)
        G = 2*(A*(c.y - b.y) - B*(c.x - b.x))
        if G == 0: return False, None, None  # Colinear points

        center_x = (D*E - B*F) / G
        center_y = (A*F - C*E) / G
        center = Vector2D(center_x, center_y)

        radius = math.sqrt((a.x-center_x)**2 + (a.y-center_y)**2)
        rightmost_x = center_x + radius
        
        return True, rightmost_x, center

    def _find_parabola_intersection(self, site, parabola):
        if not parabola: return False, None
        if parabola.focus.x == site.x: return False, None

        upper_bound = lower_bound = float('inf')
        if parabola.left:
            upper_bound = self._get_intersection_point(
                parabola.left.focus, parabola.focus, site.x).y
        if parabola.right:
            lower_bound = self._get_intersection_point(
                parabola.focus, parabola.right.focus, site.x).y

        if ((not parabola.left or upper_bound <= site.y) and 
            (not parabola.right or site.y <= lower_bound)):
            intersect_y = site.y
            intersect_x = ((parabola.focus.x**2 + (parabola.focus.y-intersect_y)**2 - site.x**2) 
                          / (2*parabola.focus.x - 2*site.x))
            return True, Vector2D(intersect_x, intersect_y)
        return False, None

    def _get_intersection_point(self, p1, p2, sweep_line):
        if p1.x == p2.x:
            mid_y = (p1.y + p2.y) / 2.0
        elif p2.x == sweep_line:
            mid_y = p2.y
        elif p1.x == sweep_line:
            mid_y = p1.y
            p1, p2 = p2, p1
        else:
            # Quadratic equation solution
            denom1 = 2.0 * (p1.x - sweep_line)
            denom2 = 2.0 * (p2.x - sweep_line)
            a = 1.0/denom1 - 1.0/denom2
            b = -2.0 * (p1.y/denom1 - p2.y/denom2)
            c = ((p1.y**2 + p1.x**2 - sweep_line**2) / denom1 
                - (p2.y**2 + p2.x**2 - sweep_line**2) / denom2)
            mid_y = (-b - math.sqrt(b*b - 4*a*c)) / (2*a)
        mid_x = (p1.x**2 + (p1.y-mid_y)**2 - sweep_line**2) / (2*p1.x - 2*sweep_line)
        return Vector2D(mid_x, mid_y)

    def _finalize_edges(self):
        if not self.beachline: return
        
        large_value = self.max_x + (self.max_x - self.min_x) + (self.max_y - self.min_y)
        current = self.beachline
        while current.right:
            if current.right_edge:
                end_point = self._get_intersection_point(
                    current.focus, current.right.focus, large_value * 2.0)
                current.right_edge.complete(end_point)
            current = current.right

    def get_edges(self):
        return [(edge.origin.x, edge.origin.y, 
                 edge.terminus.x if edge.terminus else None, 
                 edge.terminus.y if edge.terminus else None) 
                for edge in self.edges]


# Spatial Partitioning Structure

class Cell:
    def __init__(self, x_min, x_max, y_min, y_max, nearest_site=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.nearest_site = nearest_site

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

class SpatialPartition:
    def __init__(self, edges, sites, bounds):
        self.cells = self._build_partition(bounds, sites)

    def _build_partition(self, bounds, sites):
        x0, x1, y0, y1 = bounds
        cells = []
        grid_size = 10
        dx = (x1 - x0) / grid_size
        dy = (y1 - y0) / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                cell_x_min = x0 + i * dx
                cell_x_max = x0 + (i+1) * dx
                cell_y_min = y0 + j * dy
                cell_y_max = y0 + (j+1) * dy
                center_x = (cell_x_min + cell_x_max) / 2
                center_y = (cell_y_min + cell_y_max) / 2
                closest_site = None
                min_distance = float('inf')
                for site in sites:
                    distance = (site[0]-center_x)**2 + (site[1]-center_y)**2
                    if distance < min_distance:
                        closest_site = site
                        min_distance = distance
                cells.append(Cell(cell_x_min, cell_x_max, cell_y_min, cell_y_max, closest_site))
        return cells

    def find_cell(self, x, y):
        for cell in self.cells:
            if cell.contains(x, y):
                return cell
        return None


# Interactive Visualization

class VoronoiVisualizer:
    POINT_RADIUS = 3
    GRID_SIZE = 50
    VIEWPORT_SIZE = 500
    ORIGIN = VIEWPORT_SIZE // 2

    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Voronoi Diagram Visualizer")
        
        self.canvas = tk.Canvas(self.root, width=self.VIEWPORT_SIZE, 
                               height=self.VIEWPORT_SIZE, bg='white')
        self.canvas.pack()
        
        self.query_button = tk.Button(root_window, text="Find Nearest Site", 
                                    command=self.find_nearest_site)
        self.query_button.pack()
        
        self.sites = []  # Stored in mathematical coordinates
        self.view_scale = 1
        
        self.load_sites("input.txt")
        self.canvas.bind('<Double-1>', self.handle_double_click)
        self.redraw_diagram()

    def draw_grid(self):
        for x in range(0, self.VIEWPORT_SIZE + 1, self.GRID_SIZE):
            self.canvas.create_line(x, 0, x, self.VIEWPORT_SIZE, fill="lightgray")
        for y in range(0, self.VIEWPORT_SIZE + 1, self.GRID_SIZE):
            self.canvas.create_line(0, y, self.VIEWPORT_SIZE, y, fill="lightgray")
        self.canvas.create_line(0, self.ORIGIN, self.VIEWPORT_SIZE, self.ORIGIN, 
                               fill="black", width=2)
        self.canvas.create_line(self.ORIGIN, 0, self.ORIGIN, self.VIEWPORT_SIZE, 
                               fill="black", width=2)

    def calculate_scale(self):
        if not self.sites: return 1
        max_coord = max(max(abs(x), abs(y)) for x, y in self.sites)
        return (self.VIEWPORT_SIZE / 2 * 0.9) / max_coord if max_coord else 1

    def to_canvas_coords(self, x, y):
        return self.ORIGIN + x * self.view_scale, self.ORIGIN - y * self.view_scale

    def load_sites(self, filename):
        try:
            with open(filename) as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    self.sites.append((x, y))
        except FileNotFoundError:
            print(f"File {filename} not found")

    def handle_double_click(self, event):
        math_x = (event.x - self.ORIGIN) / self.view_scale
        math_y = (self.ORIGIN - event.y) / self.view_scale
        self.sites.append((math_x, math_y))
        self.redraw_diagram()

    def redraw_diagram(self):
        self.view_scale = self.calculate_scale()
        self.canvas.delete("all")
        self.draw_grid()
        
        # Draw sites
        for x, y in self.sites:
            canvas_x, canvas_y = self.to_canvas_coords(x, y)
            self.canvas.create_oval(
                canvas_x - self.POINT_RADIUS, canvas_y - self.POINT_RADIUS,
                canvas_x + self.POINT_RADIUS, canvas_y + self.POINT_RADIUS, 
                fill="black")
        
        # Generate and draw Voronoi edges if enough points
        if len(self.sites) > 1:
            diagram = VoronoiBuilder(self.sites)
            diagram.construct_diagram()
            edges = diagram.get_edges()
            for x1, y1, x2, y2 in edges:
                if x2 is not None and y2 is not None:
                    start = self.to_canvas_coords(x1, y1)
                    end = self.to_canvas_coords(x2, y2)
                    self.canvas.create_line(start[0], start[1], end[0], end[1], 
                                          fill='blue')

    def find_nearest_site(self):
        try:
            with open("query_point.txt") as f:
                query_points = []
                for line in f:
                    line = line.strip()
                    if line:
                        qx, qy = map(float, line.split())
                        query_points.append((qx, qy))
                
                if not query_points:
                    messagebox.showinfo("Result", "No query points found in file")
                    return
                
                # Clear previous highlights
                self.canvas.delete("query")
                
                xs = [p[0] for p in self.sites]
                ys = [p[1] for p in self.sites]
                bounds = (min(xs), max(xs), min(ys), max(ys))
                padding = (max(xs) - min(xs) + 1) / 5.0
                padded_bounds = (
                    bounds[0] - padding, bounds[1] + padding,
                    bounds[2] - padding, bounds[3] + padding
                )

                partition = SpatialPartition([], self.sites, padded_bounds)
                results = []
                
                # Process all query points
                for i, (qx, qy) in enumerate(query_points):
                    # Draw query point (using different colors)
                    color = ["red", "green", "blue", "orange", "purple"][i % 5]
                    qx_canvas, qy_canvas = self.to_canvas_coords(qx, qy)
                    self.canvas.create_oval(
                        qx_canvas - self.POINT_RADIUS, qy_canvas - self.POINT_RADIUS,
                        qx_canvas + self.POINT_RADIUS, qy_canvas + self.POINT_RADIUS,
                        fill=color, outline="black", tags="query")
                    
                    cell = partition.find_cell(qx, qy)
                    if cell:
                        nearest = cell.nearest_site
                        # Draw corresponding site with matching color
                        cx, cy = self.to_canvas_coords(nearest[0], nearest[1])
                        self.canvas.create_oval(
                            cx - self.POINT_RADIUS - 2, cy - self.POINT_RADIUS - 2,
                            cx + self.POINT_RADIUS + 2, cy + self.POINT_RADIUS + 2,
                            outline=color, width=2, tags="query")
                        
                        # Draw line connecting query point to its site
                        self.canvas.create_line(
                            qx_canvas, qy_canvas, cx, cy,
                            fill=color, dash=(3,3), tags="query")
                        
                        results.append(f"Query {i+1}: ({qx:.2f}, {qy:.2f}) → Site: ({nearest[0]:.2f}, {nearest[1]:.2f})")
                    else:
                        results.append(f"Query {i+1}: ({qx:.2f}, {qy:.2f}) → Outside bounds")

                messagebox.showinfo("Results", "\n".join(results))

        except FileNotFoundError:
            messagebox.showinfo("Error", "query_point.txt not found")
        except ValueError:
            messagebox.showinfo("Error", "Invalid query point format in file")

def main():
    root = tk.Tk()
    app = VoronoiVisualizer(root)
    root.mainloop()

if __name__ == '__main__':
    main()