import heapq
import itertools
import random
import math
import tkinter as tk

class Coordinate:
    def __init__(self, x_val, y_val):
        self.x = x_val
        self.y = y_val

class CircleEvent:
    def __init__(self, x_pos, vertex, parabola):
        self.x = x_pos
        self.vertex = vertex
        self.parabola = parabola
        self.active = True

class Parabola:
    def __init__(self, focus, left=None, right=None):
        self.focus = focus
        self.left = left
        self.right = right
        self.event = None
        self.left_edge = None
        self.right_edge = None

class VoronoiEdge:
    def __init__(self, start_pt):
        self.start = start_pt
        self.end = None
        self.completed = False

    def complete_edge(self, end_pt):
        if not self.completed:
            self.end = end_pt
            self.completed = True

class EventQueue:
    def __init__(self):
        self.heap = []
        self.entry_map = {}
        self.counter = itertools.count()

    def add_event(self, event):
        if event in self.entry_map: return
        count = next(self.counter)
        entry = [event.x, count, event]
        self.entry_map[event] = entry
        heapq.heappush(self.heap, entry)

    def remove_event(self, event):
        entry = self.entry_map.pop(event)
        entry[-1] = 'INACTIVE'

    def get_next(self):
        while self.heap:
            priority, count, event = heapq.heappop(self.heap)
            if event != 'INACTIVE':
                del self.entry_map[event]
                return event
        raise IndexError('Queue is empty')

    def peek_next(self):
        while self.heap:
            priority, count, event = heapq.heappop(self.heap)
            if event != 'INACTIVE':
                del self.entry_map[event]
                self.add_event(event)
                return event
        raise IndexError('Queue is empty')

    def has_events(self):
        return bool(self.heap)

class VoronoiDiagram:
    def __init__(self, sites):
        self.edges = []
        self.beachline = None
        self.site_events = EventQueue()
        self.circle_events = EventQueue()
        
        # Initialize boundaries
        self.min_x = -50.0
        self.max_x = -50.0
        self.min_y = 550.0
        self.max_y = 550.0

        # Process input sites
        for site in sites:
            pt = Coordinate(site[0], site[1])
            self.site_events.add_event(pt)
            # Update boundaries
            self.min_x = min(self.min_x, pt.x)
            self.min_y = min(self.min_y, pt.y)
            self.max_x = max(self.max_x, pt.x)
            self.max_y = max(self.max_y, pt.y)

        # Add padding to boundaries
        pad_x = (self.max_x - self.min_x + 1) / 5.0
        pad_y = (self.max_y - self.min_y + 1) / 5.0
        self.min_x -= pad_x
        self.max_x += pad_x
        self.min_y -= pad_y
        self.max_y += pad_y

    def generate(self):
        while self.site_events.has_events():
            if (self.circle_events.has_events() and 
                self.circle_events.peek_next().x <= self.site_events.peek_next().x):
                self.handle_circle_event()
            else:
                self.handle_site_event()

        # Process remaining circle events
        while self.circle_events.has_events():
            self.handle_circle_event()

        self.complete_unfinished_edges()

    def handle_site_event(self):
        site = self.site_events.get_next()
        self.add_parabola(site)

    def handle_circle_event(self):
        event = self.circle_events.get_next()
        if event.active:
            new_edge = VoronoiEdge(event.vertex)
            self.edges.append(new_edge)
            parabola = event.parabola
            
            # Update parabola neighbors
            if parabola.left:
                parabola.left.right = parabola.right
                parabola.left.right_edge = new_edge
            if parabola.right:
                parabola.right.left = parabola.left
                parabola.right.left_edge = new_edge

            # Complete adjacent edges
            if parabola.left_edge: parabola.left_edge.complete_edge(event.vertex)
            if parabola.right_edge: parabola.right_edge.complete_edge(event.vertex)

            # Check new potential circle events
            if parabola.left: self.detect_circle_event(parabola.left, event.x)
            if parabola.right: self.detect_circle_event(parabola.right, event.x)

    def add_parabola(self, site):
        if not self.beachline:
            self.beachline = Parabola(site)
            return
            
        current = self.beachline
        while current:
            has_intersection, intersection = self.find_intersection(site, current)
            if has_intersection:
                has_second_intersection, _ = self.find_intersection(site, current.right)
                if current.right and not has_second_intersection:
                    current.right.left = Parabola(current.focus, current, current.right)
                    current.right = current.right.left
                else:
                    current.right = Parabola(current.focus, current)
                current.right.right_edge = current.right_edge

                # Insert new parabola between current and current.right
                current.right.left = Parabola(site, current, current.right)
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
                self.detect_circle_event(current, site.x)
                self.detect_circle_event(current.left, site.x)
                self.detect_circle_event(current.right, site.x)
                return
            current = current.right

        # Add to end if no intersections found
        current = self.beachline
        while current.right:
            current = current.right
        current.right = Parabola(site, current)
        
        # Create edge between last two parabolas
        x = self.min_x
        y = (current.right.focus.y + current.focus.y) / 2.0
        start = Coordinate(x, y)
        edge = VoronoiEdge(start)
        current.right_edge = current.right.left_edge = edge
        self.edges.append(edge)

    def detect_circle_event(self, parabola, x_limit):
        if parabola.event and parabola.event.x != self.min_x:
            parabola.event.active = False
        parabola.event = None

        if not parabola.left or not parabola.right: return

        valid, x, center = self.compute_circle_center(
            parabola.left.focus, parabola.focus, parabola.right.focus)
        if valid and x > x_limit:
            parabola.event = CircleEvent(x, center, parabola)
            self.circle_events.add_event(parabola.event)

    def compute_circle_center(self, a, b, c):
        # Check for counter-clockwise turn
        cross = (b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y)
        if cross > 0: return False, None, None

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
        center = Coordinate(center_x, center_y)

        radius = math.sqrt((a.x-center_x)**2 + (a.y-center_y)**2)
        rightmost_x = center_x + radius
        
        return True, rightmost_x, center

    def find_intersection(self, site, parabola):
        if not parabola: return False, None
        if parabola.focus.x == site.x: return False, None

        left_bound = right_bound = float('inf')
        if parabola.left:
            left_bound = self.get_parabola_intersection(
                parabola.left.focus, parabola.focus, site.x).y
        if parabola.right:
            right_bound = self.get_parabola_intersection(
                parabola.focus, parabola.right.focus, site.x).y

        if ((not parabola.left or left_bound <= site.y) and 
            (not parabola.right or site.y <= right_bound)):
            intersect_y = site.y
            intersect_x = ((parabola.focus.x**2 + (parabola.focus.y-intersect_y)**2 - site.x**2) 
                         / (2*parabola.focus.x - 2*site.x))
            return True, Coordinate(intersect_x, intersect_y)
        return False, None

    def get_parabola_intersection(self, p1, p2, sweep_line):
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
        return Coordinate(mid_x, mid_y)

    def complete_unfinished_edges(self):
        if not self.beachline: return
        
        sweep_limit = self.max_x + (self.max_x - self.min_x) + (self.max_y - self.min_y)
        current = self.beachline
        while current.right:
            if current.right_edge:
                end_pt = self.get_parabola_intersection(
                    current.focus, current.right.focus, sweep_limit * 2.0)
                current.right_edge.complete_edge(end_pt)
            current = current.right

    def get_edges(self):
        return [(edge.start.x, edge.start.y, 
                 edge.end.x if edge.end else None, 
                 edge.end.y if edge.end else None) 
                for edge in self.edges]

class VoronoiVisualizer:
    POINT_SIZE = 3
    GRID_STEP = 50
    CANVAS_DIM = 500
    ORIGIN = CANVAS_DIM // 2

    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi Visualization")
        
        self.canvas = tk.Canvas(root, width=self.CANVAS_DIM, height=self.CANVAS_DIM, bg='white')
        self.canvas.pack()
        
        self.sites = []
        self.scale_factor = 1
        
        self.load_sites("input.txt")
        self.canvas.bind('<Double-1>', self.handle_click)
        self.render_diagram()

    def draw_grid(self):
        """Draws grid lines and coordinate axes"""
        for x in range(0, self.CANVAS_DIM + 1, self.GRID_STEP):
            self.canvas.create_line(x, 0, x, self.CANVAS_DIM, fill="lightgray")
        for y in range(0, self.CANVAS_DIM + 1, self.GRID_STEP):
            self.canvas.create_line(0, y, self.CANVAS_DIM, y, fill="lightgray")
        
        # Draw axes
        self.canvas.create_line(0, self.ORIGIN, self.CANVAS_DIM, self.ORIGIN, fill="black", width=2)
        self.canvas.create_line(self.ORIGIN, 0, self.ORIGIN, self.CANVAS_DIM, fill="black", width=2)

    def calculate_scale(self):
        """Determines appropriate scale to fit all points"""
        if not self.sites: return 1
        max_coord = max(max(abs(x), abs(y)) for x, y in self.sites)
        return (self.CANVAS_DIM / 2 * 0.9) / max_coord if max_coord else 1

    def convert_coords(self, x, y):
        """Converts mathematical coordinates to canvas coordinates"""
        return self.ORIGIN + x * self.scale_factor, self.ORIGIN - y * self.scale_factor

    def load_sites(self, filename):
        """Loads points from file"""
        try:
            with open(filename) as f:
                for line in f:
                    x, y = map(int, line.strip().split())
                    self.sites.append((x, y))
        except FileNotFoundError:
            print(f"File {filename} not found")

    def handle_click(self, event):
        """Adds new point on double-click"""
        math_x = (event.x - self.ORIGIN) / self.scale_factor
        math_y = (self.ORIGIN - event.y) / self.scale_factor
        self.sites.append((math_x, math_y))
        self.render_diagram()

    def render_diagram(self):
        """Redraws the entire diagram"""
        self.scale_factor = self.calculate_scale()
        self.canvas.delete("all")
        self.draw_grid()
        
        # Draw sites
        for x, y in self.sites:
            canvas_x, canvas_y = self.convert_coords(x, y)
            self.canvas.create_oval(
                canvas_x - self.POINT_SIZE, canvas_y - self.POINT_SIZE,
                canvas_x + self.POINT_SIZE, canvas_y + self.POINT_SIZE, 
                fill="black")
        
        # Generate and draw Voronoi edges if enough points
        if len(self.sites) > 1:
            diagram = VoronoiDiagram(self.sites)
            diagram.generate()
            edges = diagram.get_edges()
            for x1, y1, x2, y2 in edges:
                if x2 is not None and y2 is not None:
                    start = self.convert_coords(x1, y1)
                    end = self.convert_coords(x2, y2)
                    self.canvas.create_line(start[0], start[1], end[0], end[1], fill='blue')

def main():
    window = tk.Tk()
    app = VoronoiVisualizer(window)
    window.mainloop()

if __name__ == '__main__':
    main()