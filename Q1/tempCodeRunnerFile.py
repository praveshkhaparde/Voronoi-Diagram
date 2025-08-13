import heapq
import itertools
import random
import math
import tkinter as tk

import itertools
import heapq

import itertools
import heapq

class Point:
    x = 0.0
    y = 0.0
    
    def __init__(self, x, y):
        # Initializes a Point with given x and y coordinates
        self.x = x
        self.y = y

class Event:
    x = 0.0
    p = None
    a = None
    valid = True
    
    def __init__(self, x, p, a):
        # Initializes an Event with x-coordinate, point reference, and arc reference
        self.x = x
        self.p = p
        self.a = a
        self.valid = True

class Arc:
    p = None
    pprev = None
    pnext = None
    e = None
    s0 = None
    s1 = None
    
    def __init__(self, p, a=None, b=None):
        # Initializes an Arc with a point and optional previous and next arcs
        self.p = p
        self.pprev = a
        self.pnext = b
        self.e = None
        self.s0 = None
        self.s1 = None

class Segment:
    start = None
    end = None
    done = False
    
    def __init__(self, p):
        # Initializes a Segment starting from a given point
        self.start = p
        self.end = None
        self.done = False

    def finish(self, p):
        # Marks the segment as complete by setting its end point
        if self.done: return
        self.end = p
        self.done = True        

class PriorityQueue:
    def __init__(self):
        # Initializes an empty priority queue
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def push(self, item):
        # Adds an item to the priority queue if not already present
        if item in self.entry_finder: return
        count = next(self.counter)
        entry = [item.x, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_entry(self, item):
        # Marks an item as removed from the priority queue
        entry = self.entry_finder.pop(item)
        entry[-1] = 'Removed'

    def pop(self):
        # Removes and returns the highest priority item from the queue
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def top(self):
        # Returns the highest priority item without removing it
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                self.push(item)
                return item
        raise KeyError('top from an empty priority queue')

    def empty(self):
        # Checks if the priority queue is empty
        return not self.pq



class Voronoi:
    def __init__(self, points):
        self.output = [] # list of line segment
        self.arc = None  # binary tree for parabola arcs

        self.points = PriorityQueue() # site events
        self.event = PriorityQueue() # circle events

        # bounding box
        self.x0 = -50.0
        self.x1 = -50.0
        self.y0 = 550.0
        self.y1 = 550.0

        # insert points to site event
        for pts in points:
            point = Point(pts[0], pts[1])
            self.points.push(point)
            # keep track of bounding box size
            if point.x < self.x0: self.x0 = point.x
            if point.y < self.y0: self.y0 = point.y
            if point.x > self.x1: self.x1 = point.x
            if point.y > self.y1: self.y1 = point.y

        # add margins to the bounding box
        dx = (self.x1 - self.x0 + 1) / 5.0
        dy = (self.y1 - self.y0 + 1) / 5.0
        self.x0 = self.x0 - dx
        self.x1 = self.x1 + dx
        self.y0 = self.y0 - dy
        self.y1 = self.y1 + dy

    def process(self):
        while not self.points.empty():
            if not self.event.empty() and (self.event.top().x <= self.points.top().x):
                self.process_event() # handle circle event
            else:
                self.process_point() # handle site event

        # after all points, process remaining circle events
        while not self.event.empty():
            self.process_event()

        self.finish_edges()

    def process_point(self):
        # get next event from site pq
        p = self.points.pop()
        # add new arc (parabola)
        self.arc_insert(p)

    def process_event(self):
        # get next event from circle pq
        e = self.event.pop()

        if e.valid:
            # start new edge
            s = Segment(e.p)
            self.output.append(s)

            # remove associated arc (parabola)
            a = e.a
            if a.pprev is not None:
                a.pprev.pnext = a.pnext
                a.pprev.s1 = s
            if a.pnext is not None:
                a.pnext.pprev = a.pprev
                a.pnext.s0 = s

            # finish the edges before and after a
            if a.s0 is not None: a.s0.finish(e.p)
            if a.s1 is not None: a.s1.finish(e.p)

            # recheck circle events on either side of p
            if a.pprev is not None: self.check_circle_event(a.pprev, e.x)
            if a.pnext is not None: self.check_circle_event(a.pnext, e.x)

    def arc_insert(self, p):
        if self.arc is None:
            self.arc = Arc(p)
        else:
            # find the current arcs at p.y
            i = self.arc
            while i is not None:
                flag, z = self.intersect(p, i)
                if flag:
                    # new parabola intersects arc i
                    flag, zz = self.intersect(p, i.pnext)
                    if (i.pnext is not None) and (not flag):
                        i.pnext.pprev = Arc(i.p, i, i.pnext)
                        i.pnext = i.pnext.pprev
                    else:
                        i.pnext = Arc(i.p, i)
                    i.pnext.s1 = i.s1

                    # add p between i and i.pnext
                    i.pnext.pprev = Arc(p, i, i.pnext)
                    i.pnext = i.pnext.pprev

                    i = i.pnext # now i points to the new arc

                    # add new half-edges connected to i's endpoints
                    seg = Segment(z)
                    self.output.append(seg)
                    i.pprev.s1 = i.s0 = seg

                    seg = Segment(z)
                    self.output.append(seg)
                    i.pnext.s0 = i.s1 = seg

                    # check for new circle events around the new arc
                    self.check_circle_event(i, p.x)
                    self.check_circle_event(i.pprev, p.x)
                    self.check_circle_event(i.pnext, p.x)

                    return
                        
                i = i.pnext

            # if p never intersects an arc, append it to the list
            i = self.arc
            while i.pnext is not None:
                i = i.pnext
            i.pnext = Arc(p, i)
            
            # insert new segment between p and i
            x = self.x0
            y = (i.pnext.p.y + i.p.y) / 2.0;
            start = Point(x, y)

            seg = Segment(start)
            i.s1 = i.pnext.s0 = seg
            self.output.append(seg)

    def check_circle_event(self, i, x0):
        # look for a new circle event for arc i
        if (i.e is not None) and (i.e.x  != self.x0):
            i.e.valid = False
        i.e = None

        if (i.pprev is None) or (i.pnext is None): return

        flag, x, o = self.circle(i.pprev.p, i.p, i.pnext.p)
        if flag and (x > self.x0):
            i.e = Event(x, o, i)
            self.event.push(i.e)

    def circle(self, a, b, c):
        # check if bc is a "right turn" from ab
        if ((b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y)) > 0: return False, None, None

        # Joseph O'Rourke, Computational Geometry in C (2nd ed.) p.189
        A = b.x - a.x
        B = b.y - a.y
        C = c.x - a.x
        D = c.y - a.y
        E = A*(a.x + b.x) + B*(a.y + b.y)
        F = C*(a.x + c.x) + D*(a.y + c.y)
        G = 2*(A*(c.y - b.y) - B*(c.x - b.x))

        if (G == 0): return False, None, None # Points are co-linear

        # point o is the center of the circle
        ox = 1.0 * (D*E - B*F) / G
        oy = 1.0 * (A*F - C*E) / G

        # o.x plus radius equals max x coord
        x = ox + math.sqrt((a.x-ox)**2 + (a.y-oy)**2)
        o = Point(ox, oy)
           
        return True, x, o
        
    def intersect(self, p, i):
        # check whether a new parabola at point p intersect with arc i
        if (i is None): return False, None
        if (i.p.x == p.x): return False, None

        a = 0.0
        b = 0.0

        if i.pprev is not None:
            a = (self.intersection(i.pprev.p, i.p, 1.0*p.x)).y
        if i.pnext is not None:
            b = (self.intersection(i.p, i.pnext.p, 1.0*p.x)).y

        if (((i.pprev is None) or (a <= p.y)) and ((i.pnext is None) or (p.y <= b))):
            py = p.y
            px = 1.0 * ((i.p.x)**2 + (i.p.y-py)**2 - p.x**2) / (2*i.p.x - 2*p.x)
            res = Point(px, py)
            return True, res
        return False, None

    def intersection(self, p0, p1, l):
        # get the intersection of two parabolas
        p = p0
        if (p0.x == p1.x):
            py = (p0.y + p1.y) / 2.0
        elif (p1.x == l):
            py = p1.y
        elif (p0.x == l):
            py = p0.y
            p = p1
        else:
            # use quadratic formula
            z0 = 2.0 * (p0.x - l)
            z1 = 2.0 * (p1.x - l)

            a = 1.0/z0 - 1.0/z1;
            b = -2.0 * (p0.y/z0 - p1.y/z1)
            c = 1.0 * (p0.y**2 + p0.x**2 - l**2) / z0 - 1.0 * (p1.y**2 + p1.x**2 - l**2) / z1

            py = 1.0 * (-b-math.sqrt(b*b - 4*a*c)) / (2*a)
            
        px = 1.0 * (p.x**2 + (p.y-py)**2 - l**2) / (2*p.x-2*l)
        res = Point(px, py)
        return res

    def finish_edges(self):
        if self.arc is None:
            return
        l = self.x1 + (self.x1 - self.x0) + (self.y1 - self.y0)
        i = self.arc
        while i.pnext is not None:
            if i.s1 is not None:
                p = self.intersection(i.p, i.pnext.p, l * 2.0)
                i.s1.finish(p)
            i = i.pnext

    def print_output(self):
        it = 0
        for o in self.output:
            it = it + 1
            p0 = o.start
            p1 = o.end
            print (p0.x, p0.y, p1.x, p1.y)

    def get_output(self):
        res = []
        for o in self.output:
            p0 = o.start
            p1 = o.end
            res.append((p0.x, p0.y, p1.x, p1.y))
        return res            



class VoronoiApp:
    RADIUS = 3
    GRID_SPACING = 50
    CANVAS_SIZE = 500
    CENTER = CANVAS_SIZE // 2

    def __init__(self, master):
        self.master = master
        self.master.title("Voronoi Diagram")
        
        self.canvas = tk.Canvas(self.master, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, background='white')
        self.canvas.pack()
        
        self.points = []  # Points are stored in Cartesian coordinates (origin at center)
        self.scale = 1   # Scale factor to zoom out/in
        
        self.load_points_from_file("input.txt")
        self.canvas.bind('<Double-1>', self.on_double_click)
        self.update_voronoi()

    def draw_grid(self):
        """Draws a fixed background grid and center axes."""
        for x in range(0, self.CANVAS_SIZE + 1, self.GRID_SPACING):
            self.canvas.create_line(x, 0, x, self.CANVAS_SIZE, fill="lightgray")
        for y in range(0, self.CANVAS_SIZE + 1, self.GRID_SPACING):
            self.canvas.create_line(0, y, self.CANVAS_SIZE, y, fill="lightgray")
        
        # Draw axes through the center
        self.canvas.create_line(0, self.CENTER, self.CANVAS_SIZE, self.CENTER, fill="black", width=2)  # X-axis
        self.canvas.create_line(self.CENTER, 0, self.CENTER, self.CANVAS_SIZE, fill="black", width=2)  # Y-axis

    def compute_scale(self):
        """Computes a scale factor so that all points fit in the canvas.
           The scale is chosen so that the maximum absolute coordinate maps to 90% of half the canvas."""
        if not self.points:
            return 1
        max_val = max(max(abs(x), abs(y)) for x, y in self.points)
        if max_val == 0:
            return 1
        return (self.CANVAS_SIZE / 2 * 0.9) / max_val

    def transform_coordinates(self, x, y):
        """Converts a Cartesian coordinate (with origin at center) to canvas coordinates using current scale."""
        return self.CENTER + x * self.scale, self.CENTER - y * self.scale

    def load_points_from_file(self, filename):
        """Loads points from an input file, where each line contains two integers (x y)."""
        try:
            with open(filename, "r") as file:
                for line in file:
                    x, y = map(int, line.strip().split())
                    self.points.append((x, y))
        except FileNotFoundError:
            print(f"Error: {filename} not found.")

    def on_double_click(self, event):
        """Handles double-click events on the canvas.
           Converts canvas coordinates to Cartesian coordinates using the current scale and adds the new point."""
        # Inverse transformation: canvas -> Cartesian
        cart_x = (event.x - self.CENTER) / self.scale
        cart_y = (self.CENTER - event.y) / self.scale
        self.points.append((cart_x, cart_y))
        self.update_voronoi()

    def update_voronoi(self):
        """Recomputes the scale, clears and redraws the grid, points, and Voronoi lines."""
        self.scale = self.compute_scale()
        self.canvas.delete("all")
        self.draw_grid()
        
        # Redraw points
        for x, y in self.points:
            cx, cy = self.transform_coordinates(x, y)
            self.canvas.create_oval(cx - self.RADIUS, cy - self.RADIUS,
                                    cx + self.RADIUS, cy + self.RADIUS, fill="black")
        
        # Process and draw Voronoi diagram if there are enough points
        if len(self.points) > 1:
            vp = Voronoi(self.points)  # Voronoi expects Cartesian coordinates
            vp.process()
            lines = vp.get_output()
            print("Voronoi Output Lines:", lines)  # Debug output to terminal
            for x1, y1, x2, y2 in lines:
                p1 = self.transform_coordinates(x1, y1)
                p2 = self.transform_coordinates(x2, y2)
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill='blue', tags="voronoi_line")

def main():
    root = tk.Tk()
    app = VoronoiApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
    