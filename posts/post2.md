# Python Raytracing

Raytracing, at the most basic level, is a way of rendering images by shooting out 'rays' from the camera and seeing what was hit. For each hit, bounce. For each bounce, hit. And so on. This is a very high-level overview, but it's enough to get started.

## The Basics

Reflections can be calculated in a single line, with the most expensive part of it being the dot product.
`vector - 2 * dot(vector, normal) * normal`

Rays are literal raycasts with a direction and a starting point. They can be represented as a line, but are more commonly represented as a starting point and a direction.
```python
def trace_ray(ray, objects, light, depth=3):
    if depth <= 0:
        return np.zeros(3)

    t_min = float('inf')
    obj_hit = None
    
    hit_anything = False
    for obj in objects:
        t = obj.intersect(ray)
        if t is not None:
            hit_anything = True
            if t < t_min:
                t_min = t
                obj_hit = obj
    
    if not hit_anything:
        return np.zeros(3)

    hit_point = ray.origin + t_min * ray.direction
    normal = obj_hit.normal(hit_point)

    hit_point = hit_point + normal * 1e-6

    if isinstance(obj_hit, Checkerboard):
        color = obj_hit.color_at(hit_point)
    else:
        color = obj_hit.color

    to_light = normalize(light - hit_point)
    diffuse = np.maximum(np.dot(normal, to_light), 0)

    reflected = reflect(-to_light, normal)
    specular = np.power(np.maximum(np.dot(-ray.direction, reflected), 0), obj_hit.specular)

    shadow_ray = Ray(hit_point, to_light)
    in_shadow = any(obj.intersect(shadow_ray) is not None for obj in objects 
                   if obj != obj_hit)

    reflection_color = np.zeros(3)
    if hasattr(obj_hit, 'reflective') and obj_hit.reflective > 0:
        reflection_ray = Ray(hit_point, reflect(ray.direction, normal))
        reflection_color = trace_ray(reflection_ray, objects, light, depth - 1)

    light_intensity = 0 if in_shadow else 1
    final_color = color * (0.1 + 0.9 * diffuse * light_intensity) + specular * light_intensity
    if hasattr(obj_hit, 'reflective'):
        final_color = final_color * (1 - obj_hit.reflective) + reflection_color * obj_hit.reflective
    return np.clip(final_color, 0, 1)
```

We can disembowel this function into a few parts:
1. Find the closest object hit by the ray
2. Calculate the hit point and normal
3. Calculate the color of the object at the hit point
4. Calculate the diffuse and specular lighting
5. Calculate the reflection color if the object is reflective
6. Return the final color

## The Scene
The scene is a list of objects and a light source. The objects are defined by their intersection function and their normal function. The light source is a point in space.

```python
objects = [
    Sphere([0, 0, -5], 1, [1, 0, 0]),
    Cube([-1.5, -1, -3], [-0.5, 0, -2], [0, 1, 0]),
    Checkerboard()
]
```

The objects are easy enough to design, with the code looking like:

```python
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.1):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

    def intersect(self, ray):
        oc = ray.origin - self.center

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        if t1 < 0 and t2 < 0:
            return None
        
        return min(t for t in (t1, t2) if t > 0)


    def normal(self, point):
        return (point - self.center) / self.radius

class Cube:
    def __init__(self, min_point, max_point, color, specular=50, reflective=0.1):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

    def intersect(self, ray):
        t_min = (self.min_point - ray.origin) / ray.direction
        t_max = (self.max_point - ray.origin) / ray.direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return None
        
        return t_near if t_near > 0 else t_far


    def normal(self, point):
        eps = 1e-6
        if abs(point[0] - self.min_point[0]) < eps: return np.array([-1, 0, 0])
        if abs(point[0] - self.max_point[0]) < eps: return np.array([1, 0, 0])
        if abs(point[1] - self.min_point[1]) < eps: return np.array([0, -1, 0])
        if abs(point[1] - self.max_point[1]) < eps: return np.array([0, 1, 0])
        if abs(point[2] - self.min_point[2]) < eps: return np.array([0, 0, -1])
    
        return np.array([0, 0, 1])


class Checkerboard:
    def __init__(self, y=-2):
        self.y = y
        self.specular = 100
        self.reflective = 0.5


    def intersect(self, ray):
        if abs(ray.direction[1]) < 1e-6:
            return None
        
        t = -(ray.origin[1] - self.y) / ray.direction[1]
        if t < 0:
            return None
        return t

    def color_at(self, point):
        x = point[0]
        z = point[2]
        return np.array([1, 1, 1]) if (int(x * 2) + int(z * 2)) % 2 == 0 else np.array([0, 0, 0])

    def normal(self, point):
        return np.array([0, 1, 0])
```

To explain the code line by line, let's take a higher level look at what each class is doing:

```python
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.1):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective
```

The `__init__` contains all of the things that define the actual shape.

```python
    def intersect(self, ray):
        oc = ray.origin - self.center

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        if t1 < 0 and t2 < 0:
            return None
        
        return min(t for t in (t1, t2) if t > 0)
```

The `intersect` function is the function that determines if a ray intersects with the object. It returns the distance from the ray origin to the hit point.

```python
    def normal(self, point):
        return (point - self.center) / self.radius
```

The `normal` function returns the normal of the object at a given point. This is used for lighting calculations.

So, each class has 3 functions.


## The Result
The result:

![https://i.imgur.com/rX2WZOh.png](https://i.imgur.com/rX2WZOh.png)

The performance of this code is not great, but it's a good starting point for understanding raytracing (especially in Python).