# MVC-Project-Sem-1

This tool lets you visualize a multivariable function z = f(x,y) as a 3D surface, define parametric paths on that surface, and compare those paths using a discrete energy model based on gradients, directional derivatives, and path length.

It is designed for multivariable calculus (MVC) students who want to explore:
- Partial derivatives
- The gradient vector
- Directional derivatives
- Tangent vectors
- Parametric curves
- Slope along a path

Approximation of climb "effort." This is not a physics simulation. The “energy” is a mathematical measure of steepness × distance, not real Joules.

## Using the Program

## 1. Enter your Function z = f(x,y) 

You must type a Python expression using any or all of the following:
Variables: x, y
Operators: + - * / **
Functions: sin, cos, exp, log, sqrt, tan, etc.
Constants: pi, e

Examples you can type:
```
x**2 + y**2
sin(x)*cos(y)
sqrt(x**2 + y**2)
11/(3.35 + ((0.17*(x**2) + 0.24*(y**2))/1.7)**0.8)
```
Do not type ^ for powers—use **.
If you press Enter with nothing typed, the program exits.

## 2. Choose the X–Y range

You'll be asked:
```
Use custom range? (y/n):
```
Type n for default [-10, 10] in both x and y. Type y to enter custom bounds.
Example:
```
Enter x minimum: -5
Enter x maximum: 5
Enter y minimum: -5
Enter y maximum: 5
```
If you enter something invalid, defaults will be used.

## 3. Add Parametric Paths (Optional but Recommended) 

The script asks:
```
How many parametric paths do you want to plot? (0-5):
```
- Type 0 to draw only the surface.
- Type 1–5 to compare multiple routes.

For each path, you’ll enter:
### (a) Parametric x(t)
Example:
```
Enter x(t) for path 1: t
```
It must be a valid Python expression.

### (b) Parametric y(t)
Example:
```
Enter y(t) for path 1: 0.5*t**2
```
It must be a valid Python expression.

### (c) t-min and t-max
Example:
```
Enter t minimum: 0
Enter t maximum: 9
```
Invalid inputs fall back to defaults. If x(t) or y(t) is missing, that path is skipped.

## Understanding the Output
Understanding the Output

After your inputs, the program:
- Converts your function into a JAX function
- Computes the gradient ∇f automatically (the code does NOT approximate)
- Computes path derivatives dp/dt
- Computes the energy for each path
- Displays a 3D plot
- Prints energy values in the terminal
Example Output
```
Energy: Path 1 - 123.45J
Energy: Path 2 - 87.21J
```
How to Interpret Energy:
- The value is not real physical Joules.
- It’s a mathematical measure of steepness × distance.
- Lower energy = the path is less steep overall.

### Interacting with the 3D Graph 
Once the plot window opens:
- Left-click + drag = rotate
- Scroll = zoom
- Right-click + drag = pan (on some systems)
Each parametric path appears as a colored line lying on the surface. 

## Troubleshooting
### “Error evaluating function”
Check:
- Did you forget *? (Write 2*x NOT 2x)
- Did you type ^ instead of **?
- Did you use sin, not sine?

### “NameError: name ‘exp’ is not defined”
- Make sure your math functions are lowercase and valid: sin, cos, sqrt, exp, log, tan.

### Graph doesn’t appear
If you’re using VS Code:
- Set the Python interpreter correctly
- Make sure Matplotlib is installed

### JAX errors
If gradient/energy fails:
- Try reinstalling JAX
- Use classic CPU build
- Upgrade Python to 3.10 or 3.11


## Installation
```
pip install numpy matplotlib jax
```

## Misc
Push changes: 
```
git add . 
git commit -m "name it" 
git push origin main
```

Fetch changes: 
```
git fetch origin 
git reset --hard origin/main 
```