import numpy as np
import matplotlib
import math
import jax
import jax.numpy as jnp
#import numdifftools as nd
try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility on macOS
except Exception:
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        pass  # Use default backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import re

def create_3d_graph(function_str, x_range, y_range, resolution=50, parametric_paths=None):
    """
    Creates an interactive 3D graph of a function z = f(x, y)
    
    Parameters:
    -----------
    function_str : str
        The function expression as a string (e.g., "x**2 + y**2", "sin(x) * cos(y)")
    x_range : tuple
        Range for x values (min, max)
    y_range : tuple
        Range for y values (min, max)
    resolution : int
        Number of points along each axis for the mesh grid
    parametric_paths : list, optional
        List of tuples containing (x_func_str, y_func_str, t_min, t_max, t_resolution)
        for plotting parametric paths on the surface
    """
    # Create a mesh grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Safe evaluation namespace - only allow math functions and numpy
    safe_dict = {
        'x': X,
        'y': Y,
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
    }
    
    # Replace common math function names to use numpy versions
    function_str = function_str.replace('^', '**')  # Replace ^ with ** for exponentiation
    
    try:
        # Evaluate the function
        print(f"Evaluating function: {function_str}")
        Z = eval(function_str, {"__builtins__": {}}, safe_dict)

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Remove the orientation indicator (colored + sign) at the bottom
        # Hide the colored panes that create the orientation indicator
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Make pane edges invisible
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        # Alternative method to hide orientation indicator
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Create surface plot with terrain-style colormap (like topographic maps)
        colormap = 'terrain'  # Options: 'terrain', 'gist_earth', 'RdYlGn', 'nipy_spectral', 'tab20', 'plasma'
        # Reduce alpha slightly if parametric paths exist so they're more visible on top
        surface_alpha = 0.6 if parametric_paths else 0.8
        # Set zorder=1 so surface appears behind parametric paths
        surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=surface_alpha, linewidth=0, antialiased=True, zorder=1)
        
        # Add wireframe overlay to better see the structure in all quadrants
        #wire = ax.plot_wireframe(X, Y, Z, alpha=0.2, linewidth=0.5, color='black')
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        # Title hidden - uncomment the line below to show the function string at the top
        # ax.set_title(f'z = {function_str}', fontsize=14, fontweight='bold')
        
        # Explicitly set axis limits to ensure all quadrants are visible
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        
        # Set tick spacing for x and y axes
        # Adjust these values to change the interval between tick marks
        x_tick_interval = 3.0  # Change this to adjust x-axis tick spacing (e.g., 0.5, 2.0, 5.0)
        y_tick_interval = 3.0  # Change this to adjust y-axis tick spacing (e.g., 0.5, 2.0, 5.0)
        
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
        
        # Set z limits based on the data range, with some padding
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        z_range = z_max - z_min
        if z_range > 0:
            ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
        else:
            # If z is constant, set a small range around it
            ax.set_zlim(z_min - 1, z_max + 1)
        
        # Make the actual grid/graph bigger by adjusting subplot position
        # This makes the plot area take up more of the figure, making the grid appear larger
        # Adjust these values: [left, bottom, width, height] - all values between 0 and 1
        # Smaller left/bottom and larger width/height = bigger grid
        plot_left = -0.17    # Decrease to move plot left and make it wider
        plot_bottom = 0.05  # Decrease to move plot down and make it taller  
        plot_width = 0.90   # Increase to make plot wider (max 1.0)
        plot_height = 0.90  # Increase to make plot taller (max 1.0)
        
        # Apply the position adjustment to make the grid bigger
        ax.set_position([plot_left, plot_bottom, plot_width, plot_height])
        
        # Also adjust the aspect ratio to make x and y axes appear more spaced out
        # Adjust this multiplier to make the grid appear larger (higher = bigger grid)
        grid_scale_factor = 1.2  # Increase this value to make the grid bigger (e.g., 1.5, 2.0)
        
        # Calculate the ranges for aspect ratio
        x_range_size = x_range[1] - x_range[0]
        y_range_size = y_range[1] - y_range[0]
        z_range_size = z_max - z_min if z_range > 0 else 2
        
        # Set box aspect to make the grid appear larger and more spaced out
        try:
            # Scale up the x and y dimensions relative to z
            ax.set_box_aspect([x_range_size * grid_scale_factor, 
                              y_range_size * grid_scale_factor, 
                              z_range_size * 2])
        except:
            # Fallback for older matplotlib versions
            pass
        
        # Add grid for better visualization
        ax.grid(True)
        
        # Draw axes lines through origin to show all quadrants/octants
        # X-axis line (y=0, z=0)
        if x_range[0] <= 0 <= x_range[1]:
            ax.plot([x_range[0], x_range[1]], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.5)
        # Y-axis line (x=0, z=0)
        if y_range[0] <= 0 <= y_range[1]:
            ax.plot([0, 0], [y_range[0], y_range[1]], [0, 0], 'k-', linewidth=1, alpha=0.5)
        # Z-axis line (x=0, y=0)
        if z_min <= 0 <= z_max:
            ax.plot([0, 0], [0, 0], [z_min, z_max], 'k-', linewidth=1, alpha=0.5)
        
        # Plot parametric paths if provided
        if parametric_paths:
            # Bright distinct colors for up to 5 paths
            path_colors = ['red', 'blue', 'green', 'orange', 'magenta']
            
            for idx, path in enumerate(parametric_paths):
                x_func_str, y_func_str, t_min, t_max, t_resolution = path
                
                # Create t values
                t = np.linspace(t_min, t_max, t_resolution)
                
                # Safe evaluation namespace for parametric functions
                param_safe_dict = {
                    't': t,
                    'np': np,
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'exp': np.exp,
                    'log': np.log,
                    'sqrt': np.sqrt,
                    'abs': np.abs,
                    'pi': np.pi,
                    'e': np.e,
                    'arcsin': np.arcsin,
                    'arccos': np.arccos,
                    'arctan': np.arctan,
                    'sinh': np.sinh,
                    'cosh': np.cosh,
                    'tanh': np.tanh,
                }
                
                try:
                    # Replace ^ with ** for exponentiation
                    x_func_str = x_func_str.replace('^', '**')
                    y_func_str = y_func_str.replace('^', '**')
                    
                    # Evaluate x(t) and y(t)
                    x_t = eval(x_func_str, {"__builtins__": {}}, param_safe_dict)
                    y_t = eval(y_func_str, {"__builtins__": {}}, param_safe_dict)
                    
                    # Calculate z(t) = f(x(t), y(t)) by evaluating the main function at each point
                    z_t = np.zeros_like(x_t)
                    for i in range(len(x_t)):
                        x_val = x_t[i]
                        y_val = y_t[i]
                        # Evaluate the main function at this point
                        eval_dict = {
                            'x': x_val,
                            'y': y_val,
                            'np': np,
                            'sin': np.sin,
                            'cos': np.cos,
                            'tan': np.tan,
                            'exp': np.exp,
                            'log': np.log,
                            'sqrt': np.sqrt,
                            'abs': np.abs,
                            'pi': np.pi,
                            'e': np.e,
                            'arcsin': np.arcsin,
                            'arccos': np.arccos,
                            'arctan': np.arctan,
                            'sinh': np.sinh,
                            'cosh': np.cosh,
                            'tanh': np.tanh,
                        }
                        try:
                            z_val = eval(function_str, {"__builtins__": {}}, eval_dict)
                            # Handle scalar or array results
                            if np.isscalar(z_val):
                                z_t[i] = z_val
                            else:
                                z_t[i] = z_val if len(z_val) == 1 else z_val[0]
                        except:
                            # If evaluation fails, use NaN (will skip that point)
                            z_t[i] = np.nan
                    
                    # Filter points: remove NaN values and points outside x/y domain
                    # Only plot points that are within the x_range and y_range bounds
                    in_x_domain = (x_t >= x_range[0]) & (x_t <= x_range[1])
                    in_y_domain = (y_t >= y_range[0]) & (y_t <= y_range[1])
                    in_domain = in_x_domain & in_y_domain
                    valid_z = ~np.isnan(z_t)
                    
                    # Combine all validity checks: must be in domain AND have valid z
                    valid_mask = in_domain & valid_z
                    
                    if np.any(valid_mask):
                        x_t_clean = x_t[valid_mask]
                        y_t_clean = y_t[valid_mask]
                        z_t_clean = z_t[valid_mask]
                    else:
                        # If no valid points, create empty arrays to avoid plotting
                        x_t_clean = np.array([])
                        y_t_clean = np.array([])
                        z_t_clean = np.array([])
                    
                    # Plot the parametric path with distinct color (only if there are valid points)
                    if len(x_t_clean) > 0:
                        # Set zorder=10 so paths appear on top of the surface
                        # Use full opacity and thicker line to ensure visibility
                        color = path_colors[idx % len(path_colors)]
                        ax.plot(x_t_clean, y_t_clean, z_t_clean, color=color, linewidth=4, alpha=1.0, 
                               label=f'Path {idx+1}', zorder=10)
                        
                        #print(f"Path {idx+1} plotted successfully in {color} ({len(x_t_clean)} points within domain)")
                    else:
                        print(f"Path {idx+1} has no points within the x/y domain, skipping plot")
                    
                except Exception as e:
                    print(f"Error plotting path {idx+1}: {e}")
            
            # Add legend if paths are plotted
            if parametric_paths:
                ax.legend(loc='upper right')
        
        # Set initial view angle to show all quadrants clearly
        # Elevation and azimuth angles for better initial view
        ax.view_init(elev=30, azim=45)

        plt.show(block=True)
        print("Graph window closed.")
        
    except Exception as e:
        print(f"Error evaluating function: {e}")
        print("\nMake sure your function uses valid Python syntax.")
        print("Examples:")
        print("  - x**2 + y**2")
        print("  - sin(x) * cos(y)")
        print("  - exp(-(x**2 + y**2))")
        print("  - x*y + 2*x")


def main():
    """
    Main function to get user input and create the 3D graph
    """
    # Test matplotlib backend
    try:
        backend = matplotlib.get_backend()
        print(f"Using matplotlib backend: {backend}")
    except Exception as e:
        print(f"Warning: Could not get backend info: {e}")
    
    print("=" * 60)
    print("3D Function Grapher")
    print("=" * 60)
    print("\nEnter a function z = f(x, y)")
    print("You can use: x, y, sin, cos, tan, exp, log, sqrt, abs, pi, e, etc.")
    print("Use ** for exponentiation (e.g., x**2 for x squared)")
    print("\nExamples:")
    print("  - x**2 + y**2")
    print("  - sin(x) * cos(y)")
    print("  - exp(-(x**2 + y**2))")
    print("  - x*y + 2*x")
    print("  - sqrt(x**2 + y**2)")
    print("-" * 60)
    
    # Get function input
    function_str = input("\nEnter your function: ")
    #function_str = "11/(3.35 + ((0.17*(x**2) + 0.24*(y**2) )/1.7)**0.8)"
    function_str.strip() 
    
    if not function_str:
        print("No function entered. Exiting.")
        return
    
    # Optional: Get custom ranges
    print("\nDefault range: x and y from -10 to 10")
    custom_range = input("Use custom range? (y/n): ")
    u = 10
    custom_range = custom_range.strip().lower()
    
    if custom_range == 'y':
        try:
            x_min = float(input("Enter x minimum: "))
            x_max = float(input("Enter x maximum: "))
            y_min = float(input("Enter y minimum: "))
            y_max = float(input("Enter y maximum: "))
            x_range = (x_min, x_max)
            y_range = (y_min, y_max)
        except ValueError:
            print("Invalid input. Using default range.")
            x_range = (-u, u)
            y_range = (-u, u)
    else:
        x_range = (-u, u)
        y_range = (-u, u)
    
    # Collect parametric paths from user
    parametric_paths = []
    print("\n" + "=" * 60)
    print("Parametric Path Input (Optional)")
    print("=" * 60)
    num_paths_input = input("How many parametric paths do you want to plot? (0-5, press Enter for 0): ").strip()
    
    if num_paths_input:
        try:
            num_paths = int(num_paths_input)
            # Limit to maximum of 5 paths
            num_paths = min(max(0, num_paths), 5)
            
            if num_paths > 0:
                print(f"\nYou can plot up to {num_paths} parametric paths.")
                print("For each path, provide x(t) and y(t) functions.")
                print("The z coordinate will be calculated as z(t) = f(x(t), y(t))")
                print("Use 't' as the parameter. Examples:")
                print("  - x(t) = cos(t), y(t) = sin(t)")
                print("  - x(t) = t, y(t) = t**2")
                print("-" * 60)
                
                for i in range(num_paths):
                    print(f"\nPath {i+1}:")
                    x_func_str = input(f"  Enter x(t) for path {i+1}: ").strip()
                    y_func_str = input(f"  Enter y(t) for path {i+1}: ").strip()
                    
                    if x_func_str and y_func_str:
                        # Get t range (number of points is automatically determined for smooth curves)
                        t_min_input = input(f"  Enter t minimum (default -10): ").strip()
                        t_max_input = input(f"  Enter t maximum (default 10): ").strip()
                        
                        t_min = float(t_min_input) if t_min_input else -10.0
                        t_max = float(t_max_input) if t_max_input else 10.0
                        # Use default resolution for smooth curves (200 points)
                        t_resolution = 200
                        
                        parametric_paths.append((x_func_str, y_func_str, t_min, t_max, t_resolution))
                        print(f"  Path {i+1} added successfully!")
                    else:
                        print(f"  Skipping path {i+1} (empty input)")
        except ValueError:     
            print("Invalid input. No parametric paths will be plotted.")
            return
    # Compute the Energy using exact derivatives with JAX
    if parametric_paths:
        print("\n" + "=" * 60)
        print("Computing Energy Values (Using JAX)")
        print("=" * 60)
        
        # Prepare function string for evaluation
        func_str_clean = function_str.replace('^', '**')
        
        # Create JAX-compatible function for f(x, y)
        def create_f_function(func_str):
            """Create a JAX-compatible function from string"""
            # Safe namespace for JAX functions
            jax_safe_dict = {
                'jnp': jnp,
                'sin': jnp.sin,
                'cos': jnp.cos,
                'tan': jnp.tan,
                'exp': jnp.exp,
                'log': jnp.log,
                'sqrt': jnp.sqrt,
                'abs': jnp.abs,
                'pi': jnp.pi,
                'e': jnp.e,
                'arcsin': jnp.arcsin,
                'arccos': jnp.arccos,
                'arctan': jnp.arctan,
                'sinh': jnp.sinh,
                'cosh': jnp.cosh,
                'tanh': jnp.tanh,
            }
            
            # Create a function that evaluates the string
            def f_eval(x, y):
                eval_dict = jax_safe_dict.copy()
                eval_dict['x'] = x
                eval_dict['y'] = y
                return eval(func_str, {"__builtins__": {}}, eval_dict)
            
            return f_eval
        
        # Create JAX-compatible functions for x(t) and y(t)
        def create_param_function(param_str):
            """Create a JAX-compatible parametric function from string"""
            jax_safe_dict = {
                'jnp': jnp,
                'sin': jnp.sin,
                'cos': jnp.cos,
                'tan': jnp.tan,
                'exp': jnp.exp,
                'log': jnp.log,
                'sqrt': jnp.sqrt,
                'abs': jnp.abs,
                'pi': jnp.pi,
                'e': jnp.e,
                'arcsin': jnp.arcsin,
                'arccos': jnp.arccos,
                'arctan': jnp.arctan,
                'sinh': jnp.sinh,
                'cosh': jnp.cosh,
                'tanh': jnp.tanh,
            }
            
            def param_eval(t):
                eval_dict = jax_safe_dict.copy()
                eval_dict['t'] = t
                return eval(param_str, {"__builtins__": {}}, eval_dict)
            
            return param_eval
        
        # Create the f function
        f_func = create_f_function(func_str_clean.replace('^', '**'))
        
        # Compute gradient function using JAX
        grad_f = jax.grad(f_func, argnums=(0, 1))  # Returns (df/dx, df/dy)
        
        for i in range(len(parametric_paths)):
            x_func_str = parametric_paths[i][0]
            y_func_str = parametric_paths[i][1]
            tMIN = parametric_paths[i][2]
            tMAX = parametric_paths[i][3]
            
            print(f"\nPath {i+1}: ({x_func_str}, {y_func_str})")
            print(f"  t range: [{tMIN}, {tMAX}]")
            
            # Prepare parametric function strings
            x_func_clean = x_func_str.replace('^', '**')
            y_func_clean = y_func_str.replace('^', '**')
            
            # Create JAX-compatible parametric functions
            x_func = create_param_function(x_func_clean)
            y_func = create_param_function(y_func_clean)
            
            # Create vector function p(t) = <x(t), y(t)>
            def p_func(t):
                return jnp.array([x_func(t), y_func(t)])
            
            # Compute dp/dt using JAX
            dp_dt_func = jax.jacfwd(p_func)  # Returns [dx/dt, dy/dt]
            
            # Initialize Energy
            Energy = 0.0
            
            # Compute the sum
            max_j = int(math.floor((tMAX - 0.01) / 0.01))
            for j in range(1, max_j + 1):
                t_curr = j / 100.0
                t_next = (j + 1) / 100.0
                
                # Evaluate p(t) at current and next points
                p_curr = p_func(t_curr)
                p_next = p_func(t_next)
                
                # Compute dp/dt at current point
                dp_dt_curr = dp_dt_func(t_curr)
                
                # Compute |dp/dt|
                dp_dt_magnitude = jnp.sqrt(dp_dt_curr[0]**2 + dp_dt_curr[1]**2)
                
                # Avoid division by zero
                dp_dt_magnitude = jnp.where(dp_dt_magnitude < 1e-10, 1e-10, dp_dt_magnitude)
                
                # Compute unit tangent vector u^ = dp/dt / |dp/dt|
                u_hat = dp_dt_curr / dp_dt_magnitude
                
                # Evaluate f at current point
                x_curr_val = float(p_curr[0])
                y_curr_val = float(p_curr[1])
                
                # Compute gradient ∇f at (x(t), y(t))
                grad_f_curr = grad_f(x_curr_val, y_curr_val)
                grad_f_vec = jnp.array([grad_f_curr[0], grad_f_curr[1]])
                
                # Compute directional derivative D_û f = ∇f · u^
                directional_deriv = jnp.dot(grad_f_vec, u_hat)
                
                # Take absolute value
                abs_directional_deriv = jnp.abs(directional_deriv)
                
                # Compute distance |p((j+1)/100) - p(j/100)|
                distance = jnp.sqrt((p_next[0] - p_curr[0])**2 + (p_next[1] - p_curr[1])**2)
                
                # Add to Energy: |D_û f| * |p((j+1)/100) - p(j/100)|
                opop = float(abs_directional_deriv * distance) 
                if not math.isnan(opop): 
                    Energy += opop
            print(f"Energy: Path {i+1} - {Energy}J") 
        
        print("\n" + "=" * 60) 
        
    # Create and display the graph  
    print("\nGenerating graph...") 
    print("Tip: Click and drag with left mouse button (or touchpad) to rotate the graph!")
    create_3d_graph(function_str, x_range, y_range, parametric_paths=parametric_paths if parametric_paths else None)


if __name__ == "__main__":
    main()