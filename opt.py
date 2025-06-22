import numpy as np
from cmaes import CMA
from cpg import kuramoto_cpg, configure_trot_gait
from scipy.integrate import solve_ivp

# =============================================
# Función objetivo para optimización
# =============================================
def evaluate_gait(parameters):
    """
    Evalúa el desempeño de un conjunto de parámetros
    Devuelve: (desplazamiento, consumo_energético)
    """
    # Desempaquetar parámetros
    n = 8
    omega = parameters[:n]
    R = parameters[n:2*n]
    K_scale = parameters[2*n]
    phi_scale = parameters[2*n+1]
    
    # Configuración base
    _, K_base, phi_base, a, _ = configure_trot_gait()
    
    # Escalar matrices de acoplamiento y desfase
    K = K_base * K_scale
    phi = phi_base * phi_scale
    
    # Simular CPG
    theta0 = np.zeros(n)
    r0 = 0.1 * np.ones(n)
    y0 = np.concatenate([theta0, r0])
    t_span = [0, 5]
    sol = solve_ivp(kuramoto_cpg, t_span, y0, args=(omega, K, phi, a, R), 
                    dense_output=True)
    
    if not sol.success:
        return -10, 1000  # Penalización si falla
    
    # Calcular ángulos articulares
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol_eval = sol.sol(t_eval)
    theta = sol_eval[:n]
    r = sol_eval[n:2*n]
    joint_angles = r * np.sin(theta)
    
    # 1. Calcular desplazamiento (simplificado)
    # (Asumimos velocidad proporcional al movimiento de las caderas)
    hip_angles = joint_angles[[0, 2, 4, 6]]
    hip_velocity = np.mean(np.abs(np.diff(hip_angles)))
    displacement = hip_velocity * t_span[1]
    
    # 2. Calcular consumo energético (simplificado)
    # (Proporcional al cuadrado de las velocidades angulares)
    joint_velocity = np.diff(joint_angles, axis=1)
    energy = np.sum(joint_velocity**2) * (t_eval[1] - t_eval[0])
    
    return displacement, energy

def objective_function(params):
    """Función objetivo para el optimizador"""
    displacement, energy = evaluate_gait(params)
    
    # Queremos maximizar desplazamiento y minimizar energía
    # Ponderaciones (ajustables)
    w_displacement = 1.0
    w_energy = 0.01
    
    return -(w_displacement * displacement - w_energy * energy)

# =============================================
# Optimización con CMA-ES
# =============================================
def optimize_parameters():
    # Espacio de búsqueda: [omega (8), R (8), K_scale, phi_scale]
    n_params = 8 + 8 + 1 + 1
    
    # Configuración inicial (basada en marcha tipo trote)
    omega, _, _, _, R = configure_trot_gait()
    initial_params = np.concatenate([omega, R, [1.0], [1.0]])
    
    # Límites de los parámetros
    lower_bounds = [0.1*np.pi] * 8 + [0.1] * 8 + [0.1] + [0.1]
    upper_bounds = [4.0*np.pi] * 8 + [2.0] * 8 + [5.0] + [2.0]
    bounds = np.array(list(zip(lower_bounds, upper_bounds)))
    
    # Configurar CMA-ES
    optimizer = CMA(mean=initial_params, 
                   sigma=0.5, 
                   population_size=20,
                   bounds=bounds)
    
    # Ejecutar optimización
    best_params = None
    best_value = float('inf')
    
    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()
            value = objective_function(params)
            solutions.append((params, value))
            
            if value < best_value:
                best_value = value
                best_params = params
        
        optimizer.tell(solutions)
        print(f"Generación {generation}: Mejor valor = {best_value:.4f}")
    
    # Guardar mejores parámetros
    np.save('optimized_params.npy', best_params)
    return best_params

# =============================================
# Ejecutar optimización
# =============================================
if __name__ == "__main__":
    optimized_params = optimize_parameters()
    print("Parámetros optimizados:", optimized_params)