import random
import matplotlib.pyplot as plt
import networkx as nx
from ortools.linear_solver import pywraplp
import math

def plot_solution(n, m, selected, assignments, title):
    G = nx.Graph()
    
    # Agregar nodos de almacenes y clientes
    pos = {}
    for i in range(n):
        pos[f'A{i}'] = (0, -i)
        G.add_node(f'A{i}', color='red' if i in selected else 'gray', type='warehouse')
    for j in range(m):
        pos[f'C{j}'] = (1, -j)
        G.add_node(f'C{j}', color='blue', type='customer')

    # Agregar conexiones (asignaciones)
    for (i, j) in assignments:
        G.add_edge(f'A{i}', f'C{j}', weight=assignments[(i, j)] if isinstance(assignments, dict) else 1)

    # Dibujar el grafo
    colors = [G.nodes[n]['color'] for n in G.nodes]
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=colors, edge_color='black', font_size=10)
    plt.title(title)
    plt.show()

def localización_sin_capacidad(n, m, c, f, p=None):
    solver = pywraplp.Solver('PlantLocation', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    # Variables
    y = {i: solver.BoolVar(f'y[{i}]') for i in range(n)}
    z = {(i, j): solver.BoolVar(f'z[{i},{j}]') for i in range(n) for j in range(m)}
    
    # Función objetivo: minimizar costos de transporte y construcción
    solver.Minimize(solver.Sum(c[i][j] * z[i, j] for i in range(n) for j in range(m)) +
                    solver.Sum(f[i] * y[i] for i in range(n)))
    
    # Restricciones
    # Cada cliente debe ser atendido por exactamente una planta
    for j in range(m):
        solver.Add(solver.Sum(z[i, j] for i in range(n)) == 1)
    
    # Solo podemos asignar clientes a plantas abiertas
    for i in range(n):
        for j in range(m):
            solver.Add(z[i, j] <= y[i])
    
    # Límite en la cantidad de plantas abiertas (si aplica)
    if p is not None:
        solver.Add(solver.Sum(y[i] for i in range(n)) <= p)
    
    # Resolver el problema
    solver.Solve()
    
    # Imprimir resultados
    print('Costo total =', solver.Objective().Value(), "in", solver.WallTime()/1000, "seconds")
    selected_plants = [i for i in range(n) if y[i].solution_value() > 0.5]
    assignments = [(i, j) for i in range(n) for j in range(m) if z[i, j].solution_value() > 0.5]
    print('Plantas seleccionadas:', selected_plants)
    print('Asignaciones óptimas:', assignments)
    plot_solution(n, m, selected_plants, assignments, "Localización sin capacidad")
    return selected_plants, assignments

def localización_con_capacidad(n, m, c, f, q, d, l, p=None):
    solver = pywraplp.Solver('TransporteCapacitado', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    # Variables
    y = {i: solver.BoolVar(f'y[{i}]') for i in range(n)}  # Almacén abierto o no
    x = {(i, j): solver.NumVar(0, l[i][j], f'x[{i},{j}]') for i in range(n) for j in range(m)}  # Cantidad transportada
    
    # Función objetivo: minimizar costos de transporte y costos fijos
    solver.Minimize(solver.Sum(c[i][j] * x[i, j] for i in range(n) for j in range(m)) +
                    solver.Sum(f[i] * y[i] for i in range(n)))
    
    # Restricciones
    # Satisfacer la demanda de cada cliente
    for j in range(m):
        solver.Add(solver.Sum(x[i, j] for i in range(n)) == d[j])
    
    # No exceder la capacidad de cada almacén
    for i in range(n):
        solver.Add(solver.Sum(x[i, j] for j in range(m)) <= q[i] * y[i])
    
    # Restringir el flujo si el almacén no está abierto
    for i in range(n):
        for j in range(m):
            solver.Add(x[i, j] <= l[i][j] * y[i])
    
    # Límite en la cantidad de almacenes abiertos (si aplica)
    if p is not None:
        solver.Add(solver.Sum(y[i] for i in range(n)) <= p)
    
    # Resolver el problema
    solver.Solve()
    
    # Imprimir resultados
    print('Costo total =', solver.Objective().Value(), "in", solver.WallTime()/1000, "seconds")
    almacenes_seleccionados = [i for i in range(n) if y[i].solution_value() > 0.5]
    asignaciones = {(i, j): x[i, j].solution_value() for i in range(n) for j in range(m) if x[i, j].solution_value() > 0}
    print('Almacenes seleccionados:', almacenes_seleccionados)
    print('Asignaciones óptimas:', asignaciones)
    plot_solution(n, m, almacenes_seleccionados, asignaciones, "Localización con capacidad")
    return almacenes_seleccionados, asignaciones

def localización_p_mediana(n, m, c, f, d, p):
    solver = pywraplp.Solver('PMedianLocation', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    # Variables
    y = {i: solver.BoolVar(f'y[{i}]') for i in range(n)}
    z = {(i, j): solver.BoolVar(f'z[{i},{j}]') for i in range(n) for j in range(m)}
    
    # Función objetivo: minimizar costos de transporte y construcción
    solver.Minimize(solver.Sum(c[i][j] * d[j] * z[i, j] for i in range(n) for j in range(m)) +
                    solver.Sum(f[i] * y[i] for i in range(n)))
    
    # Restricciones
    # Cada cliente es atendido por exactamente una planta
    for j in range(m):
        solver.Add(solver.Sum(z[i, j] for i in range(n)) == 1)
    
    # Un cliente solo puede ser asignado a una planta si esta está abierta
    for i in range(n):
        for j in range(m):
            solver.Add(z[i, j] <= y[i])
    
    # Límite en la cantidad de plantas abiertas
    solver.Add(solver.Sum(y[i] for i in range(n)) <= p)
    
    # Resolver el problema
    solver.Solve()
    
    # Imprimir resultados
    print('Costo total =', solver.Objective().Value(), "in", solver.WallTime()/1000, "seconds")
    selected_plants = [i for i in range(n) if y[i].solution_value() > 0.5]
    assignments = {(i, j) for i in range(n) for j in range(m) if z[i, j].solution_value() > 0.5}
    print('Plantas seleccionadas:', selected_plants)
    print('Asignaciones óptimas:', assignments)
    plot_solution(n, m, selected_plants, assignments, "Localización p-Mediana")
    return selected_plants, assignments

# Generación aleatoria de datos de prueba
def generate_random_data(n, m):
    # Generación de ubicaciones aleatorias (en el rango [0, 1] en el plano)
    locations_plants = [(random.random(), random.random()) for _ in range(n)]  # Ubicación de las plantas
    locations_customers = [(random.random(), random.random()) for _ in range(m)]  # Ubicación de los clientes
    
    # Costos de transporte: distancia euclidiana entre planta y cliente
    c = [[math.sqrt((locations_plants[i][0] - locations_customers[j][0])**2 + 
                    (locations_plants[i][1] - locations_customers[j][1])**2) 
          for j in range(m)] for i in range(n)]
    
    # Costos fijos aleatorios
    f = [random.randint(30, 100) for _ in range(n)]
    
    # Capacidades aleatorias de las plantas
    q = [random.randint(20, 50) for _ in range(n)]
    
    # Demandas aleatorias de los clientes
    d = [random.randint(5, 20) for _ in range(m)]
    
    # Límites máximos de envío aleatorios
    l = [[random.randint(5, 15) for _ in range(m)] for _ in range(n)]
    
    return c, f, q, d, l

# Datos aleatorios
n = 4  # Número de almacenes
m = 5  # Número de clientes
c, f, q, d, l = generate_random_data(n, m)
p = 3  # Límite opcional de almacenes abiertos

# Ejecución de los métodos con datos aleatorios
localización_sin_capacidad(n, m, c, f, p)
localización_con_capacidad(n, m, c, f, q, d, l, p)
localización_p_mediana(n, m, c, f, d, p)
