#TARDA UNOS 30 SEGUNDOS EN DEVOLVER EL RESULTADO PERO LO DEVUELVE
#PONE A PRUEBA LA PACIENCIA DEL USUARIO


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import random
from datetime import datetime, timedelta
def generate_random_data(num_requests, grid_size=10, time_window_width=120, vehicle_capacity=3, seed=None):    
    """Genera datos aleatorios para el problema CDARP-TW.
    
    Args:
        num_requests: Número de solicitudes (personas a transportar)
        grid_size: Tamaño del área de servicio (grid_size x grid_size)
        time_window_width: Ancho de las ventanas de tiempo en minutos
        vehicle_capacity: Capacidad máxima del vehículo
        seed: Semilla para reproducibilidad
        
    Returns:
        Un diccionario con todos los datos del problema
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 1. Ubicaciones (depósito + pickup/delivery para cada solicitud)
    num_locations = 2 * num_requests + 1
    locations = np.random.randint(0, grid_size, size=(num_locations, 2))
    
    # El depósito es la primera ubicación (índice 0)
    depot = locations[0]
    
    # 2. Matriz de distancias (simétrica)
    distances = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(i+1, num_locations):
            dist = np.linalg.norm(locations[i] - locations[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    # 3. Tiempos de servicio (en minutos)
    service_time = np.random.randint(1, 5, size=num_locations)
    service_time[0] = 0  # No hay tiempo de servicio en el depósito
    
    # 4. Ventanas de tiempo
    # Hora de inicio (8:00 AM)
    start_time = datetime.strptime("08:00", "%H:%M")
    
    # Tiempos de apertura (earliest time)
    open_time = [start_time for _ in range(num_locations)]
    
    # Tiempos de cierre (latest time)
    close_time = [start_time + timedelta(minutes=240) for _ in range(num_locations)]
    
    # Generar ventanas de tiempo para pickups y deliveries
    for i in range(1, num_requests + 1):
        # Índices de pickup (1, 3, 5,...) y delivery (2, 4, 6,...)
        pickup_idx = 2*i - 1
        delivery_idx = 2*i
        
        # Tiempo de viaje entre pickup y delivery
        travel_time = distances[pickup_idx][delivery_idx] * 2  # Factor de conversión
        
        # Ventana para pickup
        pickup_open = start_time + timedelta(minutes=random.randint(0, 60))
        pickup_close = pickup_open + timedelta(minutes=time_window_width)
        open_time[pickup_idx] = pickup_open
        close_time[pickup_idx] = pickup_close
        
        # Ventana para delivery debe ser después del pickup
        delivery_open = pickup_close + timedelta(minutes=travel_time//2)
        delivery_close = delivery_open + timedelta(minutes=time_window_width)
        open_time[delivery_idx] = delivery_open
        close_time[delivery_idx] = delivery_close
    
    # Convertir tiempos a minutos desde inicio
    open_time_min = [(t - start_time).total_seconds() // 60 for t in open_time]
    close_time_min = [(t - start_time).total_seconds() // 60 for t in close_time]
    
    # 5. Demandas (pickup: +1, delivery: -1)
    demands = [0] * num_locations
    for i in range(1, num_requests + 1):
        pickup_idx = 2*i - 1
        delivery_idx = 2*i
        demands[pickup_idx] = 1
        demands[delivery_idx] = -1
    
    # 6. Nombres de las ubicaciones
    location_names = ["Depósito"]
    for i in range(1, num_requests + 1):
        location_names.append(f"Pickup {i}")
        location_names.append(f"Delivery {i}")
    
    return {
        'num_requests': num_requests,
        'num_locations': num_locations,
        'locations': locations,
        'distances': distances,
        'service_time': service_time,
        'open_time': open_time_min,
        'close_time': close_time_min,
        'demands': demands,
        'vehicle_capacity': vehicle_capacity,
        'location_names': location_names,
        'start_time': start_time
    }

def plot_problem(data):
    """Visualiza las ubicaciones del problema."""
    plt.figure(figsize=(10, 8))
    
    # Dibujar el depósito
    plt.scatter(data['locations'][0, 0], data['locations'][0, 1], 
                c='red', s=200, marker='s', label='Depósito')
    
    # Dibujar pickups y deliveries
    for i in range(1, data['num_locations']):
        if i % 2 == 1:  # Pickup
            plt.scatter(data['locations'][i, 0], data['locations'][i, 1], 
                        c='blue', s=100, marker='^', label='Pickup' if i == 1 else "")
        else:  # Delivery
            plt.scatter(data['locations'][i, 0], data['locations'][i, 1], 
                        c='green', s=100, marker='v', label='Delivery' if i == 2 else "")
    
    # Conectar pickups con sus deliveries
    for i in range(1, data['num_requests'] + 1):
        pickup_idx = 2*i - 1
        delivery_idx = 2*i
        plt.plot([data['locations'][pickup_idx, 0], data['locations'][delivery_idx, 0]],
                 [data['locations'][pickup_idx, 1], data['locations'][delivery_idx, 1]],
                 'k--', alpha=0.3)
    
    plt.title(f"Problema CDARP-TW con {data['num_requests']} solicitudes")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Mostrar información de las ventanas de tiempo
    print("\nVentanas de tiempo:")
    for i in range(data['num_locations']):
        open_time = data['start_time'] + timedelta(minutes=data['open_time'][i])
        close_time = data['start_time'] + timedelta(minutes=data['close_time'][i])
        print(f"{data['location_names'][i]}: {open_time.strftime('%H:%M')} - {close_time.strftime('%H:%M')}")
        
def solve_cdarp_tw(data, time_limit=30):
    """Resuelve el problema CDARP-TW usando OR-Tools."""
    # 1. Crear el índice de gestión
    manager = pywrapcp.RoutingIndexManager(
        data['num_locations'],  # Número de ubicaciones
        1,                     # Número de vehículos
        0                      # Índice del depósito
    )
    
    # 2. Crear el modelo de routing
    routing = pywrapcp.RoutingModel(manager)
    
    # 3. Definir la función de costo (distancia)
    def distance_callback(from_index, to_index):
        """Retorna la distancia entre dos nodos."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distances'][from_node][to_node] * 10)  # Convertir a entero
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 4. Restricción de capacidad
    def demand_callback(from_index):
        """Retorna la demanda de un nodo."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # slack null
        [data['vehicle_capacity']],  # capacidades de los vehículos
        True,  # start cumul to zero
        'Capacity'
    )
    
    # 5. Restricciones de tiempo
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
    
        # Convert distance to travel time (minutes)
        # Assuming speed of 1 distance unit = 2 minutes
        travel_time = int(data['distances'][from_node][to_node] * 2)
        service_time = data['service_time'][from_node]
        return travel_time + service_time
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Calcular el horizonte temporal máximo (en minutos)
    max_horizon = int(max(data['close_time'])) + 120  # Añadir margen
    
    # Añadir dimensión de tiempo
    routing.AddDimension(
        time_callback_index,
        int(60),  # slack máximo (permite esperar)
        int(max_horizon),  # tiempo máximo permitido
        False,  # Don't start cumul to zero (debe ser bool, no int)
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Añadir ventanas de tiempo para cada ubicación
    for location_idx in range(data['num_locations']):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(
            int(data['open_time'][location_idx]),
            int(data['close_time'][location_idx])
        )
    
    # Forzar que el pickup se realice antes que el delivery
    for request in range(1, data['num_requests'] + 1):
        pickup_index = manager.NodeToIndex(2*request - 1)
        delivery_index = manager.NodeToIndex(2*request)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))
    
    # Configurar parámetros de búsqueda
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit
    
    # ESTA ES LA ÚNICA LÍNEA QUE NECESITAS CAMBIAR:
    search_parameters.log_search = False  # Cambiar de True a False
    
    # 6. Resolver el problema
    solution = routing.SolveWithParameters(search_parameters)
    
    return manager, routing, solution
        
        
        
def print_solution(data, manager, routing, solution):
    """Imprime la solución en la consola."""
    if not solution:
        print("No se encontró solución!")
        return
    
    print(f"Distancia total: {solution.ObjectiveValue()} unidades")
    
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    
    for vehicle_id in range(1):  # Solo un vehículo en este ejemplo
        index = routing.Start(vehicle_id)
        plan_output = f"Ruta del vehículo {vehicle_id}:\n"
        route_load = 0
        route_time = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            load_var = capacity_dimension.CumulVar(index)
            
            location_name = data['location_names'][node_index]
            time_min = solution.Min(time_var)
            time_max = solution.Max(time_var)
            load = solution.Value(load_var)
            
            arrival_time = data['start_time'] + timedelta(minutes=time_min)
            departure_time = data['start_time'] + timedelta(minutes=time_max)
            
            plan_output += (
                f"{location_name} "
                f"Llegada: {arrival_time.strftime('%H:%M')} "
                f"Salida: {departure_time.strftime('%H:%M')} "
                f"Carga: {load}\n"
            )
            
            route_load = load
            route_time = time_max
            index = solution.Value(routing.NextVar(index))
        
        # Nodo final (depósito)
        node_index = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        time_min = solution.Min(time_var)
        time_max = solution.Max(time_var)
        arrival_time = data['start_time'] + timedelta(minutes=time_min)
        
        plan_output += (
            f"{data['location_names'][node_index]} "
            f"Llegada: {arrival_time.strftime('%H:%M')}\n"
            f"Tiempo total de la ruta: {route_time} minutos\n"
            f"Carga final: {route_load}\n"
        )
        
        print(plan_output)
        total_time += route_time
    
    print(f"Tiempo total de todas las rutas: {total_time} minutos")

def plot_solution(data, manager, routing, solution):
    """Visualiza la solución gráficamente."""
    if not solution:
        print("No hay solución para visualizar")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Dibujar todas las ubicaciones
    for i in range(data['num_locations']):
        if i == 0:  # Depósito
            plt.scatter(data['locations'][i, 0], data['locations'][i, 1], 
                        c='red', s=200, marker='s', label='Depósito')
        elif i % 2 == 1:  # Pickup
            plt.scatter(data['locations'][i, 0], data['locations'][i, 1], 
                        c='blue', s=100, marker='^', label='Pickup' if i == 1 else "")
        else:  # Delivery
            plt.scatter(data['locations'][i, 0], data['locations'][i, 1], 
                        c='green', s=100, marker='v', label='Delivery' if i == 2 else "")
    
    # Dibujar la ruta
    index = routing.Start(0)
    route_x = []
    route_y = []
    
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route_x.append(data['locations'][node_index, 0])
        route_y.append(data['locations'][node_index, 1])
        index = solution.Value(routing.NextVar(index))
    
    # Añadir el depósito al final
    node_index = manager.IndexToNode(index)
    route_x.append(data['locations'][node_index, 0])
    route_y.append(data['locations'][node_index, 1])
    
    plt.plot(route_x, route_y, 'r-', linewidth=2, label='Ruta')
    
    plt.title(f"Solución CDARP-TW con {data['num_requests']} solicitudes")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.legend()
    plt.show()        
        
# Relajar las ventanas de tiempo
data = generate_random_data(num_requests=3, time_window_width=60, seed=42)

# Aumentar el límite de tiempo
manager, routing, solution = solve_cdarp_tw(data, time_limit=60)
# Visualizar el problema
plot_problem(data)


# Mostrar resultados
print_solution(data, manager, routing, solution)
plot_solution(data, manager, routing, solution)       
        
        
        
        
        
        