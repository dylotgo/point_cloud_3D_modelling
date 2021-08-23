# point_cloud_3D_modelling
# En este respositorio se muestran unicamente las funciones creadas para el modelado 3D de la nube de puntos
# Se trata de un modelado tridimensionales orientado a simulaciones acústicas en recintos interiores:
# - Modelos alámbricos 3D basados en líneas y polilíneas
# - Superficies planas y supficientemente grande en relación a la longitud de onda
# - Superfices almacenadas por capas

# Se distinguen dos etapas: detección y reconstrucción de la envolvente del ambiente industrial (Planos -Intersecciones -Vértices)
# y detección y reconstrucción de los obstáculos/maquinaria ( Clusterizado -Proyección - Extrusión)

"""
Created on Tue Mar 16 13:09:14 2021

@author: Dylan Otero González

Functions
"""
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import random
import time
import sympy
from sympy.solvers import solve
from sympy import  Symbol, sqrt
import operator
import math
from functools import reduce
from sympy.solvers import solve
from sympy import Point, Line, sqrt, Symbol
import sklearn.cluster
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d


#--------------------
#LECTURA Y VISUALIZACIÓN DE FICHERO EN .TXT
# -------------------

def nube_txt(file, etiqueta,skiprows = 1):
    
    """ El fichero .txt es cargado,para ser posteriormente visualizado.
    
    El método contiene 2 argumentos:
        
        file: nube de puntos en formato .txt
        etiqueta: número de habitación que se desea cargar
        skiprows: número de filas que se desean eliminar. Puede tomar los siguientes
                    valores:
                        0: No tiene marcados los índices: x,y,x,r,g,b ...
                        1: Tiene marcados los índices. 
    Devuelve:
        
        -Nube de puntos de la habitación escogida
        -Gráfico con la visualización de la nube de puntos (Ventana: Plots)
        """    
    # Cargar la nube habiendo eliminado la primera línea de títulos
    point_cloud= np.loadtxt(file, skiprows=1) 
    nube = []
    for i in range(len(point_cloud)):
        if int(point_cloud[i,3]) == etiqueta:
            nube.append(point_cloud[i])
    #Visualización nube
    outfile = open('minas_etiqueta.txt', 'w')
    for row in nube:
        for number in row:
            outfile.write(f'{number} ')
        outfile.write('\n')
    outfile.close()

    # txt a pcd --> objeto
    pcd = o3d.io.read_point_cloud('minas_etiqueta.txt',format='xyz')
    
    return pcd    
    

#------------------------
#ESTIMACIÓN DE NORMALES
#------------------------

def estimate_normals (pcd):
    
    """La función para estimar la normal calcula la normal para cada punto, buscando
    puntos adyacentes dentro de un radio de búsqueda de 0.3 m y un número 
    de vecinos máximo de 50. Para radios de búsqueda inferiores se obtienen peores reusltados.
    El valor nz hace referencia a la normal en el eje Z. Para superficies muy proximas a la horizontal, 
    es aproximadamente 1. Por tanto, para suelos, nz es practicamente 1, mientras que para techos, los cuales pueden 
    tener cubiertas inclinadas se conoce que la inclinación máxima será de 25º según el CTE. nz = acos(25º) = 0.9
    Para las paredes, totalmente verticales, nz es próximo a 0.
    
    El método contiene 1 argumento:
        
        pcd: nube de puntos a tratar
               
    Algunas de las normales estimadas no están orientadas de forma coherente,
    es por ello que, se propaga la orientación normal utilizando un árbol de 
    expansión mínimo
    
    Devuelve:
    
        -Nubes de puntos correspondientes a la orientación vertical y horizontal.
    """
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) 
    # Búsqueda de puntos adyacentes
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    # Orientación coherente de los vectores
    pcd.orient_normals_consistent_tangent_plane(100)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    normals = np.array(pcd.normals)  
    nz = 0.90
    horizontal_2, vertical_2 = [], []
    # Almacenamiento de puntos según su normal en el eje Z
    for i in range (len(normals)):
        if np.absolute(normals[i,2]) > nz:
             horizontal_2.append(pcd.points[i])
        elif np.absolute(normals[i,2]) < 0.05:
             vertical_2.append(pcd.points[i])
    # Convertir lista de puntos en objeto (nube de puntos)
    pcd_hor = o3d.geometry.PointCloud()
    pcd_hor.points = o3d.utility.Vector3dVector(horizontal_2)
    pcd_ver = o3d.geometry.PointCloud()
    pcd_ver.points = o3d.utility.Vector3dVector(vertical_2)
    return (pcd_hor,pcd_ver)

def sep_planes (pcd_hor,pcd_ver,distance_threshold,ransac_n,num_iterations,points):
    
    """Segmentación de geometrías a partir de nubes de puntos mediante el algoritmo RANSAC.
   
    El método contiene 5 argumentos:
        
        pcd_hor: define la nube de puntos con normal en Z superior a 0.9
        pcd_ver: define la nube de puntos con normal en Z inferior a 0.05
        
        distance_threshold: define la distancia máxima que un punto puede tener a un plano estimado
                            para ser considerado inlier. En este caso, para el cálculo de geometría externa,
                            interesa trabajar con un valor robusto que permita obviar pequeños obstáculos.
                            Valor predeterminado: 0.2
        ransac_n: define el número de puntos que se muestan aleatoriamente para
                    estimar un plano.
                    Valor predeterminado: 3
        num_iterations: define la frecuencia con la que se muestrea y verifica 
                        un plano aleatorio.
                        Valor predeterminado: 1000
        points: mínimo número de puntos necesarios para que se produzca la iteración
                del cálculo de planos
                Valor predeterminado: 100
                
    Con los 3 puntos escogidos aleatoriamente define la ecuación del plano según el
    modelo (a,b,c,d), de forma que, a cada punto (x,y,z) se obtiene:
                                  
                                    ax + by +cz + d = 0
    
    La función iterará tantas veces como se seleccione hasta obtener el plano
    que contenga mayor número de puntos dentro del umbral "distance_thres_hold"
    Primero se obtienen los planos horizontales a partir de la nube de puntos "pcd_hor",
    donde se clasifican entre suelo y techo. El suelo será quel plano que tenga el valor de Z 
    mínimo, mientras que para el techo, interesa almacenar 1 o varios planos, dependiendo si la cubierta
    es simple, a 2 aguas o a 4. Es por ello que se escogerán aquellos planos con puntos cuyo valor máximo de Z 
    sea el mayor, asumiendo como error el umbral "distance_threshold" a la hora de seleccionarlos
    
    Devuelve:
        
        -Visualización de los planos obtenidos
        -Nubes de puntos correspondientes a cada elemento de la geometría externa: Suelo, techo y paredes
      
        """
    color = [i for i in np.arange (0,1,0.1)] 
    #Lista de valores que puede optar un color entre 0 y 0.9. ELimino el blanco, puesto que 
    # no se distinguiría con el fondo.
    plane, list_in_h, list_out_h =[], [], [] #Listas vacías para insertar valores en el bucle
    list_out_h.insert(0,[pcd_hor]) #En la posición 0 interesa almacenar la nube inicial de partida
    i=0
    while  len(list_out_h[i][0].points) > points: #Puntos que quedan fuera son más de 100
        pcd_i = list_out_h[i][0] #La nueva nube de puntos será la nube de outliers de la iteración anterior
        plane_model, inliers = pcd_i.segment_plane(distance_threshold,ransac_n,num_iterations) #Segmentación de planos
        plane.insert(i, plane_model) #Agregar a la lista los planos obtenidos
        #Creación de valores aleatorios para asignar el atributo color a cada plano
        l = random.choice(color)
        m = random.choice(color)
        n = random.choice(color)
        list_in_h.insert(i,[pcd_i.select_by_index(inliers)]) #Agregar a lista de inliers los planos obtenidos
        list_in_h[i][0].paint_uniform_color([l,m,n]) #Dar el atributo de color
        i += 1
        list_out_h.insert(i, [pcd_i.select_by_index(inliers, invert=True)]) #Agregar a lista de outliers la nube de puntos sobrantes
            
    mean_z = []
    for i in range (len(list_in_h)):
        z_max = np.mean(np.asarray(list_in_h[i][0].points)[:,2])
        mean_z.append(z_max)
    floor_pcd = list_in_h[mean_z.index(min(mean_z))]
    
    ceiling_pcd = []
    for i in range(len(mean_z)):
        if (max(mean_z) - 2*distance_threshold) <= mean_z[i]:
            ceiling_pcd.append(list_in_h[i][0])
    ceiling_pcd = [ceiling_pcd[0]]         
    # Vertical
    plane, list_in_v, list_out_v =[], [], [] #Listas vacías para insertar valores en el bucle
    list_out_v.insert(0,[pcd_ver]) #En la posición 0 interesa almacenar la nube inicial de partida
    i, cont = 0, 0
    while  cont < 5000: 
        pcd_i = list_out_v[i][0] #La nueva nube de puntos será la nube de outliers de la iteración anterior
        plane_model, inliers = pcd_i.segment_plane(distance_threshold,ransac_n,num_iterations) #Segmentación de planos
        if np.abs(plane_model[2]) < 0.1:
            plane.insert(i, plane_model) #Agregar a la lista los planos obtenidos
            #Creación de valores aleatorios para asignar el atributo color a cada plano
            l = random.choice(color)
            m = random.choice(color)
            n = random.choice(color)
            list_in_v.append([pcd_i.select_by_index(inliers)]) #Agregar a lista de inliers los planos obtenidos
            list_in_v[i][0].paint_uniform_color([l,m,n]) #Dar el atributo de color
            i += 1
            list_out_v.append([pcd_i.select_by_index(inliers, invert=True)]) #Agregar a lista de outliers la nube de puntos sobrantes
        else: 
            cont += 1
    # list_in_v.pop(4)
    planes_dif = [floor_pcd,ceiling_pcd,list_in_v]
    # Añadir etiqueta al final de todo según color
    geometry = []
    l = 0
    for i in range(len(planes_dif)):
        for n in range(len(planes_dif[i])):
            if len(planes_dif[i]) == 1:
                try: #Porque el techo puede tener 1 o varios planos, podría dar error
                    placa = np.array(planes_dif[i][n].points)
                except:
                    placa = np.array(planes_dif[i][n][0].points)
                for m in range(len(placa)):
                    a = list(placa[m])
                    a.append(l)   
                    geometry.append(a)
            if len(planes_dif[i]) > 1:
                placa = np.array(planes_dif[i][n][0].points)
                for m in range(len(placa)):
                    a = list(placa[m])
                    a.append(l)   
                    geometry.append(a)
            l += 1    
    
    # Guardar en txt
    outfile = open('external_geometry.txt', 'w')
    for row in geometry:
        for number in row:
            outfile.write(f'{number} ')
        outfile.write('\n')
    outfile.close()
    
    # Visualizar
    pcd_global = []
    color = [i for i in np.arange (0,1,0.2)]
    for i in range (l):
        pcd_ext = nube_txt('external_geometry.txt',i) #Es un función de este script
        k = random.choice(color)
        m = random.choice(color)
        n = random.choice(color)
        pcd_ext.paint_uniform_color([k,m,n])
        pcd_global.append(pcd_ext)
    # o3d.visualization.draw_geometries(pcd_global)
    return  [floor_pcd,ceiling_pcd,list_in_v,pcd_global]

#-------------
# VERTEX
#-------------

def vertex (pcd, distance_threshold, planes_dif):
    
     ''' La función permite obtener los vértices de la geometría externa del habitáculo o espacio interior. A partir de los planos obtenidos 
     mediante el algoritmo RANSAC y su posterior clasificcación entre: techo, suelo y paredes; se obtienen los puntos de intersección.
     La intersección de 3 planos da como resultante un punto de intersección, de forma que, el sistema está condicionado a que el plano del
     techo o suelo siempre está involucrado en la resolución, por tratarse de planos adyacentes.
     Inicialmente se calcula las ecuaciones de los planos partiendo de las nubes de puntos, para permitir las resolución de 3 ecuaciones (3 planos)
     y tres incógnitas (x,y,z).
                                        Ecuación del plano: Ax + By + Cz + D = 0                               
                                        Plano: [A,B,C,D]
    
    Para obtener solo las soluciones de planos adyacentes se condiciona a que los puntos obtenidos estén dentro del rango mínimo y máximo de las
    coordenadas de la nube de puntos. A estos límites se le añade el umbral empleado en el algoritmo RANSAC, ya que forma parte del error asumido.
    También se almacenan los ínidices de los planos verticales (Paredes) para conocer cuales osn adyacentes. 
    
    El método contiene 3 argumentos:
        
        pcd: define la nube de puntos cargada 
        
        distance_threshold: define la distancia máxima que un punto puede tener a un plano estimado para ser considerado inlier.
                            Valor predeterminado: 0.2
        planes_dif: lista compuesta por sublistas correspondientes a las nubes de puntos de suelo, techo y paredes.
        
    Devuelve:
        
        -Vértices del espacio interior
        - Visualización de los vértices en conjunto con la nube de puntos
    '''
     floor_v = planes_dif[0]
     ceiling_h = planes_dif[1]
     vertical = planes_dif[2] 
         

    # Cálculos de los puntos máximos y mínimos de la nube y les añado el error/umbral que puede cometerse al crear el plano
     min_x = min(np.asarray(pcd.points)[:,0]) - 0.5
     min_y = min(np.asarray(pcd.points)[:,1]) - 0.5
     min_z = min(np.asarray(pcd.points)[:,2]) - 0.5
     max_x = max(np.asarray(pcd.points)[:,0]) + 0.5
     max_y = max(np.asarray(pcd.points)[:,1]) + 0.5
     max_z = max(np.asarray(pcd.points)[:,2]) + 0.5
          
    # Obtener ecuación de los planos: Suelo, techo y paredes
    # Ecuación de la forma Ax + By +Cz + D = 0
    # A,B,C pertenecen al vector normal al plano
    # Regularizo --> Suelo y techo C=1, paredes C=0
     p_vert = []
     p_floor, inliers = floor_v[0].segment_plane(0.1,3,100) #[A,B,C,D]
     p_ceiling, inliers = ceiling_h[0].segment_plane(0.1,3,100)
     p_floor[2] = 1
     for i in range(len(vertical)):
        p_vert_1, inliers = vertical[i][0].segment_plane(0.1,3,100)
        p_vert.insert(i, list(p_vert_1))
     for i in range(len(p_vert)):
        p_vert[i][2] = 0
        
    # =====================================================
    # INTERSECCIÓN DE UN SUELO PLANO CON LAS PAREDES
    # =====================================================
    
    # Con la ecuación de 3 planos se resuelve la ecuación y se obtiene el pto de intersección
     x = Symbol('x')
     y = Symbol('y')
     z = Symbol('z')
     planes_inter = [] #Lista de los índices de planos que van a intersectar
     floor_points, solution =  [], [] #Lista vacía para añadir putos de intersección
    #Conociendo las relaciones de adyacencia empiezo con el suelo + 2 planos
    #Bucle anidado para recorrer esos 2 planos (paredes)
    
     angle_degree, angle_round, angle_new = [], [], []
     suma_cont = 0
     for i in range(len(vertical)):
        for n in range(len(vertical)):
            if n < i: #Para que no sean el mismo plano y que no repita dos veces la misma combinación
                solution_i = solve([p_floor[0]*x + p_floor[1]*y + p_floor[2]*z + p_floor[3],
                                    p_vert[i][0]*x +  p_vert[i][1]*y +  p_vert[i][2]*z +  p_vert[i][3], 
                                    p_vert[n][0]*x + p_vert[n][1]*y + p_vert[n][2]*z + p_vert[n][3]],[x,y,z]) #3 ecuaciones de planos y 3 incógnitas
                if solution_i == []: None #Ocurre en caso de no encontrar intersección entre planos. Ej: paralelos
                else:
                    solution_i = list(solution_i.values()) #Pasar la solución de diccionario a lista
                    solution_i = [ float("%.2f" % elem) for elem in solution_i ] #Redondear elementos de la lista con 2 decimales
                    
                    if (solution_i[0] >= min_x  and solution_i[1] >= min_y and solution_i[2] >= min_z and 
                    (solution_i[0] <= max_x and  solution_i[1] <= max_y and solution_i[2] <= max_z)): 
                         #Para solo escoger los valores que hay dentro lo los límtes
                         #Permite descartar planos con tendencia a ser paralelos
                        floor_points.append (solution_i)
                        solution.append (solution_i)
                        a = [n,i]
                        planes_inter.append(a) #Lista para ir almacenando los planos que son adyacentes
                        #Calcular el ángulo que hay entre los dos planos adyacentes
                        normal_product = np.abs(p_vert[n][0]*p_vert[i][0]+ p_vert[n][1]*p_vert[i][1]+ p_vert[n][2]*p_vert[i][2])
                        normal_n =math.sqrt((p_vert[n][0])**2 + (p_vert[n][1])**2 + (p_vert[n][2])**2)
                        normal_i =math.sqrt((p_vert[i][0])**2 + (p_vert[i][1])**2 + (p_vert[i][2])**2)
                        alfa = math.acos (np.abs(normal_product)/(normal_n*normal_i))
                        degree = math.degrees(alfa)
                        angle_degree.append(float("%.2f" % degree))
                        #Redondeo el ángulo para que quede regularizado. Redondeo va de 10 en 10º
                        rounded = round(degree,0)
                        r_rad = math.radians(rounded)
                        angle_round.append(rounded)
                        if suma_cont < (len(p_vert)-1): #Esto evita que se modifique dos veces el útlimo plano a estudiar en el caso
                            #Puesto que el último plano tiene adyacentes dos planos que ya han sido regularizados y uno de ellos ya fue el plano de referencia para regularizar éste.
                            #Mediante la matriz de rotación en el eje z voy a modificar la orientación de la normal del plano i
                            #El plano n y los puntos de intersección obtenidos se toman como referencia
                            #La siguiente matriz de rotación en eje Z modifica la normal del plano i [A,B,C]=[Nx,Ny,Nz]
                            new_plane = [p_vert[n][0]*math.cos(r_rad) - p_vert[n][1]*math.sin(r_rad),p_vert[n][0]*math.sin(r_rad) + p_vert[n][1]*math.cos(r_rad),p_vert[n][2]]
                            #Calcular el parámetro independiente D de la ecuación del nuevo plano
                            #Con el punto de intersección y la normal de plano se calcula
                            D = Symbol('D')    
                            param_d = solve([new_plane[0]*solution_i[0] + new_plane[1]*solution_i[1] + new_plane[2]*solution_i[2] + D],[D])
                            param_d = list(param_d.values())
                            p_vert[i] = [new_plane[0],new_plane[1],new_plane[2],param_d[0]]
                            normal_product = p_vert[n][0]*p_vert[i][0]+ p_vert[n][1]*p_vert[i][1]+ p_vert[n][2]*p_vert[i][2]
                            normal_n =math.sqrt((p_vert[n][0])**2 + (p_vert[n][1])**2 + (p_vert[n][2])**2)
                            normal_i =math.sqrt((p_vert[i][0])**2 + (p_vert[i][1])**2 + (p_vert[i][2])**2)        
                            beta = math.acos (np.abs(normal_product)/(normal_n*normal_i))
                            degree_beta = math.degrees(beta)
                            angle_new.append(float("%.2f" % degree_beta))
                            suma_cont += 1 
                        else:
                            angle_new.append(360 - sum(angle_new))
                            
     print('\nLos ángulos que forman incialmente las paredes adyacentes son: \n\n'  f'{angle_degree} \n\n Los ángulos regularizados que forman las paredes adyacentes son: \n\n'  f'{angle_new}')
     
     # ================================================
     # INTERSECCIÓN DE 1 O 2 CUBIERTAS CON LAS PAREDES
     # ================================================
    #Regularizar techo respecto a la horizontal
     p_horiz = [0,0,1]
     
 
     normal_product = np.abs(p_ceiling[0]*p_horiz[0]+ p_ceiling[1]*p_horiz[1]+p_ceiling[2]*p_horiz[2])
     normal_n =math.sqrt((p_ceiling[0])**2 + (p_ceiling[1])**2 + (p_ceiling[2])**2)
     normal_i =math.sqrt((p_horiz[0])**2 + (p_horiz[1])**2 + (p_horiz[2])**2)
     alfa = math.acos (np.abs(normal_product)/(normal_n*normal_i))
     degree = math.degrees(alfa)                     
     print('\nEl ángulo que forma inicialmente el techo con respecto a la horizontal es: \n\n'  f'{degree}')
    # Redondeo el ángulo calculado. Rangos de 2º en 2º
     interval = np.linspace(0,15,151)            
     
     for x in interval:
        if x < len(interval)-1:
            if degree > x and degree < x + 0.1:
                if (degree - x) < (x +0.1 - degree):
                    rounded = x
                else:
                    rounded = x + 0.1
    #Pasar a radianes la diferencia entre redondeo y el ángulo, puesto que es lo que se debe rotar
     r_rad = math.radians((rounded - degree))
    # ========================================
    # VÉRTICES DEL TECHO
    # ========================================
    #Con los nuevos planos de las paredes ya puedo obtner los vértices superiores
    #Bucle anidado para recorrer esos 2 planos (paredes)
     x = Symbol('x')
     y = Symbol('y')
     z = Symbol('z')
    # Calculo el primer vértice a partir de las paredes regularizadas y el techo sin regularizar
    # Regularizo la inclinación del techo y a partir de la normal y el primer vértice obtengo el nuevo plano del techo regularizado
    # Lo sobreescribo sobre el plano inicial del techo
     cont = 0
     ceiling_points = []
     for i in range(len(vertical)):
        for n in range(len(vertical)):
            if n < i:
                solution_i = solve([p_ceiling[0]*x + p_ceiling[1]*y + p_ceiling[2]*z + p_ceiling[3],
                                    p_vert[i][0]*x +  p_vert[i][1]*y +  p_vert[i][2]*z +  p_vert[i][3], 
                                    p_vert[n][0]*x + p_vert[n][1]*y + p_vert[n][2]*z + p_vert[n][3]],[x,y,z])
                if solution_i == []: None
                else:
                    solution_i = list(solution_i.values())
                    solution_i = [ float("%.2f" % elem) for elem in solution_i ] #Redondear elementos de la lista con 2 decimales
                    if (solution_i[0] >= min_x  and solution_i[1] >= min_y and solution_i[2] >= min_z and 
                    (solution_i[0] <= max_x and  solution_i[1] <= max_y and solution_i[2] <= max_z)):
                        ceiling_points.append ([float(solution_i[0]),float(solution_i[1]),float(solution_i[2])])
                        solution.append ([float(solution_i[0]),float(solution_i[1]),float(solution_i[2])])
                        a = [n,i]
                        planes_inter.append(a)
                        if cont == 0:
                            #Para rotar la cubierta el método es distinto al de rotación de paredes (Nz=0). 
                            # Se debe a que es necesario emplear 2 matrices de rotación distintas, en el eje Z y X
                            # ===================================================
                            # PASO 1
                            # Rotar la normal en torno al eje Z hasta que Nx = 0
                            # Empleando la matriz de rotación en Z se desconocen los nuevos datos: Ny y teta
                            ny = Symbol('ny')
                            teta = Symbol('teta')
                            normal_rot_z = solve([p_ceiling[0]*sympy.cos(teta) - p_ceiling[1]*sympy.sin(teta), 
                                                 p_ceiling[0]*sympy.sin(teta) + p_ceiling[1]*sympy.cos(teta) - ny],[ny,teta])
                            ceil_rot_z = [0,normal_rot_z[0][0],p_ceiling[2]]    
                            teta_2 = normal_rot_z[0][1] 
                            #================================================== 
                            # PASO 2
                            # Ahora que Nx = 0, puedo trabajar con un Sistema de Coordenas Globales (SCG)
                            # Esto me permite  rotar la nueva normal hasta llegar a los 2º respecto a la horizontal
                            ny_2 = Symbol('ny_2')
                            nz_2 = Symbol('nz_2')
                            normal_rot_x = solve([ceil_rot_z[1]*sympy.cos(r_rad) - ceil_rot_z[2]*sympy.sin(r_rad) - ny_2, 
                                                 ceil_rot_z[1]*sympy.sin(r_rad) + ceil_rot_z[2]*sympy.cos(r_rad) - nz_2],[ny_2,nz_2])
                            normal_rot_x = list(normal_rot_x.values())
                            normal_rot_x = [ float( elem) for elem in normal_rot_x ] #Redondear elementos de la lista con 2 decimales
                            ceil_rot_x = [0, normal_rot_x[0], normal_rot_x[1]]
                            # ================================================
                            # PASO 3
                            # Abandono el SCG y regreso al sistema de coordenadas local que tenía incialmente
                            # Para ello roto la normal en torno al eje Z un ángulo -teta
                            nx_3 = Symbol('nx_3')
                            ny_3 = Symbol('ny_3')
                            normal_rot_z2 = solve([nx_3 + ceil_rot_x[1]*sympy.sin(-teta_2), 
                                                 ceil_rot_x[1]*sympy.cos(-teta_2) - ny_3],[nx_3,ny_3])
                            normal_rot_z2 = list(normal_rot_z2.values())
                            normal_rot_z2 = [ float( elem) for elem in normal_rot_z2 ] #Redondear elementos de la lista con 2 decimales
                            normal_ceil = [normal_rot_z2[0], normal_rot_z2[1], normal_rot_x[1]]
                            
                            # Comprobar que quedó inclinado correctamente
                            normal_product = np.abs(normal_ceil[0]*p_horiz[0]+ normal_ceil[1]*p_horiz[1]+normal_ceil[2]*p_horiz[2])
                            normal_n =math.sqrt((normal_ceil[0])**2 + (normal_ceil[1])**2 + (normal_ceil[2])**2)
                            normal_i =math.sqrt((p_horiz[0])**2 + (p_horiz[1])**2 + (p_horiz[2])**2)
                            alfa = math.acos (np.abs(normal_product)/(normal_n*normal_i))
                            degree_2 =  float("%.2f" % math.degrees(alfa))
                                                    
                            # Constuir el plano con la normal y un vértice
                            D = Symbol('D')    
                            param_d = solve([normal_ceil[0]*solution_i[0] + normal_ceil[1]*solution_i[1] + normal_ceil[2]*solution_i[2] + D],[D])
                            param_d = list(param_d.values())
                            p_ceiling = [normal_ceil[0],normal_ceil[1],normal_ceil[2],param_d[0]]
                            cont +=1        
     print('\nEl ángulo que forma el techo con respecto a la horizontal tras la regularización es: \n\n'  f'{degree_2}')
    # Vértices del suelo a la misma altura (MEDIA)
    # Modificate z_values to a comun z_mean
    # Floor
     z_changes_floor = []
     for i in range(len(floor_points)):
        z_changes_floor.append(floor_points[i][2])
     mean_z_floor = np.mean(z_changes_floor)   
     for i in range(len(floor_points)):
        floor_points[i][2] = float("%.2f" % mean_z_floor)
    
    # El primer vértice no coincide la X del suelo con la del techo. Hago la media usando 2*dist_threshold
     for i in range(len(solution)):
         for n in range(len(solution)):
             if n >= i:
                 if solution[i][0] <= (solution[n][0] + 2*distance_threshold) and solution[i][0] >= (solution[n][0] - 2*distance_threshold):
                      
                     solution[i][0] =  float("%.2f" % np.mean([solution[i][0],solution[n][0]]))
                     solution[n][0] = solution[i][0]
                 if solution[i][1] <= (solution[n][1] + 2*distance_threshold) and solution[i][1] >= (solution[n][1] - 2*distance_threshold):
                     solution[i][1] =  float("%.2f" % np.mean([solution[i][1],solution[n][1]]))
                     solution[n][1] = solution[i][1]
     solution  
    
            
    # Conocer los 4 vértices de cada pared
     vertex_plano = []
     for n in range(len(vertical)):
        vertex_plano_i = []
        for i in range(len(planes_inter)):
            if n in planes_inter[i]:
                vertex_plano_i.insert (n,(solution[i]))
        vertex_plano.insert(n, list(vertex_plano_i))
    
    
    # =========================
    # GUARDAR VÉRTICES
    # =========================
     outfile = open('vertex_fundicion.txt', 'w')
     for row in solution:
        for number in row:
            outfile.write(f'{number} ')
        outfile.write('\n')
     outfile.close()
    
    # .txt a pcd
     vertex = o3d.io.read_point_cloud('vertex_fundicion.txt',format='xyz')
     vertex.paint_uniform_color([1,0,0])
     pcd.paint_uniform_color([1,1,0.5])
     vertex_final = o3d.geometry.VoxelGrid.create_from_point_cloud(vertex,0.15)
     o3d.visualization.draw_geometries([pcd,vertex_final])
     pcd.paint_uniform_color([0,1,0])
    
    # ====================
    # ORDENAR COORDENADAS
    # ====================
    #Ordenar coordenadas de los vértices de cada cada mediante giro horario
     pp = []
     vertex_order = []
     for i in range(len(vertex_plano)):
       pp = vertex_plano[i]
       # centroid
       cent=(sum([p[0] for p in pp])/len(pp),sum([p[1] for p in pp])/len(pp),sum([p[2] for p in pp])/len(pp))
       # sort by polar angle
       pp.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
       pp.sort(key=lambda p: math.atan2(p[2]-cent[2],p[1]-cent[1]))
       vertex_order.insert(i,pp)
       
     middle = int(len(solution)/2)
     floor_points = solution[0:middle]   
     ceiling_points = solution[middle:len(solution)]
     
    #SUELO
     for i in range(len(floor_points)):
       center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), floor_points), [len(floor_points)] * 2))
       floor_order = sorted(floor_points, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    
    #TECHO
     for i in range(len(ceiling_points)):
       center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), ceiling_points), [len(ceiling_points)] * 2))
       ceiling_order = sorted(ceiling_points, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    
     # for i in range(len(floor_order)):
     #      for n in range(len(ceiling_order)):
     #          if n >= i:
     #              if floor_order[i][0] < (ceiling_order[n][0] + 2*distance_threshold) and floor_order[i][0] > (ceiling_order[n][0] - 2*distance_threshold):
     #                  floor_order[i][0] =  float("%.2f" % np.mean([floor_order[i][0],ceiling_order[n][0]]))
     #                  ceiling_order[n][0] = floor_order[i][0]
     #              if floor_order[i][1] < (ceiling_order[n][1] + 2*distance_threshold) and floor_order[i][1] > (ceiling_order[n][1] - 2*distance_threshold):
     #                  floor_order[i][1] =  float("%.2f" % np.mean([floor_order[i][1],ceiling_order[n][1]]))
     #                  ceiling_order[n][1] = floor_order[i][1]
                    
     p_surf = p_vert
     p_surf.append(p_ceiling)
     p_surf.append(list(p_floor))
     return [p_surf,solution,planes_inter,floor_order,ceiling_order,vertex_order,vertex_final,p_vert]       
    
# ==========================
# Clusterizado de obstáculos
# ==========================
 
def obstacles(obstac,floor_order):
    
    """ A partir de los puntos que forman parte de los obstáculos se clasifican mediante 
    un clusterizado con el algoritmo DBSCAN, el cual permite agrupar los puntos gracias a 
    dos parámetros fundamentales:
        Epsilon: radio de la esfera de búsqueda de puntos
        Min_points: mínimo número de puntos que debe haber dentro de la esfera para ser considerado 
                    como un cluster
    Valores muy reducidos de epsilon implica la no formación de clusters, dando como resultado
    muchos puntos pertenecientes a ruido, mientras que valores muy elevados implican la unión 
    de clusters cercanos.
    En entornos grandes, como pueden ser los industriales, es recomendable emplear valores elevados
    para el parámetro min_points.
    
    El método consiste en clusterizar la nube de puntos que forman los obstáculos a modo de primer filtrado,
    y posteriormente realizar un segundo clusterizado sobre cada uno de ellos.
    Una vez obtenidos los clusters se calcula el valor mínimo de la coordenada Z de cada uno de ellos y se 
    proyectan todos sus puntos sobre ésta.
    A continuación, se genera el contorno exterior con estos puntos y se extruye hasta su máxima 
    coordenada en Z.
    
    El método contiene 1 único argumento:
            
        obstac: define la nube de puntos de los obstáculos. Se obtiene a partir del cálculo de puntos interiores
        a los planos pertenecientes a la geometría exterior.
        floor_order: lista con las coordenadas del suelo de la nave.
    
    Devuelve:
        
        - Lista que contiene las coordenadas de cada superficie de todos los obstáculos.
        """
    cloud =obstac
    # Con esto obtengo los índices de los clusteres, si es -1 implica ruido
    obstacleLabels = np.array(cloud.cluster_dbscan(eps=0.3, min_points=60))
    #Para la nave industrial, que es un conjunto de datos grande y ruidoso conviene aumentar el
    #min_samples, mientras que eps controla la vecindad local de los puntos, con valores muy bajos 
    #los puntos no se agrupan y se etiquetan como ruido -1. Con valores muy altos los clústeres cercanos se fusionan 
    max_label = obstacleLabels.max()
    colors = plt.get_cmap("tab20")(obstacleLabels / (max_label if max_label > 0 else 1))
    colors[obstacleLabels < 0] = 0
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #Delete outliers of DBSCAN:
    #outliers marked with -1
    ind = np.where(obstacleLabels==-1)
    obstac_1 = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud,0.02)
    o3d.visualization.draw_geometries([obstac_1])
    
    max(obstacleLabels)
  
    # Clasificación de las etiquetas
    
    obstacles = np.loadtxt("obstacles.txt", skiprows=1)
    
    # Añadir etiquetas a la nube
    obs_new = []
    for i in range(len(obstacleLabels)-1):
        row = list(obstacles[i])
        row.append(obstacleLabels[i])
        obs_new.append(row)
    vertex_global = []  
    for n in range (max(obstacleLabels)):
        cluster = []
        for i in range(len(obs_new)):
            if obs_new[i][3] == n:
                cluster.append([obs_new[i][0],obs_new[i][1],obs_new[i][2]])
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
        cloud_2 = cluster_pcd
        obstacleLabels_2 = np.array(cloud_2.cluster_dbscan(eps=0.25, min_points=120))
        max_label = obstacleLabels_2.max()
        colors = plt.get_cmap("tab20")(obstacleLabels_2 / (max_label if max_label > 0 else 1))
        colors[obstacleLabels_2 < 0] = 0
        cloud_2.colors = o3d.utility.Vector3dVector(colors[:, :3])
        ind = np.where(obstacleLabels_2==-1)
        max(obstacleLabels_2)
        obs_2_new = []
        # Añadir índice
        if max(obstacleLabels) == 0:
            row = list(cluster[i])
            row.append(obstacleLabels_2[i])
            obs_2_new.append(row)
        if max(obstacleLabels_2) >= 0:
            for i in range(max(obstacleLabels_2)):
                row = list(cluster[i])
                row.append(obstacleLabels_2[i])
                obs_2_new.append(row)
            vertex_obs_cluster = []
            for m in range(max(obstacleLabels_2)):
                cluster_2 = []
                for l in range(len(obs_2_new)):
                    if obs_2_new[l][3] == m:
                        cluster_2.append([obs_2_new[l][0],obs_2_new[l][1],obs_2_new[l][2]])
                if cluster_2 != []: 
                    if len(cluster_2) > 100:
                        z_min = np.min(np.array(cluster_2)[:,2])
                        z_max = float("%.2f" % (np.max(np.array(cluster_2)[:,2])))
                        cluster_proj = []
                        for b in range(len(cluster_2)):
                          cluster_proj.append([cluster_2[b][0],cluster_2[b][1]])  
                        # from scipy.spatial import ConvexHull, convex_hull_plot_2d
                        cluster_array = np.array(cluster_proj)
                        hull = ConvexHull(cluster_array)    
                        hull.vertices
                        # Agregar vértices a la lista
                        obs_vert = []
                        for c in range(len(hull.vertices)):
                            obs_vert.append(cluster_2[hull.vertices[c]])
                        # Mismo valor de z que el suelo en caso de ser inferior
                        if z_min < floor_order[0][2]:
                            z_min = floor_order[0][2] 
                        # Con z_min y z_max crear el suelo y techo del cluster 
                        obs_floor, obs_ceil  = [], []
                        for v in range(len(obs_vert)):
                            obs_floor.append([obs_vert[v][0],obs_vert[v][1],z_min])
                            obs_ceil.append([obs_vert[v][0],obs_vert[v][1],z_max])
                        #Nueva lista que incluya todas la superficies del cluster
                        # Añadir las superficies laterales
                        vertex_obs = [obs_floor,obs_ceil]
                        for z in range(len(obs_vert)):
                            if z < (len(obs_vert)-1):
                                vertex_obs.append([obs_floor[z],obs_floor[z+1],obs_ceil[z+1],obs_ceil[z]])
                            else:
                                vertex_obs.append([obs_floor[z],obs_floor[0],obs_ceil[0],obs_ceil[z]])
                        vertex_obs_cluster.append(vertex_obs)  
                    else: None
        vertex_global.append(vertex_obs_cluster)
    vertex_obs_sep = []
    for i in range(len(vertex_global)):
        for m in range(len(vertex_global[i])):
            vertex_obs_sep.append(vertex_global[i][m])
    return vertex_obs_sep    
    

# =======================
# GUARDAR EN FORMATO DXF
# =======================
# Lineas de código para generar el DXF a partir de los vértices (No pertenece a una función)
dwg = dxf.drawing('fundicion_todos_obst.dxf')
#Create walls
for i in range(len(vertex_order)):
    polyline = dxf.polyline(vertex_order[i], color = 2, layer = f'Wall {i}')
    polyline.close()
    dwg.add(polyline)
#Create Floor
polyline = dxf.polyline(floor_order, color =3, layer = 'Floor')
polyline.close()
dwg.add(polyline) 

#Create Ceiling   
polyline = dxf.polyline(ceiling_order, color = 4, layer = 'Ceiling')
polyline.close()
dwg.add(polyline) 

#Create Door 
polyline = dxf.polyline(door_coord[0:4], color = 5, layer = 'Door')
polyline.close()
dwg.add(polyline) 

polyline = dxf.polyline(door_coord[4:8], color = 7, layer = 'Gate')
polyline.close()
dwg.add(polyline) 

polyline = dxf.polyline(door_coord[8:12], color = 6, layer = 'Windows')
polyline.close()
dwg.add(polyline) 


polyline = dxf.polyline(door_coord[12:16], color = 6, layer = 'Windows')
polyline.close()
dwg.add(polyline) 

#Create obst
for i in range(len(vertex_obs_sep)):
    polyline = dxf.polyline(vertex_obs_sep[i], color = 152, layer = 'Obstacle')
    polyline.close()
    dwg.add(polyline)
    
dwg.save()
      
    
        










