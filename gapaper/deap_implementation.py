import csv
import sys
import datetime as dt
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from operator import attrgetter
from scoop import futures

def array_to_matrix(conn_array):
    '''
    Takes an array (1 x n*(n-1)/2), and outputs an incidence matrix (n x n)
    '''
    #print(conn_array)
    array_len = len(conn_array)
    n = int(math.sqrt(array_len * 2)) + 1
    conn_matrix = np.zeros((n,n), dtype=np.uint8)
    
    k = 0
    for i in range(n-1):
        for j in range(i+1,n):
            conn_matrix[i][j] = conn_array[k]
            conn_matrix[j][i] = conn_array[k]
            k += 1
    
    return conn_matrix

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = np.random.randint(1, size)
    cxpoint2 = np.random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
def eaSimplePlus(population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__, logfile=None):
    '''
    Copied from deap source, with modifications:
    -to output the individuals (gen_best, a list of the best individuals by gen)
    '''
    gen_best = []
    best_indiv = min(population, key=attrgetter('fitness.values'))
    gen_best.append(best_indiv)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    toprint = ' '.join([logbook.stream, dt.datetime.strftime(dt.datetime.now(), '%X')])
    if verbose:
        print(toprint)
    #Write to txt log
    if logfile:
        with open(logfile + '.txt', 'a') as f:
            f.write(toprint + '\n')

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = varAndPlus(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        toprint = ' '.join([logbook.stream, dt.datetime.strftime(dt.datetime.now(), '%X')])
        if verbose:
            print(toprint)
        #Write to txt log
        if logfile:
            with open(logfile + '.txt', 'a') as f:
                f.write(toprint + '\n')
        
        #Record the best individual
        best_indiv = min(population, key=attrgetter('fitness.values'))
        gen_best.append(best_indiv)

    return population, logbook, gen_best

def evaluate_solution(conn_array, test_system, startnode=0, display=False):
    '''
    This function takes a solution as input, and evaluates its fitness.
    Negative attributes are: total cost (investment + operation), loops
    Positive attributes are: connectivity, good voltage
    '''
    #Several tests are built already
    #DONE: parameter n mismatch, isolated nodes, too many links, FBS
    #SCORING: 50 points for connectivity
    #         50 points for tree structure (not sure what happens if not a tree... FBS might asplode)
    #         100 points for voltage goodness
    
    def forward_backward(conn_matrix, test_system, startnode):
        '''
        Given a network, calculates the voltages & currents
        '''
        #print(conn_matrix)
        
        #print(nx.is_connected(nx.to_networkx_graph(conn_matrix)))
        n = len(conn_matrix)
        num_periods = len(test_system['demands'][0])
        max_num_iterations = 10
        nom_voltage = test_system['voltage']
        conv_thresh = 1         #FBS convergence check, V
        
        #build z matrix
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if conn_matrix[i][j] > 0:
                    per_mile_imp = test_system['resistances'][conn_matrix[i][j]]
                    Z[i][j] = per_mile_imp * test_system['distances'][i][j]
        
        #build upstream/downstream lists
        upstream = {i:None for i in range(n)}
        downstreams = {i:None for i in range(n)}
        #nodes which have have their downstreams already determined
        visited = set()
        #current edges of network (starting at center, working out)
        exterior = {0}
        num_visited = len(visited)
        prev_num_visited = -1
        while prev_num_visited != num_visited:  #check if we've finished exploring
            new_exterior = set()
            for node in exterior:
                # record child nodes
                connected = [i for i in range(n) if conn_matrix[node][i] > 0]
                downstreams[node] = list(set(connected).difference(visited))
                #print(node, 'downstream', repr(downstreams[node]))
                visited.add(node)
                for i in downstreams[node]:
                    # record parent node
                    upstream[i] = node
                new_exterior |= set(downstreams[node])
            exterior = new_exterior
            prev_num_visited = num_visited
            num_visited = len(visited)
            #print(num_visited, '/', n)
        #print('finished building up/downstreams',num_visited,'/',n)
        
        #matrix for (node #, iteration)
        V_fwd = np.zeros((n, max_num_iterations))
        for i in range(n):      #initialize voltages to 1
            V_fwd[i,0] = nom_voltage
        V_bwd = V_fwd.copy()
        #matrix for (node #, iteration). Current downstream of node n
        I_fwd = np.zeros((n, max_num_iterations))
        I_bwd = np.zeros((n, max_num_iterations))
        
        #print(downstreams)
        end_nodes = [i for i in range(n) if downstreams[i] == []]
        #print('end_nodes', end_nodes)
        
        V_solution = np.zeros((n,num_periods))
        I_solution = np.zeros((n*(n-1)/2,num_periods))
        losses = np.zeros(num_periods)
        
        for t in range(num_periods):
            curr_iter = 1
            for curr_iter in range(1,max_num_iterations):
                #print('FBS iter.', curr_iter)
                
                #Backward sweep
                updated = set()
                for e in end_nodes:
                    #calculate current at edges, using prev. iteration voltages
                    V_bwd[e][curr_iter] = V_fwd[e][curr_iter-1]
                    I_bwd[e][curr_iter] = demands[e][t] / V_fwd[e][curr_iter-1]
                    updated.add(e)
                #print('num updated', len(updated), '/', n)
                #print(I_bwd[:,curr_iter])
                while len(updated) < n:
                    #work upstream, calculating currents/voltages from prev. Fwd
                    for i in range(n):
                        #print(max(I_bwd[:,curr_iter]))
                        if not (i in updated) and (set(downstreams[i]) < updated):
                            # if we haven't updated the node, but its children have updated
                            # pick random child to use to set voltage
                            update_node = np.random.choice(downstreams[i])
                            voltage_drop = I_bwd[update_node][curr_iter] * test_system['resistances'][conn_matrix[i][update_node]]
                            V_bwd[i][curr_iter] = V_bwd[update_node][curr_iter] + voltage_drop
                            # calc current using this new voltage
                            demand_curr = demands[i][t] / V_bwd[i][curr_iter]
                            downstream_curr = np.sum([I_bwd[e][curr_iter] for e in downstreams[i]])
                            total_curr = demand_curr + downstream_curr
                            I_bwd[i][curr_iter] = total_curr
                            updated.add(i)
                    #print('num updated', len(updated), '/', n)
                #Forward sweep
                #Start with known source voltage
                V_fwd[0][curr_iter] = nom_voltage
                updated = {0}
                #Use current from this Bwd sweep to find voltage at downstream nodes
                while len(updated) < n:
                    for i in range(n):
                        if not (i in updated) and (upstream[i] in updated):
                            #if we haven't updated the node, but we know the upstream voltage
                            voltage_drop = test_system['resistances'][conn_matrix[i][upstream[i]]] * I_bwd[i][curr_iter]
                            V_fwd[i][curr_iter] = V_fwd[upstream[i]][curr_iter] - voltage_drop
                            updated.add(i)
                
                #check for convergence
                V_diff = V_fwd[:,curr_iter] - V_fwd[:,curr_iter-1]
                V_diff = [abs(v) for v in V_diff]
                if max(V_diff) < conv_thresh:
                    #print('convergence on', curr_iter)
                    #print(V_diff)
                    #print(V_fwd[:,curr_iter])
                    break
            #print(V_diff)
            
            V_solution[:,t] = V_fwd[:,curr_iter]
            #Old useless line, leaving it here in case i fuck up
            #I_solution[:,t] = I_bwd[:,curr_iter]
            
            #calculate losses
            for i in range(n-1):
                for j in range(i+1,n):
                    if conn_matrix[i][j] > 0:
                        V_drop = V_solution[i][t] - V_solution[j][t]
                        resistance = test_system['resistances'][conn_matrix[i][j]]
                        branch_loss = V_drop**2 / resistance
                        losses[t] += branch_loss
            
            #translate currents downstream from node into currents in each line
            current_matrix = np.zeros((n,n))
            for i in range(n):
                if upstream[i] is not None:
                    current_matrix[i][upstream[i]] = I_bwd[i,curr_iter]
            array_length = n * (n-1) / 2
            current_array = np.zeros(array_length)
            k = 0
            for i in range(n-1):
                for j in range(i+1,n):
                    current_array[k] = current_matrix[i][j]
                    k += 1
            I_solution[:,t] = current_array
        
        return V_solution, I_solution, losses
    
    def system_cost(conn_matrix, test_system):
        '''
        Calculates the cost of building the system
        '''
        n = len(conn_matrix)
        cost = 0
        
        for i in range(n):
            for j in range(n):
                if conn_matrix[i][j] > 0:
                    try:
                        line_cost = test_system['weights'][conn_matrix[i][j]]
                        cost += test_system['distances'][i][j] * line_cost
                    except IndexError:
                        print('bad line choice', conn_matrix[i][j])
                        print(test_system['weights'])
                        print(conn_matrix, type(conn_matrix))
                        print(n)
                        print(conn_matrix[i])
                        print(i,j)
                        quit()
        return cost

    conn_matrix = array_to_matrix(conn_array)
    
    system_good = True
    error_msgs = []
    score = 0
    
    #load parameters from dict
    coordinates = test_system['coordinates']
    #add zero-demand for all time periods for substation
    t = len(test_system['demands'][0])
    demands = np.concatenate([np.zeros((1,t)), test_system['demands']])
    distances = test_system['distances']
    
    n = len(conn_matrix)
    
    invest_cost = system_cost(conn_matrix, test_system)
    score = invest_cost
    
    #check forward-backward solution
    voltages, currents, loss = forward_backward(conn_matrix, test_system, startnode)
    voltage_floor = test_system['voltage'] * (1 - test_system['v_range'])
    #for each node, for each time period, 1 if voltage is violated
    voltage_violations = [1 if j < voltage_floor else 0 for i in voltages for j in i]
    current_limits = [test_system['capacities'][int(i)] for i in conn_array]
    #for each line, for each time period, 1 if current is violated
    current_violations = [1 if (j > current_limits[i]) else 0 for i, curr in enumerate(currents) for j in curr]
    #print(np.amax(currents))
    #print(sum(current_violations), len(current_violations))
    #print(len(voltage_violations), sum(voltage_violations))
    current_penalty = 100 * sum(current_violations)
    voltage_penatly = 100 * sum(voltage_violations)
    loss_penalty = np.mean(loss) * test_system['loss_cost']
    score += (voltage_penatly + loss_penalty + current_penalty)
    
    #normalize for number of nodes
    score /= n
    
    if display:
        print('score', score, 'invest', invest_cost / n, 'loss', loss_penalty / n, 'voltage', voltage_penatly / n, 'current', current_penalty / n)
        #print(voltages)
    return (score,)

def fix_bad_tree(child):
    child_copy = child.copy()
    #print('input max', np.amax(child))
    
    #print('checking child', np.count_nonzero(child), 'links')
    #print(child, type(child))
    conn_matrix = array_to_matrix(child)
    nx_matrix = nx.to_networkx_graph(conn_matrix)
    n = len(conn_matrix)
    all_vertices = set(range(n))
    
    #check for disconnected sections, fix
    gdict = matrix_to_dict(conn_matrix)
    #print(gdict)
    num_tries = int(n / 5)
    for i in range(num_tries):
        if nx.is_connected(nx_matrix):
            break
        #print('not connected, connecting')
        
        connected_component = nx.node_connected_component(nx_matrix,0)
        connect_from = np.random.choice(connected_component)
        #print(connected_component)
        #print(set(connected_component))
        connect_to = np.random.choice(list(all_vertices.difference(set(connected_component))))
        nx_matrix.add_edge(connect_from, connect_to, weight=1)
    else:
        #I've tried n/5 times to fix this graph, but there's no hope
        print('can\'t fix graph, generating new one')
        conn_matrix = generate_spanning_tree_matrix(n)
        nx_matrix = nx.to_networkx_graph(conn_matrix)
    
    #check for extra links, causing cycles
    while len(nx_matrix.edges()) > (n-1):
        test_edges = nx_matrix.edges()
        np.random.shuffle(test_edges)
        for edge in test_edges:
            test_nx_matrix = nx_matrix.copy()
            test_nx_matrix.remove_edge(*edge)
            if nx.is_connected(test_nx_matrix):
                nx_matrix = test_nx_matrix.copy()
                break
                
    conn_matrix = np.array(nx.to_numpy_matrix(nx_matrix), dtype=np.uint8)
    conn_array = matrix_to_array(conn_matrix)
    conn_matrix = array_to_matrix(conn_array)
    nx_matrix = nx.to_networkx_graph(conn_matrix)
    if np.amax(conn_matrix) > 4:
        print('bad fix_bad_tree', np.amax(conn_matrix))
        quit()
    if not nx.is_connected(nx_matrix):
        print('WTF??????  1')
        quit()
    #print(conn_matrix, type(conn_matrix))
    conn_array = matrix_to_array(conn_matrix)
    child = conn_array
    #print(conn_array, type(conn_array))
    #print(child == child_copy)
    
    return child
   
def generate_minimum_spanning_tree(distances):
    '''
    '''
    n = len(distances)
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=distances[i,j])
    T = nx.minimum_spanning_tree(G)
    #nx.draw_networkx(T)
    #plt.show()
    conn_matrix = np.array(nx.to_numpy_matrix(T))
    #print(np.sum(conn_matrix))
    conn_array = matrix_to_array(conn_matrix)
    #print(sum(conn_array))
    
    #Convert to 1/0
    conn_array = (conn_array > 0).astype(int)
    #print(sum(conn_array))
    #multiply by random 1-4
    conn_array = np.multiply(conn_array, np.random.randint(1,5,len(conn_array)))
    
    #print(sum(conn_array))
        
    #print(n, len(conn_array), conn_array)
    return conn_array
   
def generate_spanning_tree_matrix(n, min_level=1, max_level=1):
    '''
    Given a size n, generates a spanning tree.
    Edge values are picked randomly between min_level and max_level
    '''
    conn_matrix = np.zeros((n, n), dtype=np.uint8)
    visited = [0]
    unvisited = [i for i in range(1,n)]
    while len(unvisited) > 0:
        new_edge_source = np.random.choice(visited)
        new_edge_dest = np.random.choice(unvisited)
        if min_level != max_level:
            level = np.random.randint(min_level, max_level+1)
        else:
            level = min_level
        conn_matrix[new_edge_source][new_edge_dest] = level
        conn_matrix[new_edge_dest][new_edge_source] = level
        visited.append(new_edge_dest)
        unvisited.remove(new_edge_dest)
    conn_matrix = creator.Individual(conn_matrix)
    return conn_matrix    

def generate_spanning_tree_array(n, min_level=1, max_level=1):
    '''
    Generates a spanning tree (matrix representation), converts to array
    '''
    conn_matrix = generate_spanning_tree_matrix(n, min_level, max_level)
    conn_array = matrix_to_array(conn_matrix)
    return conn_array
    
def generate_locations(n, width):
    '''
    Generate the locations of n nodes, and one substation, and the distances
    between them. Returns these distances as a numpy array.
    The substation is the 0th node and is by default in the center of the field
    '''
    substation_coordinates = np.array([[0.5,0.5]]) * width
    node_coordinates = np.random.random((n,2)) * width
    coordinates = np.concatenate((substation_coordinates, node_coordinates), axis=0)
    distances = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            dist = np.sqrt((coordinates[i][0] - coordinates[j][0])**2 + 
                           (coordinates[i][1] - coordinates[j][1])**2)
            distances[i,j] = dist
    return coordinates, distances
    
def load_test_system(n):
    '''
    Load test system from expected place
    '''
    def read_system_from_files(input_files):
        '''
        Given a dictionary of filenames, reads test system, returns parameters
        '''
        coordinates = []
        with open(input_files['coordinates']) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                coordinates.append([float(cell) for cell in row])
        coordinates = np.array(coordinates)
        
        demands = []
        with open(input_files['demands']) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                demands.append(float(row[0]))
        demands = np.array(demands)
        
        distances = []
        with open(input_files['distances']) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                distances.append([float(cell) for cell in row])
        distances = np.array(distances)
         
        parameters = {}
        with open(input_files['parameters']) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                parameters[row[0]] = [float(cell) for cell in row[1:]]
        single_parameters = ['voltage', 'v_range', 'loss_cost']
        for p in single_parameters:
            parameters[p] = parameters[p][0]
        
        test_system = {'coordinates':coordinates,
                       'demands':demands,
                       'distances':distances,
                       'parameters':parameters}
        return test_system
    
    input_names = ['coordinates', 'demands', 'distances', 'parameters']
    f1, f2 = 'test_systems', str(n) + '_bus'      #folder names
    input_files = {name: os.path.join(f1,f2,name+'.csv') for name in input_names}
    try:
        test_system = read_system_from_files(input_files)
    except FileNotFoundError:
        print('Error: no test system with', n, 'nodes found in expected location')
        quit()
    return test_system
    
def matrix_to_array(conn_matrix):
    '''
    Takes an incidence matrix (n x n), and outputs an array (1 x n*(n-1)/2),
    consisting of the strictly upper entries of the matrix 
    '''
    n = len(conn_matrix)
    array_length = n * (n-1) / 2
    conn_array = np.zeros(array_length)
    k = 0
    for i in range(n-1):
        for j in range(i+1,n):
            conn_array[k] = conn_matrix[i][j]
            k += 1
    conn_array = creator.Individual(conn_array)
    return conn_array

def matrix_to_dict(conn_matrix):
    n = len(conn_matrix)
    graph = {}
    for i in range(n):
        connected_nodes = [j for j in range(n) if conn_matrix[i][j] > 0]
        graph[i] = connected_nodes
    return graph

def mutate_spanning_tree(spanning_tree):
    '''mutates the MST, without having to re-generate it for each individual'''
    conn_array = fix_bad_tree(mutate_toplevel(spanning_tree)[0])
    return conn_array

def mutate_toplevel(child):
    '''
    Given a child, mutate it in one of n ways
    '''
    def mutate_topology(child):
        '''
        break network, remake network
        '''
        nx_matrix = nx.to_networkx_graph(array_to_matrix(child))
        edges = list(nx_matrix.edges())
        remove_num = np.random.choice(len(edges))
        to_remove = edges[remove_num]
        nx_matrix.remove_edge(*to_remove)
        conn_matrix = np.array(nx.to_numpy_matrix(nx_matrix), dtype=np.uint8)
        conn_array = matrix_to_array(conn_matrix)
        child = conn_array
        
        return child

    def mutate_weights(child):
        '''
        Currently mutating weight only on non-zero branches
        '''
        
        n = len(child)
        
        non_zero_entries = np.nonzero(child)
        addition = np.zeros(n)
        for i in non_zero_entries:
            addition[i] = np.random.randint(0,2)
        child += addition
        
        subtraction = np.zeros(n)
        for i in range(n):
            subtraction[i] = np.random.randint(-1,1) if child[i] > 1 else 0
        child -= subtraction
        
        
        #print(child, type(child))
        #child = np.array(child, dtype=np.uint8)
        child = creator.Individual(child)
        #print(child, type(child))

        return child

    
    case = np.random.choice(['mutate_weights', 'mutate_topology'], p=[0.25,0.75])
    if case == 'mutate_weights':
        child = mutate_weights(child)
    elif case == 'mutate_topology':
        child = mutate_topology(child)
        
    #sanity check
    child = np.clip(child, 0, 4)
    
    return (child,)

def plot_network_from_array(child, coordinates, folder_name, title):
    #coordinates = {i:coordinates[i] for i in enumerate(coordinates)}
    conn_matrix = array_to_matrix(child)
    nx_matrix = nx.to_networkx_graph(conn_matrix)
    nx.draw_networkx_nodes(nx_matrix, pos=coordinates)
    for i in range(1,5):
        edges_to_draw = [(u,v) for (u,v,d) in nx_matrix.edges(data=True) if d['weight'] == i]
        #print(len(edges_to_draw))
        nx.draw_networkx_edges(nx_matrix, pos=coordinates,
                               edgelist=edges_to_draw,width=i)
    plt.title(title)
    plt.savefig(folder_name + '\\' + str(title) + '.png')
    plt.close()
    
def tree_check(test_system):
    '''
    This function assesses whether a newly-created individual is a tree or not
    If not, it will first connected any disconnected sectors,
    then remove any redundant links (links which create cycles
    '''
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            offspring = list(offspring)
            all_good = False
            offspring_copy = tuple(list(offspring))
            for i, child in enumerate(offspring):
                '''
                Actual child-fix code goes here
                '''
                #print(dir(child))
                child = fix_bad_tree(child)
                #print(dir(child))
                #print(type(child))
                child.fitness.values = evaluate_solution(child, test_system)
                offspring[i] = child
                        
            offspring = tuple(offspring)
            return offspring
        return wrapper
    return decorator

def varAndPlus(population, toolbox, cxpb, mutpb):
    '''
    Copied from deap source, with modifications to allow for parallelization.
    '''
    
    offspring = [toolbox.clone(ind) for ind in population]
    
    # Apply crossover and mutation on the offspring
    pair_nums = zip(range(0, len(offspring), 2), range(1, len(offspring), 2))
    pairs = [(offspring[i], offspring[j], cxpb) for i, j in pair_nums]
    crossed_offspring_pairs = toolbox.map(varAnd_crossover, pairs)
    offspring_w_mutpb = [(j, mutpb) for i in crossed_offspring_pairs for j in i]
    
    offspring = list(toolbox.map(varAnd_mutate, offspring_w_mutpb))

    return offspring

def varAnd_crossover(offspring_pair_w_cxpb):
    '''
    Copied from deap source, with modifications to allow for parallelization.
    '''
    offspring1, offspring2, cxpb = offspring_pair_w_cxpb
    if np.random.random() < cxpb:
        offspring1, offspring2 = toolbox.mate(offspring1, offspring2)
        del offspring1.fitness.values, offspring2.fitness.values
    return (offspring1, offspring2)

def varAnd_mutate(offspring1_w_mutpb):
    '''
    Copied from deap source, with modifications to allow for parallelization.
    '''
    offspring1, mutpb = offspring1_w_mutpb
    if np.random.random() < mutpb:
        offspring1, = toolbox.mutate(offspring1)
        del offspring1.fitness.values

    return offspring1
    
def main():
    
    if check_design:
        conn_matrix = np.zeros((n+1,n+1))
        with open(check_design, 'r') as f:
            for line in f:
                line = line.split(',')
                conn_matrix[int(line[0])][int(line[1])] = int(line[2])
                conn_matrix[int(line[1])][int(line[0])] = int(line[2])
        conn_array = matrix_to_array(conn_matrix)
        evaluate_solution(conn_array, test_system, display=True)
        quit()
    
    #print(test_system)
    print('n', n, 'pop', popsize, 'gens', num_gens, 'width', width, 'demands', demand_scale / 1000)
    
    pop = toolbox.population(n=popsize)
    hof = tools.HallOfFame(10, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    folder_name = ''.join(['n', str(n), 'pop', str(popsize), 'gen', str(num_gens)])
    if peak_demand:
        folder_name += '_peak'
    elif avg_demand:
        folder_name += '_avg'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    #Write parameters to txt log
    with open(folder_name + '.txt', 'w') as f:
        f.write(repr(test_system) + '\n')
    
    #Run GA
    population, logbook, gen_best = eaSimplePlus(pop, toolbox, cxpb=0.25,
                                                 mutpb=0.25, ngen=num_gens,
                                                 stats=stats, halloffame=hof,
                                                 logfile=folder_name)
    
    #Plotting
    #plot graphs add flag -p
    if "-p" in sys.argv:
        coordinates = test_system['coordinates']
        try:
            step = int(num_gens/10)
        except:
            step = 2
        for i in range(0,num_gens+1,step):
            plot_network_from_array(gen_best[i], coordinates, folder_name, title=i)
    
    evaluate_solution(gen_best[num_gens], test_system, display=True)
    
    #output line length and line weight file use flag -v
    if "-v" in sys.argv:
        dist_array = np.array(test_system['distances'])
        dnx_matrix = nx.to_networkx_graph(dist_array)
        mintree = np.array(nx.to_numpy_matrix(nx.minimum_spanning_tree(dnx_matrix)))
        mindist = np.sum(mintree)/2
        
        step = 1
        weightfile = open(folder_name + "_totallineweight.txt", 'w')
        weightfile.write("weighted,rawdistance(min=" + str(mindist) + "\n")
        for i in range(step,num_gens,step):
            #print(gen_best[i], gen_best[i].fitness.values)
            conn_matrix = array_to_matrix(gen_best[i])
            conn_array = gen_best[i]
            dist_array = matrix_to_array(test_system['distances'])
            dist_array[conn_array == 0] = 0
            totaldist = np.sum(dist_array)
            totaldistweight = np.sum(np.multiply(dist_array, conn_array))
            weightfile.write(str(totaldistweight) + "," + str(totaldist) + "\n")
    
    #write list of edges to file
    with open (folder_name + '_design.txt', 'w') as f:
        conn_matrix = array_to_matrix(gen_best[num_gens])
        nx_matrix = nx.to_networkx_graph(conn_matrix)
        edges = [(u,v,d['weight']) for (u,v,d) in nx_matrix.edges(data=True)]
        for e in edges:
            f.write(','.join([str(i) for i in e]) + '\n')
        
    #pop, stats, and hof are updated inplace
    
    print(n, popsize, num_gens)
                        
    return pop, stats, hof
    

n = 50
popsize = 200
num_gens = 500
width = 10
demand_scale = 4000 * 500/n
parallelized = True
peak_demand = False
avg_demand = False
#set this to None to ignore, or a filename to check score
check_design = None    

demand_profiles = ([0.10,0.10,0.10,0.20,0.25,0.10,0.60,0.70,0.85, #dual peak
                    0.90,0.85,0.60,0.40,0.30,0.40,0.60,0.70,0.85,
                    0.90,0.85,0.70,0.60,0.40,0.25],
                   [0.10,0.10,0.10,0.10,0.10,0.10,0.15,0.20,0.20, #evening peak
                    0.25,0.25,0.25,0.25,0.25,0.40,0.60,0.70,0.85,
                    0.90,0.85,0.70,0.60,0.40,0.25],
                   [0.10,0.10,0.10,0.20,0.25,0.40,0.60,0.70,0.85, #morning peak
                    0.90,0.85,0.60,0.40,0.25,0.20,0.15,0.10,0.10,
                    0.10,0.10,0.10,0.10,0.10,0.10])
t = len(demand_profiles[0])

#generate test system, same everytime thanks to tommy tutone
np.random.seed(8675309)
test_system = {'weights':[0,200,300,400,500],    #Cost/unit distance
               'resistances':[0, 0.8, 0.6, 0.4, 0.2],    #Ohms/unit distance
               'capacities':[0,50, 100, 150, 200], #Amps
               'voltage':7200,                   #Nominal voltage
               'v_range':0.05,                   #Maximum voltage deviation (p.u.)
               'loss_cost':10}
test_system['coordinates'], test_system['distances'] = generate_locations(n,width)
test_system['demands'] = np.array([demand_profiles[np.random.randint(3)] for i in range(n)])
test_system['demands'] += np.random.normal(0,0.1,size=(n,t))
test_system['demands'] = np.clip(test_system['demands'], 0, 1.2) * demand_scale

if peak_demand:
    test_system['demands'] = [[i[9]] for i in test_system['demands']]
elif avg_demand:
    test_system['demands'] = [[np.mean(i)] for i in test_system['demands']]
else:
    #keep original demands
    pass
#print(test_system['demands'])
t = len(demand_profiles[0])

#
np.random.seed()
# adapted from http://deap.gel.ulaval.ca/doc/default/tutorials/basic/part1.html
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

MST = generate_minimum_spanning_tree(distances=test_system['distances'])
#print('MST built')

# Here we are adding tools to our toolbox
toolbox = base.Toolbox()
#population tools
toolbox.register("individual", mutate_spanning_tree, spanning_tree=MST)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#evolution tools
toolbox.register("evaluate", evaluate_solution, test_system=test_system)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", mutate_toplevel)
toolbox.register("select", tools.selTournament, tournsize=3)
#sanity check tools
toolbox.decorate("mate", tree_check(test_system))
toolbox.decorate("mutate", tree_check(test_system))
#allow parallelization with scoop
if parallelized: toolbox.register("map", futures.map)

if __name__ == '__main__':
    
    pop, stats, hof = main()