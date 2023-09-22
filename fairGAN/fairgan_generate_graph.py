def discover_graph():
    column_names = ['male', 'age', 'debt', 'married', 'bankcustomer', 'educationlevel', 'ethnicity', 'yearsemployed',
                'priordefault', 'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income', 'approved']

    data = pd.read_csv('data/crx.data', header=None,  names=column_names)
    data.reset_index(drop=True, inplace=True) 

    data = data.dropna(how = 'all')
    data = data[data.age != '?']

    print( data.head() )

    for feat in ['male', 'married','bankcustomer', 'educationlevel', 'ethnicity','priordefault', 'employed', 'driverslicense', 'citizen', 'zip', 'approved']:
        data[feat] = preprocessing.LabelEncoder().fit_transform(data[feat])

    #####################################################
    #### For this experiment, we uniquely drop the default variable (prior default)
    ###################################################
    #data = data.drop(['educationlevel'], axis=1)
        
    from pycausal.pycausal import pycausal as pc
    pc = pc()
    pc.start_vm()

    from pycausal import prior as p
    from pycausal import search as s

    prior = p.knowledge(addtemporal = [['male', 'age','ethnicity'],[ 'debt', 'married', 'bankcustomer', 'educationlevel', 'yearsemployed',
                    'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income'],['approved']])

    tetrad = s.tetradrunner()
    tetrad.run(algoId = 'fges', scoreId = 'cg-bic-score', dfs = data, priorKnowledge = prior,
            maxDegree = -1, faithfulnessAssumed = True, verbose = False)
    tetrad.getEdges()

    edges = []
    for edge in tetrad.getEdges():
        edges.append(list([column_names.index(edge.split(' ')[0]), column_names.index(edge.split(' ')[-1])]))
    print(edges )

    # Copy the above edge list 
    column_names.index('male')