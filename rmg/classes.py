
from itertools import combinations
from dataclasses import dataclass
import os

from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
import networkx as nx
import numpy as np

from rmg import bonding
from rmg import utils


class MolecularGraph(nx.Graph):
    def __init__(self):
        super(MolecularGraph, self).__init__()

    def init_graph(self, atom_dict, **kwargs):
        super(MolecularGraph, self).__init__()
        # Use the current time in milliseconds as a random number seed
        self.atom_dict = atom_dict
        node_list = list()
        np.random.seed(None)
        for symbol, quantity in atom_dict.items():
            # Add suffixes to each atom symbol
            atom_list = ["{}{}".format(symbol, suffix) for suffix in range(quantity)]
            # For each unique atom, add a node
            nodes = [self.add_node(atom) for atom in atom_list]
            node_list.extend(atom_list)
        self.bond_dict = {atom: 1 for atom in node_list}
        # Generate the initial graph by looping over each node and adding
        # a random connection
        for index, nodeA in enumerate(node_list):
            # Make a copy of the nodes that are not the current node
            temp = [node for node in node_list if node != nodeA]
            bond_count = [self.bond_dict[atom] for atom in temp]
            norm_p = bonding.inverse_weighting(np.array(bond_count))
            nodeB = np.random.choice(temp, p=norm_p)
            # Track a bond for both nodes
            self.bond_dict[nodeB] += 1
            self.bond_dict[nodeA] += 1
            # Start with single bonds
            self.add_edge(nodeA, nodeB, weight=1)
        # Check for disconnected graphs; if there are any subgraphs,
        # randomly connect a pair of nodes to make the graph connected
        self.connection_check()
        # Randomly add bonds between nodes
        self.add_bonds()
        # Add hydrogens if they're not explicitly provided
        if "H" not in self.atom_dict:
            bonding.fill_hydrogens(self)
        self.generate_coords(**kwargs)

    def __copy__(self):
        """
        Copy method for MolecularGraph. This gets called by the networkx connected subgraphs
        function, and will break if it doesn't exist.
        :return:
        """
        return MolecularGraph(self.atom_dict)

    def __eq__(self, other):
        """
        Check if another molecular graph is equal to the current one.
        :param other: MolecularGraph object
        :return: True if the other graph is isomorphic
        """
        weight_check = nx.algorithms.isomorphism.numerical_edge_match("weight", [1, 2, 3, 4])
        return nx.is_isomorphic(self, other)

    def connection_check(self):
        """
        Check if a graph is disconnected, and if it is, add edges between
        randomly chosen nodes. The probability of a node being chosen is
        inverse to the number of bonds it has.
        """
        if nx.is_connected(self) is False:
            subgraphs = list(nx.connected_component_subgraphs(self, copy=True))
            # Loop over combinations of subgraphs - this makes it general
            # if there are more than two subgraphs
            for sub_pair in combinations(subgraphs, 2):
                node_choices = list()
                for subgraph in sub_pair:
                    # Pick a random node, weighted towards nodes with fewer bonds
                    bond_orders = np.array([bonding.sum_bonds(subgraph, node) for node in subgraph.nodes])
                    weights = bonding.inverse_weighting(bond_orders)
                    node_choices.append(
                        np.random.choice(subgraph.nodes, p=weights)
                    )
                # Connect up the two nodes
                self.add_edge(*node_choices, weight=1)

    def add_bonds(self):
        """
        Function for randomly adding bonds to an existing graph. The idea is
        to loop over each node, and assign a random number of maximum bonds
        it is allowed to have.
        """
        atom_maxes = {
            "C": 4,
            "O": 2,
            "S": 3,
            "N": 3,
        }
        for nodeA in self.nodes:
            # Ignore hydrogens
            if "H" not in nodeA:
                atom = nodeA[0]
                neighbors = list(nx.neighbors(self, nodeA))
                nodeA_sum, neighbor_sum = bonding.node_sums(self, nodeA)
                norm_weights = bonding.inverse_weighting(neighbor_sum)
                # Set a random number of bonds, with the limit being
                # set by the atom type.
                max_bonds = np.random.randint(1, atom_maxes[atom])
                # This condition is fulfilled when we have bonds to add and
                # have neighbors that can accept bonds
                while nodeA_sum < max_bonds and np.sum(norm_weights) != 0:
                    nodeB = np.random.choice(neighbors, p=norm_weights)
                    self[nodeA][nodeB]["weight"] += 1.
                    nodeA_sum, neighbor_sum = bonding.node_sums(self, nodeA)
                    norm_weights = bonding.inverse_weighting(neighbor_sum)

    def get_distances(self, node, coords):
        """
        Calculate the distances between a given node and its neighbors
        :param node: str identifier for node
        :param coords: dict containing the coordinates for each node
        :return distances: list with distances between the node and each neighbor
        """
        neighbors = nx.neighbors(self, node)
        distances = [bonding.calc_distance(coords[node], coords[neighbor]) for neighbor in neighbors]
        return distances

    def graph2xyz(self, filepath, comment):
        """
        Export the coordinates of this graph to an XYZ file.
        :param filepath:
        :param comment:
        :return:
        """
        xyz = bonding.pos2xyz(self.coords, comment)
        with open(filepath, "w+") as write_file:
            write_file.write(xyz)

    def generate_coords(self, **kwargs):
        # generate the initial coordinates
        spring_settings = {
            "k": 1. / np.sqrt(len(self)),
            "dim": 3,
            "iterations": 20,
            "scale": len(self) / 6.,
            "weight": None
        }
        spring_settings.update(kwargs)
        #success = False
        #while success is False:
        coords = nx.spring_layout(self, **spring_settings)
        #    line_check = list()
        #    for node in self.nodes:
        #        distances = self.get_distances(node, coords)
        #        line_check.append(
        #            all([0.6 < distance < 1.6 for distance in distances])
        #        )
        #    if all(line_check) is True:
        #        success = True
        self.coords = coords

    def optimize_coords(self, coords):
        """
        Optimize the coordinates following the approximate generation with nx.spring_layout.
        This implements a rudimentary Newton-Raphson optimization of the coordinates, based on
        harmonic spring energies that connect
        :param coords:
        :param max_it:
        :param delta:
        :param verbose:
        :return:
        """
        coords = np.array([coords[node] for node in self.nodes])
        result = minimize(
            self.graph_energy,
            #args=(coords),
            x0=coords,
            options={"disp": True},
            method="BFGS",
        )
        new_coords = result["x"]
        new_coords = np.reshape(new_coords, (len(self), 3))
        new_coords = {node: new_coords[index] for index, node in enumerate(self.nodes)}
        print(new_coords)
        return new_coords

    def graph_energy(self, coords):
        #coords = np.reshape(coords, (len(self), 3))
        energies = [bonding.node_energy_force(node, self, coords) for node in self.nodes]
        return np.sum(energies)


@dataclass
class Batch:
    atom_dict: dict
    niter: int
    processes: int = 1

    def __post_init__(self):
        self.counter = 0
        self.redundant = 0
        self.graphs = list()

    def _graph_gen(self, arg):
        """
        Private method for generating MolecularGraphs. The idea is to wrap
        this method with a parallel worker to allow concurrent generation.
        :return: MolecularGraph object
        """
        pbar, atom_dict, kwargs = arg
        graph = MolecularGraph()
        graph.init_graph(atom_dict, **kwargs)
        pbar.update(1)
        return graph

    def generate_graphs(self, **kwargs):
        # Progress bar to show that structures are being generated
        self.counter = 0
        self.redundant = 0
        with tqdm(total=self.niter) as pbar:
            with Parallel(n_jobs=self.processes, prefer="threads") as parallel:
                graphs = parallel(
                    delayed(self._graph_gen)(
                        (pbar, self.atom_dict, kwargs)
                    ) for _ in range(self.niter)
                )
        print("Generated {} graphs.".format(len(graphs)))
        # Take the unique graphs. This is necessary because with multiprocessing there is a
        # race condition for determining what graphs have already been generated
        _ = [self.graphs.append(graph) for graph in graphs if graph not in self.graphs]
        print("Generated {} non-redundant graphs.".format(len(self)))
        print("{} of them were redundant.".format(self.niter - len(self.graphs)))

    def __len__(self):
        return len(self.graphs)

    def to_pickle(self, filepath="rmg_batch.pkl"):
        """
        Save the Batch object to disk.
        :param filepath: str path
        """
        utils.save_obj(self, filepath)
        print("Saved Batch to {}".format(filepath))

    def export_graphs(self, root="../data/raw/xyz"):
        """
        Dump the molecular graphs to a root directory.
        :param root: str path to a root directory for dumping
        """
        if os.path.isdir(root) is False:
            os.mkdir(root)
        for index, graph in tqdm(enumerate(self.graphs)):
            dest = os.path.join(root, "{}.xyz".format(index))
            graph.graph2xyz(
                dest,
                "Structure {}".format(index)
            )
