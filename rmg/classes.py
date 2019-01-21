
import networkx as nx
import numpy as np
from itertools import combinations
from dataclasses import dataclass
from tqdm.autonotebook import tqdm
import os

from rmg import bonding
from rmg import utils


class MolecularGraph(nx.Graph):
    def __init__(self):
        super(MolecularGraph, self).__init__()

    def init_graph(self, atom_dict):
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
        # Add hydrogens
        bonding.fill_hydrogens(self)
        size = len(self)
        self.coords = nx.spring_layout(self, k=0.2, dim=3, iterations=100, scale=size / 5.5, weight="weight")

    def __copy__(self):
        return MolecularGraph(self.atom_dict)

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
        for nodeA in self.nodes:
            neighbors = list(nx.neighbors(self, nodeA))
            nodeA_sum, neighbor_sum = bonding.node_sums(self, nodeA)
            norm_weights = bonding.inverse_weighting(neighbor_sum)
            max_bonds = np.random.randint(1, 4)
            # This condition is fulfilled when we have bonds to add and
            # have neighbors that can accept bonds
            while nodeA_sum < max_bonds and np.sum(norm_weights) != 0:
                nodeB = np.random.choice(neighbors, p=norm_weights)
                self[nodeA][nodeB]["weight"] += 1.
                nodeA_sum, neighbor_sum = bonding.node_sums(self, nodeA)
                norm_weights = bonding.inverse_weighting(neighbor_sum)

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


@dataclass
class Batch:
    atom_dict: dict
    ngraphs: int
    max_iter: int = 1000

    def __post_init__(self):
        self.graphs = list()
        # Make sure we do more iterations than the specified
        # number of graphs
        if self.ngraphs > self.max_iter:
            self.max_iter = self.ngraphs + 100
        index = 0
        iterations = 0
        # Progress bar to show that structures are being generated
        with tqdm(total=self.max_iter) as pbar:
            # While loop will ensure that the generator keeps
            # running until either the requested number of graphs are
            # created, or if we've exceeded the maximum number of iterations
            while index < self.ngraphs and iterations < self.max_iter:
                graph = MolecularGraph()
                graph.init_graph(self.atom_dict)
                if graph not in self.graphs:
                    self.graphs.append(graph)
                    index += 1
                iterations += 1
                pbar.update(1)
        print("Generated {} graphs.".format(len(self)))

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
