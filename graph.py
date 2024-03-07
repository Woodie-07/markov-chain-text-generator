import random
from itertools import count
import json
import re

def split_phrase(phrase):
    return re.findall(r'\b\w+\b|\s|\S', phrase)

vowels = 'AEIOU'
consts = 'BCDFGHJKLMNPQRSTVWXYZ'
consts = consts + consts.lower()
vowels = vowels + vowels.lower()

def is_vowel(letter):
    return letter in vowels 

def is_const(letter):
    return letter in consts

# get the syllables for vc/cv
# stolen from the internet
def vc_cv(word):
    segment_length = 4 # because this pattern needs four letters to check
    pattern = [is_vowel, is_const, is_const, is_vowel] # functions above
    split_points = []

    # find where the pattern occurs
    for i in range(len(word) - segment_length):
        segment = word[i:i+segment_length]

        # this will check the four letter each match the vc/cv pattern based on their position
        # if this is new to you I made a small note about it below
        if all([fi(letter) for letter, fi in zip(segment, pattern)]):
            split_points.append(i+segment_length/2)

    # use the index to find the syllables - add 0 and len(word) to make it work
    split_points.insert(0, 0)
    split_points.append(len(word))
    syllables = []
    for i in range(len(split_points) - 1):
        start = int(split_points[i])
        end = int(split_points[i+1])
        syllables.append(word[start:end])
    return syllables

class Node:
    def __init__(self, token: str) -> None:
        self.token = token
        self.begin = False
        self.end = False
        self.edges = []

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return False
        
        if (self.begin and __value.begin) or (self.end and __value.end):
            return True
        
        if not self.begin and not self.end and self.token == __value.token:
            return True
        
    def __hash__(self) -> int:
        return hash((self.token, self.begin, self.end))

class Edge:
    def __init__(self, target: Node) -> None:
        self.target = target
        self.count = 1

class Graph:
    def __init__(self) -> None:
        begin_node = Node(None)
        begin_node.begin = True
        self.begin = begin_node
        self._index = {}

    def _save_nodes(self, nodes, edges, cur_node, ic, jc, visited):
        i = next(ic)
        for node in visited:
            if visited[node] == cur_node:
                return node
        edge_ids = []
        visited[i] = cur_node
        nodes[i] = {
            "token": cur_node.token,
            "begin": cur_node.begin,
            "end": cur_node.end,
            "edges": edge_ids
        }
        for edge in cur_node.edges:
            j = next(jc)
            edge_ids.append(j)
            edges[j] = {
                "target": self._save_nodes(nodes, edges, edge.target, ic, jc, visited),
                "count": edge.count
            }

        return i

    def save(self, filename: str) -> None:
        nodes = {}
        edges = {}
        graph = {
            "nodes": nodes,
            "edges": edges
        }
        self._save_nodes(nodes, edges, self.begin, count(), count(), {})
        print(graph)

        with open(filename, "w") as f:
            json.dump(graph, f)

    def load(self, filename: str) -> None:
        with open(filename, "r") as f:
            data = json.load(f)

        nodes_d = data["nodes"]
        edges_d = data["edges"]

        edges = {}
        placeholders = {}
        for edge in edges_d:
            edgeid = int(edge)
            edge = edges_d[edge]
            _edge = Edge(edge["target"])
            _edge.count = edge["count"]
            if edge["target"] not in placeholders:
                placeholders[edge["target"]] = []
            
            placeholders[edge["target"]].append(_edge)
            edges[edgeid] = _edge

        for node in nodes_d:
            nodeid = int(node)
            node = nodes_d[node]
            _node = Node(node["token"])
            _node.begin = node["begin"]
            _node.end = node["end"]
            for edge in node["edges"]:
                _node.edges.append(edges[edge])

            if nodeid in placeholders:
                for edge in placeholders[nodeid]:
                    edge.target = _node

            if _node.begin:
                self.begin = _node

            self._index[hash(_node)] = _node


    def _find_node(self, search_node: Node, cur_node: Node, visited: list):
        if cur_node in visited:
            return None
        if cur_node == search_node:
            return cur_node
        
        visited.append(cur_node)
        
        for edge in cur_node.edges:
            result = self._find_node(search_node, edge.target, visited)
            if result:
                return result
            
        return None

    def find_node(self, node: Node):
        if hash(node) in self._index:
            return self._index[hash(node)]
        return None
        #visited = []
        #result = self._find_node(node, self.begin, visited)
        #if result: self._index[hash(result)] = result
        #return result

    def add_node(self, prev_node: Node, cur_token: str, end: bool = False) -> Node:
        for edge in prev_node.edges:
            if (end and edge.target.end) or (edge.target.token == cur_token):
                edge.count += 1
                return edge.target
        else:
            _node = Node(cur_token)
            _node.end = end
            node = self.find_node(_node)
            if not node:
                node = _node
                self._index[hash(node)] = node
            new_edge = Edge(node)
            prev_node.edges.append(new_edge)

        return node
    
    def rand_tokens(self, node: Node):
        if node.end:
            return []
        next_node = random.choices([edge.target for edge in node.edges], [edge.count for edge in node.edges])[0]
        token = next_node.token
        if token is None:
            token = ""
        return [token] + self.rand_tokens(next_node)

graph = Graph()
graph.load("b")
        
generate = True
while True:
    phrase = input("> ").strip()
    if phrase == "save":
        graph.save("b")
        break

    elif phrase == "togglegen":
        generate = not generate
        continue
    
    elif phrase != "":  # 'train' the generator on the input text
        last_node = graph.begin
        
        parts = split_phrase(phrase)
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] == " ":  # if the next token is a space, move the space to the end of this token
                parts[i] += " "
                parts.pop(i + 1)
            
            tokens = [parts[i]]
            tokens = vc_cv(tokens[0])  # able to comment out this line to remove syllable tokenisation
            for token in tokens:
                last_node = graph.add_node(last_node, token)  # iterate through the tokens and add nodes/change weights to 'train' the generator

            i += 1

        graph.add_node(last_node, None, end=True)

    if generate: print("".join(graph.rand_tokens(graph.begin)))  # output some random text from the generator