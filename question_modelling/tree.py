from nltk import Tree

def format_node(node):
	return "_".join([node.orth_, node.pos_, node.dep_])


def to_nltk_tree(node):
	if node.n_lefts + node.n_rights > 0:
		return Tree(format_node(node), [to_nltk_tree(child) for child in node.children])
	else:
		return format_node(node)


def dependency_tree(question, ann):
	print("Tree for: ", question)
	print([to_nltk_tree(sent.root).pretty_print() for sent in ann.sents])