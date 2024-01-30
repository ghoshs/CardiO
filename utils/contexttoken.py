from spacy.tokens import Token

class ContextToken(object):
	def __init__(self, token: Token):
		self.token = token

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and
				other.token.text == self.token.text and 
				other.token.i == self.token.i)

	def __hash__(self):
		return hash(str(self.token.i) + self.token.text)
