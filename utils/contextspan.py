from spacy.tokens import Span, Token

"""
Define Entity object with additional functions: 
	substrOf: return if a Span is substring of provided Token/Span
"""
class ContextSpan(object):
	def __init__(self, span: Span):
		self.span = span

	def __eq__(self, other):
		token_equality, span_equality = False, False
		if isinstance(other, Token):
			token_equality = (self.span.text == other.text and 
				self.span.start == other.i and 
				len(self.span) == 1)
		elif isinstance(other, Span):
			span_equality = (other.text == self.span.text and
				other.start == self.span.start and
				other.end == self.span.end)
		elif isinstance(other, self.__class__):
			span_equality = (other.span.text == self.span.text and
				other.span.start == self.span.start and
				other.span.end == self.span.end)
		else:
			return False
		return token_equality or span_equality

	def __hash__(self):
		return hash(self.span.text + str(self.span.start) + str(self.span.end))

	def substrOf(self, other) -> bool:
		if isinstance(other, Span):
			return self.span.text in other.text and self.span.start >= other.start and self.span.end <= other.end
		elif isinstance(other, self.__class__):
			other = other.span
			return self.span.text in other.text and self.span.start >= other.start and self.span.end <= other.end
		else:
			return False

	def contains(self, other) -> bool:
		if isinstance(other, Span):
			return other.text in self.span.text and other.start >= self.span.start and other.end <= self.span.end
		elif isinstance(other, self.__class__):
			other = other.span
			return other.text in self.span.text and other.start >= self.span.start and other.end <= self.span.end
		else:
			return False
