# pydantic models
from typing import Literal, Any
from pydantic import BaseModel, ConfigDict


ValueType = Literal[
	"string", "number", "integer", "boolean", "object", "array"
]


class SchemaType(BaseModel):
	"""Schema for one function parameter (or its return value)."""
	type: ValueType


class PromptItem(BaseModel):
	"""One natural-language prompt from function_calling_tests.json."""
	model_config = ConfigDict(extra="forbid")

	prompt: str


class FunctionDef(BaseModel):
	"""One callable function from functions_definition.json."""
	model_config = ConfigDict(extra="forbid")

	name: str
	description: str
	parameters: dict[str, SchemaType]
	returns: SchemaType | None = None


class OutputResult(BaseModel):
	"""One entry of the output JSON as a resolved function call."""
	model_config = ConfigDict(extra="forbid")

	prompt: str
	name: str
	parameters: dict[str, Any]
