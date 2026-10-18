project init
	1) uv config: install dependencies, dev dependencies
	2) Makefile
	3) how to make the llm_sdk pyproject run when the evaluator runs uv sync
	4) make uv run python -m src print something and flake8 and mypy pass

data I/O
	1) define models: function definition (name, description, parameters, return type), test prompt, output result
	2) load valid JSON file successfully into the structure
	3) handle argparse to add the file in the terminal if needed

explore SDK
  1) test sdk functions
  2) do basic unconstrained loop
  3) add constraints, one at a time: function name -> number -> string -> boolean

main loop:
	1) read the prompt from the function calling file
	2) hardcode the full prompt: original prompt + instructions with the function definition
	3) encode the prompt into token ids
	4) feed the token ids to the llm to get logits
	5) constrain the logits depending on what's next on the json structure (-infinity)
	6) pick the best token from the valid ones
	7) append it to the struct and repeat til done
	8) validate structure and profit

constrained decoder:
	"expecting {"
	"expecting key"
	"expecting :"
	"expecting value"
	"expecting , or }"
	Handle multiple functions (the model picks the name; you constrain it to valid names)
    Handle multiple parameters
    Handle different types (number, string, boolean)
    Handle the model's choice of which function to call

error handling, tests, speed
